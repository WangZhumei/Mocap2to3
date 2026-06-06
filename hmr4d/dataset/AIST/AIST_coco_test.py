import os
from pathlib import Path
import numpy as np
import pickle
import torch
import cv2
from torch.utils import data
from hmr4d.utils.pylogger import Log
import json
import decord
from decord import cpu, gpu
from scipy.spatial.transform import Rotation as R
import math
from hmr4d.dataset.HumanML3D.utils import upsample_motion
import hmr4d.utils.matrix as matrix
import random

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayf2az, project_p2d, cvt_to_bi01_p2d,compute_T_eay2ayfz,compute_T_move,compute_T_move2,compute_T_rotY,compute_T_move,compute_T_move2,compute_T_move0
import imageio
from hmr4d.utils.camera_utils import get_camera_mat_zface,get_camera_mat_zface_wzm, get_gt2genCamera4_90_mat_wzm,get_gt2genCamera4_Y_mat_wzm,cartesian_to_spherical
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch,adjust_K
from hmr4d.network.evaluator.word_vectorizer import WordVectorizer
from hmr4d.dataset.motionx.utils import normalize_kp_2d, normalize_kp_2d_linear,adjust_K, estimate_focal_length, generate_camera_intrinsics

#from hmr4d.utils.vis_open3d import Model,ModelPose2
class Dataset(data.Dataset):
    def __init__(
        self,
        root="inputs/AIST",
        split="train",
        length=120,  # target motion length
        start_frame_interval=15,  # sample interval
        limit_size=None,
        per_shift=0.1,
        per_zoomout=0.2,
        #is_pinhole=True,
        min_motion_time=2,
        max_motion_time=10,
        max_text_len=20,
        is_ignore_transl=True,
        unit_length=4,
        is_root_next=True,
        is_pinhole=True,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        is_notext=False,
        eleva_angle=None,
        random_cam=True,
        train_fps=30,
        distance=4.5,
        patch_size=224,
        is_uniform_views=True,
        N_views=4,
        img_H=224,
        img_W=224,
        pm_W=224,
        is_cam_rel2human=False,
        **kwargs,
    ):
        super().__init__()
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_pinhole = is_pinhole
        self.is_root_next = is_root_next
        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text
        self.is_notext = is_notext
        self.eleva_angle = eleva_angle
        self.random_cam = random_cam
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size
        self.img_H = img_H
        self.img_W = img_W
        self.is_uniform_views = is_uniform_views
        self.N_views = N_views
        self.split = split
        self.is_cam_rel2human = is_cam_rel2human
        self.pm_W = pm_W

        self.aist_joints3d_coco = torch.load("inputs/AIST/joint3d_coco.pth")
        self.pose2d_detector = torch.load("inputs/AIST/pose_test_vit.pth")
        self.aist_cam = torch.load("hmr4d/dataset/AIST/resource/aist_cam.pth")

        self.idx2meta = self.prepare_meta()
        self.pointmaps = self.get_pointmaps()
        # Options
        self.limit_size = limit_size  # common usage: making validation faster
        self.args_augbbx = {"per_shift": per_shift, "per_zoomout": per_zoomout}

        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        if self.is_pinhole:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
        else:
            self.K = torch.tensor(
                [
                    [1.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, 1.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ]
            )

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    
    def __getitem__(self, idx):
        meta = self.idx2meta[idx]
        seq_name, cam_name,start_frame, end_frame, start_cam = meta 
        joints_pos_coco = self.aist_joints3d_coco[seq_name].clone().float() 
        
        key_vit = seq_name.replace('cAll','c'+str(start_cam+1).zfill(2))
        i_motion2d_det = torch.tensor(np.array(self.pose2d_detector[key_vit])[:,:,:2])
        if joints_pos_coco.shape[0]!=i_motion2d_det.shape[0]:
            joints_pos_coco = joints_pos_coco[:i_motion2d_det.shape[0]]

        ground_gap = 0.63
        joints_pos_coco[:,:,1] = joints_pos_coco[:,:,1]-ground_gap

        joints_pos_coco = joints_pos_coco[start_frame:end_frame]  
        
        data_fps = 60
        if self.train_fps != data_fps:
            joints_pos_coco = upsample_motion(joints_pos_coco, data_fps, self.train_fps)
        caption = ""

        tokens = ["sos/OTHER"]  + ["eos/OTHER"]
        sent_len = len(tokens)
        tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if isinstance(joints_pos_coco, np.ndarray):
            joints_pos_coco = torch.tensor(joints_pos_coco, dtype=torch.float32)

        
        N_views = self.N_views
        

        Ts_w2c, Ks,Ks_ori,Ks_root,Zreos_root,Ks_global = [], [], [], [], [], []
        normed_motion2ds = []
        normed_motion2ds_global = []
        angle,eleva_angle,z_angle,distance,x_t,y_t,z_t = [], [], [], [], [], [], []
        cam_keys = self.aist_cam[cam_name]
        cam_Ekeys = cam_keys[start_cam]['cam_Eay']
        T_ayfz2eay = cam_keys[start_cam]['T_ayfz2eay'].float()
        joints_pos_c = apply_T_on_points(joints_pos_coco.clone(), T_ayfz2eay)  #
        joints_pos_c, ori_joints_pos_c = self._process_motion_coco(joints_pos_c)

        length = joints_pos_c.shape[0]
        pointmaps = []

        F, J, _ = joints_pos_c.shape
        _, J_ori, _ = ori_joints_pos_c.shape
        for cam_idx in range(len(cam_Ekeys)):

            cam_key = list(cam_Ekeys.keys())[cam_idx]
            
            T_w2c = cam_Ekeys[cam_key]['T_w2c']  
            K = cam_Ekeys[cam_key]['K'].float()  
            gt_T_Eayfz2c = cam_Ekeys[cam_key]['gt_T_Eayfz2c'].float() 


            pointmap_info = self.pointmaps[cam_name][start_cam][cam_key]
            map_ = pointmap_info['map']
            map_idx = pointmap_info['idx']
            pointmap_refi = torch.zeros((self.pm_W*self.pm_W,3))
            pointmap_refi[map_idx,:] = map_.float()
            pointmap_refi = pointmap_refi.reshape(self.pm_W,self.pm_W,3)
            pointmaps.append(pointmap_refi)


            #cam_cond
            cam_mat_ =torch.inverse(gt_T_Eayfz2c).numpy()
            r = R.from_matrix(cam_mat_[:3, :3])
            pitch, roll ,yaw= r.as_euler('YXZ', degrees=False)
            Z_r = yaw
            Y_r = pitch
            X_r = roll
            angle.append(Y_r)
            eleva_angle.append(X_r)
            z_angle.append(Z_r)
            x_t.append(cam_mat_[0,-1])
            y_t.append(cam_mat_[1,-1])
            z_t.append(cam_mat_[2,-1])

            c_motion_c = matrix.get_relative_position_to(joints_pos_c, torch.inverse(gt_T_Eayfz2c)[None]) 
            i_motion2d_c = project_p2d(c_motion_c, K[None], is_pinhole=self.is_pinhole) 
                
            #detector
            if cam_idx ==0:
                key_vit = seq_name.replace('cAll','c'+str(start_cam+1).zfill(2))
                vit_pose2d = torch.tensor(np.array(self.pose2d_detector[key_vit])[:,:,:2])
                vit_pose2d = vit_pose2d[start_frame:end_frame]  
                data_fps = 60
                if self.train_fps != data_fps:
                    vit_pose2d = upsample_motion(vit_pose2d, data_fps, self.train_fps)
                root_ = (vit_pose2d[:, 11:12,:]+vit_pose2d[:, 12:13,:])/2
                vit_pose2d = torch.cat([root_,vit_pose2d ], dim=1)
                vit_pose2d = torch.tensor(vit_pose2d).type_as(i_motion2d_c)
                vit_pose2d = self._process_motion2d(vit_pose2d)
                i_motion2d_c = vit_pose2d

           
            #global
            i_motion2d_global_c = i_motion2d_c.clone()
            i_motion2d_global_c[:,:,1] = i_motion2d_global_c[:,:,1]+(self.img_W-self.img_H)//2 
            K_global = K.clone()
            K_global[1,-1] = K_global[1,-1]+(self.img_W-self.img_H)//2 
            if self.is_pinhole:
                normed_motion2d_global_c = normalize_keypoints_to_patch(i_motion2d_global_c, crop_size=self.img_W)
            else:
                normed_motion2d_global_c = i_motion2d_global_c
            normed_motion2d_c,_,bbox_c =normalize_kp_2d_linear(i_motion2d_c) 
            K_adj_c = adjust_K(K[None].repeat(bbox_c.size()[0], 1, 1) , bbox_c)
            w_h_c = normalize_keypoints_to_patch(bbox_c[:,2] * 200, crop_size=self.img_W)[:,None,None]
            w_h_c = torch.cat([w_h_c,w_h_c],dim=-1)
            #cat w h
            normed_motion2d_c = torch.cat([normed_motion2d_c,w_h_c],dim=-2) 
            normed_motion2d_global_c = torch.cat([normed_motion2d_global_c,w_h_c],dim=-2) 
            normed_motion2d_c[...,0,:] = normed_motion2d_global_c[...,0,:]
            
            Ts_w2c.append(gt_T_Eayfz2c)
            Ks.append(K_adj_c)
            normed_motion2ds.append(normed_motion2d_c)
            Ks_global.append(K_global)
            normed_motion2ds_global.append(normed_motion2d_global_c)
        pointmaps = torch.stack(pointmaps)
        
        Ts_w2c = torch.stack(Ts_w2c)
        Ks = torch.stack(Ks)
        Ks_global = torch.stack(Ks_global)
        normed_motion2ds = torch.stack(normed_motion2ds)
        normed_motion2ds_global = torch.stack(normed_motion2ds_global)
        cam_mat = torch.inverse(Ts_w2c)
        angle = torch.tensor(angle)
        eleva_angle = torch.tensor(eleva_angle)
        z_angle = torch.tensor(z_angle)
        x_t = torch.tensor(x_t)
        y_t = torch.tensor(y_t)
        z_t = torch.tensor(z_t)

        
        J_2D = normed_motion2ds.shape[-2]
        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            normed_motion2ds = torch.cat([normed_motion2ds, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            normed_motion2ds_global = torch.cat([normed_motion2ds_global, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            ori_joints_pos_c = torch.cat([ori_joints_pos_c, torch.zeros((pad_length, J_ori, 3))], dim=0)

        spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat)) 
        theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]
        if self.is_cam_rel2human:
            d_T = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=-1) 
        else:
            d_theta = theta - theta[:1]
            d_azimuth = (azimuth - azimuth[:1]) % (2 * torch.pi) 
            d_z = z - z[:1]  # N, 1
            d_T = torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1) 
        
        RT_emb = torch.cat([eleva_angle[:,None],z_angle[:,None],y_t[:,None]], dim=-1)
        K_emb = torch.zeros((N_views,4), dtype=torch.float32)
        K_emb[:,0] = Ks_global[:,0,0]
        K_emb[:,1] = Ks_global[:,1,1]
        K_emb[:,2] = Ks_global[:,0,2]
        K_emb[:,3] = Ks_global[:,1,2]
        K_emb = K_emb/3500
        # Return
        return_data = {
            "length": length, 
            "gt_motion2d": normed_motion2ds.float(), 
            "gt_motion2d_global": normed_motion2ds_global.float(),  
            "gt_motion": ori_joints_pos_c.float(),  
            "T_w2c": Ts_w2c.float(), 
            "is_pinhole": self.is_pinhole, 
            "Ks": Ks_global.float(),  
            "pointmap": pointmaps.float(), 
            "patch_size": self.img_W, 
            "cam_emb": d_T.float(), 
            "RT_emb": RT_emb.float(),  
            "K_emb": K_emb.float(),  
            "word_embs": word_embeddings.astype(np.float32),
            "pos_onehot": pos_one_hots.astype(np.float32),
            "text": caption,
            "text_len": sent_len,
            "task": "3D",
            "data_info": seq_name+'/'+str(start_cam+1),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        return return_data
    
    def get_pointmaps(self):
        pointmaps = {}
        for seq_name in self.aist_cam.keys():
            pointmaps[seq_name] = {}
            cam_keys = self.aist_cam[seq_name]
            for cam0 in cam_keys.keys():
                _cam = torch.load('inputs/AIST/WW_pointmap224/'+str(seq_name)+'_cam'+str(cam0)+'.pth')
                pointmaps[seq_name][cam0] =_cam
        return pointmaps 
    
    def prepare_meta(self):
        """
        Pre-processing the sequences. Each sequences will be devided to several segements with the length of
        `length` and intevel of 0.5s. And flatten the segements by pushing to `meta` list.
        """
        meta = []
        #rich_fps = 30
        cam_mapping = 'inputs/AIST/mapping.txt'
        max_motion_len = self.max_motion_time  * self.train_fps*2
        min_motion_len = self.min_motion_time * self.train_fps*2
        cam_map_dict = {}
        with open(cam_mapping,'r',) as ins:
           cam_mp = ins.read()
           cam_mp = cam_mp.split('\n')#delete \n 
           print('con top:',cam_mp.pop())
        ins.close()
        for cam_m in cam_mp:
            temp = cam_m.split(' ')
            cam_map_dict[temp[0]] = temp[1]
        for seq_name in self.aist_joints3d_coco.keys():
            cam_name = cam_map_dict[seq_name]
            motion= self.aist_joints3d_coco[seq_name] #start end
            s_motion = 0
            e_motion = motion.shape[0]
            slice_ = e_motion//max_motion_len
            for i in range(slice_):
                start_ = s_motion+i*max_motion_len
                end_ = start_+max_motion_len
                for start_cam in range(9):
                    key_vit = seq_name.replace('cAll','c'+str(start_cam+1).zfill(2))
                    if key_vit not in self.pose2d_detector.keys():
                       continue
                    meta.append([seq_name,cam_name,start_, end_, start_cam])
            if slice_==0:
                end_ = 0
            if e_motion-end_>=min_motion_len:
                for start_cam in range(9):
                    key_vit = seq_name.replace('cAll','c'+str(start_cam+1).zfill(2))
                    if key_vit not in self.pose2d_detector.keys():
                       continue
                    meta.append([seq_name,cam_name,end_, e_motion, start_cam])
        return meta
    
    def _process_motion_coco(self, joints_pos):
        # if self.train_fps != 20:
        #     joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        
        #coco add root
        root_ =(joints_pos[:, 11:12,:]+joints_pos[:, 12:13,:])/2
        joints_pos = torch.cat([root_,joints_pos ], dim=1)  # F, 1+J, C

        return joints_pos, ori_joints_pos
    def _process_motion2d(self, joints_pos2d):
        # if self.train_fps != 20:
        #     joints_pos2d = upsample_motion(joints_pos2d, 20, self.train_fps)
        
        return joints_pos2d
    
