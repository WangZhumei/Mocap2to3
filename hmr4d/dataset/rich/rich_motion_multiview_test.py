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
from .rich_utils import (
    get_cam2params,
    remove_extra_rules,
    squared_crop_and_resize,
    sample_idx2meta,
    remove_bbx_invisible_frame,
    get_w2az_sahmr,
    parse_seqname_info,
    parse_seqname_info_test,
    get_seqnames_of_split,
    get_seq_cam_fn,
    get_img_fn,
    get_img_key,
    get_augmented_square_bbx,
)
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
from hmr4d.utils.text_stub import build_dummy_text_features
from hmr4d.dataset.motionx.utils import normalize_kp_2d, normalize_kp_2d_linear,adjust_K, estimate_focal_length, generate_camera_intrinsics
from hmr4d.utils.net_utils import trusted_torch_load

class Dataset(data.Dataset):
    def __init__(
        self,
        root="inputs/RICH",
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
        self.seqnames_split = get_seqnames_of_split(split, skip_multi_persons=True)  # list of seqnames
        self.pm_W = pm_W

        self.scene_info_root = Path(root) / "joints3d" / "scene_info"
        seqname_info = parse_seqname_info_test(skip_multi_persons=True)  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}
        self.seqname_to_camids = {k: v[-1] for k, v in seqname_info.items()}  # ids are [int]

        self.rich_joints3d = np.load("inputs/RICH/hmr4d_support/joints3d/joints3d_smpl_v2.npy", allow_pickle=True).item()
        self.idx2meta = self.prepare_meta()
        self.cam2params = get_cam2params(self.scene_info_root)  # cam_key -> (T_w2c, K) 
        self.w2az = get_w2az_sahmr()  # scan_name -> T_w2az, w-coordinate refers to cam-1-coordinate
        self.rich_cam = trusted_torch_load("hmr4d/dataset/rich/resource/rich_cam.pth")
        self.pointmaps = self.get_pointmaps()
        # Options
        self.limit_size = limit_size  # common usage: making validation faster
        self.args_augbbx = {"per_shift": per_shift, "per_zoomout": per_zoomout}

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
        # self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        self.w_vectorizer = None

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    
    def __getitem__(self, idx):
        meta = self.idx2meta[idx]
        seq_name, s_start,start_frame, end_frame, cam_keys = meta 
        joints_pos = self.rich_joints3d[seq_name][0].copy() 
        mid_offset = self.rich_joints3d[seq_name][1]  
        joints_pos = torch.from_numpy(joints_pos)  
        scan_name = self.seqname_to_scanname[seq_name] 
        T_w2az = self.w2az[scan_name]  
        motion_az = apply_T_on_points(joints_pos, T_w2az)  
        T_az2ayfz = compute_T_ayf2az(motion_az[s_start - mid_offset][None], inverse=True)[0] 
        joints_pos = apply_T_on_points(motion_az, T_az2ayfz)  #

        mid_s, mid_e = start_frame - mid_offset, end_frame - mid_offset
        joints_pos = joints_pos[mid_s:mid_e]  
        max_motion_len = (self.max_motion_time - 4) * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        length, J = joints_pos.shape[:2]

        if length > max_motion_len:
            start = np.random.randint(0, length - max_motion_len)
            end = start + max_motion_len
            joints_pos = joints_pos[start:end] 
            length = joints_pos.shape[0]
        

        caption = ""

        # tokens = ["sos/OTHER"]  + ["eos/OTHER"]
        # sent_len = len(tokens)
        # tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        # pos_one_hots = []
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_embeddings, pos_one_hots, sent_len = build_dummy_text_features(self.max_text_len)

        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)

        N_views = self.N_views
        
        Ts_w2c, Ks,Ks_ori,Ks_root,Zreos_root,Ks_global = [], [], [], [], [], []
        normed_motion2ds = []
        normed_motion2ds_global = []
        angle,eleva_angle,z_angle,distance,x_t,y_t,z_t = [], [], [], [], [], [], []
        T_w2c,_ = self.cam2params[cam_keys[0]]
        gt_T_ayfz2c = T_w2c @ T_w2az.inverse() @ T_az2ayfz.inverse()
        T_ayfz2eay = compute_T_eay2ayfz(torch.inverse(gt_T_ayfz2c), inverse=True)[0]  # (4, 4)
        joints_pos = apply_T_on_points(joints_pos, T_ayfz2eay)  #
        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        length = joints_pos.shape[0]
        pointmaps = []


        F, J, _ = joints_pos.shape
        _, J_ori, _ = ori_joints_pos.shape
        for cam_key in cam_keys:
            cam_id = int(cam_key.split("_")[1])  
            T_w2c, K = self.cam2params[cam_key]  
            gt_T_ayfz2c = T_w2c @ T_w2az.inverse() @ T_az2ayfz.inverse() @ T_ayfz2eay.inverse()


            pointmap_info = self.pointmaps[seq_name][cam_keys[0]][cam_key]
            map_ = pointmap_info['map']
            map_idx = pointmap_info['idx']
            pointmap_refi = torch.zeros((self.pm_W*self.pm_W,3))
            pointmap_refi[map_idx,:] = map_
            pointmap_refi = pointmap_refi.reshape(self.pm_W,self.pm_W,3)
            pointmaps.append(pointmap_refi)


            #cam_cond
            cam_mat_ =torch.inverse(gt_T_ayfz2c).numpy()
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

            c_motion = matrix.get_relative_position_to(joints_pos, torch.inverse(gt_T_ayfz2c)[None]) 
            i_motion2d = project_p2d(c_motion, K[None], is_pinhole=self.is_pinhole) 
            i_motion2d_global = i_motion2d.clone()
            i_motion2d_global[:,:,1] = i_motion2d_global[:,:,1]+(self.img_W-self.img_H)//2
            K_global = K.clone()
            K_global[1,-1] = K_global[1,-1]+(self.img_W-self.img_H)//2 
            if self.is_pinhole:
                normed_motion2d_global = normalize_keypoints_to_patch(i_motion2d_global, crop_size=self.img_W)
            else:
                normed_motion2d_global = i_motion2d_global

            normed_pred_motion2d,_,bbox =normalize_kp_2d_linear(i_motion2d) 
            K_adj = adjust_K(K[None].repeat(bbox.size()[0], 1, 1) , bbox)

            w_h = normalize_keypoints_to_patch(bbox[:,2] * 200, crop_size=self.img_W)[:,None,None]
            w_h = torch.cat([w_h,w_h],dim=-1)
            normed_pred_motion2d = torch.cat([normed_pred_motion2d,w_h],dim=-2)
            normed_motion2d_global = torch.cat([normed_motion2d_global,w_h],dim=-2)


            Ts_w2c.append(gt_T_ayfz2c)
            Ks.append(K_adj)
            normed_motion2ds.append(normed_pred_motion2d)
            Ks_global.append(K_global)
            normed_motion2ds_global.append(normed_motion2d_global)
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

        
        normed_motion2ds[:,:,0,:] = normed_motion2ds_global[:,:,0,:]
        J_2D = normed_motion2ds.shape[-2]
        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            normed_motion2ds = torch.cat([normed_motion2ds, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            normed_motion2ds_global = torch.cat([normed_motion2ds_global, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat))  # N, 3
        theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]
        if self.is_cam_rel2human:
            d_T = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=-1)  # N, 4
        else:
            # delta = target - condition
            d_theta = theta - theta[:1]  # N, 1
            d_azimuth = (azimuth - azimuth[:1]) % (2 * torch.pi)  # N, 1
            d_z = z - z[:1]  # N, 1
            d_T = torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1)  # N, 4
        
        RT_emb = torch.cat([eleva_angle[:,None],z_angle[:,None],y_t[:,None]], dim=-1) #torch.Size([4]) #4,6
        K_emb = torch.zeros((N_views,4), dtype=torch.float32) #torch.Size([300, 4])
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
            "gt_motion": ori_joints_pos.float(),  
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
            "data_info": seq_name+'/'+cam_keys[0],
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        return return_data
    
    def get_pointmaps(self):
        pointmaps = {}
        for seq_name in self.rich_cam.keys():
            seq_cam = trusted_torch_load('inputs/RICH/WW_pointmap'+str(self.pm_W)+'/'+str(seq_name)+'.pth')
            pointmaps[seq_name] =seq_cam
        return pointmaps
    
    
    def prepare_meta(self):
        #extra add
        skip_cam = ['ParkingLot2_017_burpeejump2_4','ParkingLot2_017_burpeejump1_4','ParkingLot2_017_overfence2_2','ParkingLot2_017_eating1_3','ParkingLot2_017_pushup2_3','Gym_011_cooking1_2','Gym_011_cooking2_6','Gym_012_cooking2_1']
        meta = []
        #rich_fps = 30
        max_motion_len = (self.max_motion_time - 4) * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        for seq_name in self.rich_joints3d.keys():
            if seq_name not in self.seqnames_split:
                continue
            capname = seq_name.split("_")[0]
            motion, s_motion, e_motion = self.rich_joints3d[seq_name] #start end
            slice_ = (e_motion-s_motion)//max_motion_len
            for i in range(slice_):
                start_ = s_motion+i*max_motion_len
                end_ = start_+max_motion_len
                for start_cam in range(len(self.seqname_to_camids[seq_name])):
                    if seq_name+'_'+str(self.seqname_to_camids[seq_name][start_cam]) in skip_cam:
                        continue
                    cam = []
                    for now_cam in range(len(self.seqname_to_camids[seq_name])):

                        cam.append(capname+'_'+str(self.seqname_to_camids[seq_name][(now_cam+start_cam)%len(self.seqname_to_camids[seq_name])]))
                    meta.append([seq_name, s_motion,start_, end_, cam])
            if slice_==0:
                end_ =0
            if e_motion-end_>=min_motion_len:
                for start_cam in range(len(self.seqname_to_camids[seq_name])):
                    if seq_name+'_'+str(self.seqname_to_camids[seq_name][start_cam]) in skip_cam:
                        continue
                    cam = []
                    for now_cam in range(len(self.seqname_to_camids[seq_name])):
                        cam.append(capname+'_'+str(self.seqname_to_camids[seq_name][(now_cam+start_cam)%len(self.seqname_to_camids[seq_name])]))
                    meta.append([seq_name, s_motion,end_, e_motion, cam])

        return meta
    def _process_motion(self, joints_pos):
        # if self.train_fps != 20:
        #     joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        
        return joints_pos, ori_joints_pos

    
