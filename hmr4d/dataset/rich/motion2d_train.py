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
from hmr4d.dataset.motionx.utils import normalize_kp_2d, normalize_kp_2d_linear,adjust_K, estimate_focal_length, generate_camera_intrinsics
from hmr4d.utils.net_utils import trusted_torch_load


class PointMapTrainDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/RICH",
        split="train",
        limit_size=None,
        min_motion_time=2,
        max_motion_time=10,
        max_text_len=20,
        is_ignore_transl=True,
        unit_length=4,
        is_root_next=False,
        is_pinhole=False,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        is_notext=False,
        eleva_angle=None,
        train_fps=20,
        distance=1.0,
        patch_size=224,
        img_H=224,
        img_W=224,
        pm_W=224,
        N_views=4,
        is_cam_rel2human=False,
        **kwargs,
    ):
        super().__init__()
        self.split = split
        self.root = root
        self.limit_size = limit_size  # common usage: making validation faster
        Log.info(f"Loading rich {split}...")

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
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size
        self.img_H = img_H
        self.img_W = img_W
        self.N_views = N_views
        self.pm_W = pm_W
        
        self.is_cam_rel2human = is_cam_rel2human
        self.seqnames_split = get_seqnames_of_split(split, skip_multi_persons=True)  # list of seqnames
        seqname_info = parse_seqname_info(skip_multi_persons=True)  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}
        self.seqname_to_camids = {k: v[-1] for k, v in seqname_info.items()}  # ids are [int]

        self.rich_joints3d = np.load("inputs/RICH/hmr4d_support/joints3d/joints3d_smpl_v2.npy", allow_pickle=True).item()
        self.idx2meta = self.prepare_meta()
        Log.info(f"Loaded {len(self.idx2meta)} data from {self.root}")
        
        self.w2az = get_w2az_sahmr()  # scan_name -> T_w2az, w-coordinate refers to cam-1-coordinate

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
        self.rich_cam = trusted_torch_load("hmr4d/dataset/rich/resource/rich_cam.pth")
        #self.rich_cam = torch.load("hmr4d/dataset/rich/resource/rich_cam_sort.pth")
        self.pointmaps = self.get_pointmaps()

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)
    def prepare_meta(self):
        """
        Pre-processing the sequences. Each sequences will be devided to several segements with the length of
        `length` and intevel of 0.5s. And flatten the segements by pushing to `meta` list.
        """
        meta = []
        #max_motion_len = (self.max_motion_time - 4) * self.train_fps
        max_motion_len = self.max_motion_time * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        for seq_name in self.rich_joints3d.keys():
            #if seq_name not in 'LectureHall_010_sidebalancerun1':
            if seq_name not in self.seqnames_split:
                continue
            capname = seq_name.split("_")[0]
            motion, s_motion, e_motion = self.rich_joints3d[seq_name] #start end
            #print(s_motion,e_motion)
            slice_ = (e_motion-s_motion)//max_motion_len
            for i in range(slice_):
                start_ = s_motion+i*max_motion_len
                end_ = start_+max_motion_len
                meta.append([seq_name, s_motion,start_, end_, [f"{capname}_{camid}" for camid in self.seqname_to_camids[seq_name]]])
            if e_motion-end_>=min_motion_len:
                meta.append([seq_name, s_motion,end_, e_motion, [f"{capname}_{camid}" for camid in self.seqname_to_camids[seq_name]]])
                #print(end_,e_motion)
        return meta
    def get_pointmaps(self):
        pointmaps = {}
        for seq_name in self.rich_cam.keys():
            seq_cam = trusted_torch_load('inputs/RICH/WW_pointmap'+str(self.pm_W)+'/'+str(seq_name)+'.pth')
            pointmaps[seq_name] =seq_cam
        return pointmaps 
    #fit ground & fix one gen three & K
    def __getitem__(self, idx):
        meta = self.idx2meta[idx]
        seq_name, s_start,start_frame, end_frame, cam_keys = meta  
        joints_pos = self.rich_joints3d[seq_name][0].copy() 
        mid_offset = self.rich_joints3d[seq_name][1]  
        
        joints_pos = torch.from_numpy(joints_pos)  
        scan_name = self.seqname_to_scanname[seq_name] 
        T_w2az = self.w2az[scan_name]  # (4, 4)
        motion_az = apply_T_on_points(joints_pos, T_w2az)  # (F, 22, 3)
        T_az2ayfz = compute_T_ayf2az(motion_az[s_start - mid_offset][None], inverse=True)[0]  
        joints_pos = apply_T_on_points(motion_az, T_az2ayfz)  #

        mid_s, mid_e = start_frame - mid_offset, end_frame - mid_offset
        joints_pos = joints_pos[mid_s:mid_e]  # (F, 22, 3),
        
        T_0 = compute_T_move0(joints_pos[0][None], inverse=True)  
        joints_pos = apply_T_on_points(joints_pos, T_0)  #

        data_fps = 30
        if self.train_fps != data_fps:
            joints_pos = upsample_motion(joints_pos, data_fps, self.train_fps)
        
        caption = ""
        while True:
            ran_seq_idx = random.randint(0,len(self.rich_cam)-1)
            seq_name = list(self.rich_cam.keys())[ran_seq_idx]
            cam_keys = self.rich_cam[seq_name]
            a_i=0
            while True:
                a_i+=1
                ran_cam_idx = random.randint(0,len(cam_keys)-1)

                cam_name = list(cam_keys.keys())[ran_cam_idx]
                cam_key = cam_keys[cam_name]['cam_Eay']
                T_ayfz2eay = cam_keys[cam_name]['T_ayfz2eay']
                distance_ = cam_keys[cam_name]['distance']
                Y_r = cam_keys[cam_name]['Y_r']
                T_w2c = cam_key[cam_name]['T_w2c']
                K = cam_key[cam_name]['K'] 
                gt_T_Eayfz2c = cam_key[cam_name]['gt_T_Eayfz2c'] 

                pointmap_info = self.pointmaps[seq_name][cam_name][cam_name]
                map_ = pointmap_info['map']
                map_idx = pointmap_info['idx']
                pointmap_refi = torch.zeros((self.pm_W*self.pm_W,3))
                pointmap_refi[map_idx,:] = map_
                pointmap_refi = pointmap_refi.reshape(self.pm_W,self.pm_W,3)
                
                #random rot 3dpose
                if random.random()<0.9:
                    T_rotY = compute_T_rotY(inverse=True)[0] 
                    joints_pos_ = apply_T_on_points(joints_pos, T_rotY) 
                    T_moveXZ = compute_T_move2(distance_,0.5,Y_r,inverse=True)[0] 
                    joints_pos_ = apply_T_on_points(joints_pos_, T_moveXZ)  #
                    joints_pos_ = apply_T_on_points(joints_pos_.clone(), T_ayfz2eay)  #
                else:
                    joints_pos_ = apply_T_on_points(joints_pos.clone(), T_ayfz2eay)  #
                joints_pos_, ori_joints_pos_ = self._process_motion(joints_pos_)
                J = joints_pos_.shape[1]
                J_ori = ori_joints_pos_.shape[1]
                length = joints_pos_.shape[0]


                #cam_cond
                cam_mat_ =torch.inverse(gt_T_Eayfz2c)
                r = R.from_matrix(cam_mat_.numpy()[:3, :3])
                pitch, roll ,yaw= r.as_euler('YXZ', degrees=False)
                Z_r = yaw
                Y_r = pitch
                X_r = roll
                angle = torch.tensor([Y_r])
                eleva_angle = torch.tensor([X_r])
                z_angle = torch.tensor([Z_r])
                cam_mat = cam_mat_[None]
                x_t = cam_mat[0,0,-1][None]
                y_t = cam_mat[0,1,-1][None]
                z_t = cam_mat[0,2,-1][None]
                T_w2c = torch.inverse(cam_mat)[0]  
                c_motion = matrix.get_relative_position_to(joints_pos_, cam_mat)  
                i_motion2d = project_p2d(c_motion, K[None], is_pinhole=self.is_pinhole)  

                pad = 100
                x_0 = (i_motion2d[:,:,0]<0-pad).sum(axis=1)
                x_1 = (i_motion2d[:,:,0]>self.img_W+pad).sum(axis=1)
                y_0 = (i_motion2d[:,:,1]<0-pad).sum(axis=1)
                y_1 = (i_motion2d[:,:,1]>self.img_H+pad).sum(axis=1)
                sum_ = x_0+x_1+y_0+y_1
                first_idx = -1
                last_idx = -1
                ori_length = length
                positive_indices = np.where(sum_ > 0)[0]
                if len(positive_indices)!=0:
                    first_idx = positive_indices[0]
                    last_idx = positive_indices[-1]
               
                if (first_idx<0 and last_idx<0) or first_idx>0 or length-last_idx-1>0 or a_i>10:
                    break
            if (first_idx<0 and last_idx<0) or first_idx>0 or length-last_idx-1>0:
                break

        if first_idx>=0 or last_idx>=0:
            if first_idx>=length-last_idx-1:
                slice_idx = first_idx
                length = slice_idx
                i_motion2d = i_motion2d[:length]
                ori_joints_pos_ = ori_joints_pos_[:length]
                joints_pos_ = joints_pos_[:length]
            elif first_idx<length-last_idx-1:
                slice_idx = length-last_idx-1
                length = slice_idx
                i_motion2d = i_motion2d[last_idx+1:]
                ori_joints_pos_ = ori_joints_pos_[last_idx+1:]
                joints_pos_ = joints_pos_[last_idx+1:]

      
        #global
        i_motion2d_global = i_motion2d.clone()
        i_motion2d_global[:,:,1] = i_motion2d_global[:,:,1]+(self.img_W-self.img_H)//2 
        K_global = K.clone()
        K_global[1,-1] = K_global[1,-1]+(self.img_W-self.img_H)//2 
        if self.is_pinhole:
            normed_motion2d_global = normalize_keypoints_to_patch(i_motion2d_global, crop_size=self.img_W)
        else:
            normed_motion2d_global = i_motion2d_global

        normed_motion2d,_,bbox =normalize_kp_2d_linear(i_motion2d) 
        K_adj = adjust_K(K[None].repeat(bbox.size()[0], 1, 1) , bbox)

        w_h = normalize_keypoints_to_patch(bbox[:,2] * 200, crop_size=self.img_W)[:,None,None]
        w_h = torch.cat([w_h,w_h],dim=-1)
        #cat w h
        normed_motion2d = torch.cat([normed_motion2d,w_h],dim=-2) 
        #root
        normed_motion2d[:,0,:] = normed_motion2d_global[:,0,:]

        
        J_2D = normed_motion2d.shape[1] 

        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            normed_motion2d = torch.cat([normed_motion2d, torch.zeros((pad_length, J_2D, 2))], dim=0)
            ori_joints_pos_ = torch.cat([ori_joints_pos_, torch.zeros((pad_length, J_ori, 3))], dim=0)
            K_adj = torch.cat([K_adj, torch.zeros((pad_length, 3, 3))], dim=0)
            
        RT_emb = torch.cat([eleva_angle,z_angle, y_t], dim=-1) 
        K_emb = torch.zeros((4), dtype=torch.float32) 
        K_emb[0] = K_global[0,0]
        K_emb[1] = K_global[1,1]
        K_emb[2] = K_global[0,2]
        K_emb[3] = K_global[1,2]
        K_emb = K_emb/3500
        pred_masks = torch.ones((max_motion_len, J_2D), dtype=torch.float32)
        # Return
        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  
            "gt_motion": ori_joints_pos_.float(),  
            # "T_w2c": T_w2c.float(),  # (4, 4)
            "is_pinhole": self.is_pinhole,  
            "RT_emb": RT_emb.float(),  
            "K_emb": K_emb.float(),  
            "pointmap": pointmap_refi.float(),  
            "mask": pred_masks.float(), 
            "text": caption,
            "task": "2D",
        }
        return return_data
   
    
    def _process_motion(self, joints_pos):
        # if self.train_fps != 20:
        #     joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        
        return joints_pos, ori_joints_pos
