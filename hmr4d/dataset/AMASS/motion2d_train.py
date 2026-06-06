import os
from pathlib import Path
from hmr4d.utils.pylogger import Log
from tqdm import trange
import numpy as np
import pickle
import torch
import cv2
from torch.utils import data
import codecs as cs
import json
import decord
from decord import cpu, gpu
from .utils import *
from .cam_traj_utils import CameraAugmentorV11
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv,get_tgtcoord_rootparam
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict, trusted_torch_load
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points, project_p2d, cvt_p2d_from_i_to_c,transform_mat,compute_T_move,compute_T_move2,compute_T_rotY,compute_T_move,compute_T_move2,compute_T_move0

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines, convert_motion_as_line_mesh
from hmr4d.utils.smplx_utils import make_smplx

from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch
#from hmr4d.utils.vis_open3d import Model,ModelPose
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    compute_T_ayfx,
    compute_T_ayfz,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
import random
import hmr4d.utils.matrix as matrix
from hmr4d.dataset.motionx.utils import normalize_kp_2d,normalize_kp_2d_linear,adjust_K
from hmr4d.dataset.HumanML3D.utils import upsample_motion
from scipy.spatial.transform import Rotation as R
import math
from hmr4d.network.evaluator.word_vectorizer import WordVectorizer


# For exporting joints3d.pth
class BaseDataset(data.Dataset):
    def __init__(self,root="inputs/amass/joints3d.pth", cam_augmentation="v11", limit_size=None,skip_moyo=True,export_data=False,random1024=False):
        super().__init__()
        self.cam_augmentation = cam_augmentation
        self.limit_size = limit_size
        self.skip_moyo = skip_moyo
        if export_data:
            self.smplx = make_smplx("supermotion")
            self.smplx_lite = make_smplx("supermotion_smpl24")
        self.root = root
        #self.root = Path("inputs/AMASS/hmr4d_support")
        Log.info(f"Loading AMASS trian...")
        self.dataset_name = "AMASS"
        self.random1024 = random1024

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        #NotImplementedError("_load_dataset is not implemented")
        filename = self.root / "smplxpose_v2.pth"
        Log.info(f"[{self.dataset_name}] Loading from {filename} ...")
        tic = Log.time()
        if self.random1024:  # Debug, faster loading
            try:
                Log.info(f"[{self.dataset_name}] Loading 1024 samples for debugging ...")
                self.motion_files = trusted_torch_load(self.root / "smplxpose_v2_random1024.pth")
            except:
                Log.info(f"[{self.dataset_name}] Not found! Saving 1024 samples for debugging ...")
                self.motion_files = trusted_torch_load(filename)
                keys = list(self.motion_files.keys())
                keys = np.random.choice(keys, 1024, replace=False)
                self.motion_files = {k: self.motion_files[k] for k in keys}
                torch.save(self.motion_files, self.root / "smplxpose_v2_random1024.pth")
        else:
            self.motion_files = trusted_torch_load(filename)
        self.seqs = list(self.motion_files.keys())
        Log.info(f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s")


    def _get_idx2meta(self):
        self.idx2meta = []
        i = 0
        for vid in self.motion_files:
            i+=1
            
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["pose"].shape[0]
            self.idx2meta.extend([(vid)])

    

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _load_data(self, idx):
        #NotImplementedError("_load_data is not implemented")
        start_id = 0
        mid= self.idx2meta[idx]
        raw_data = self.motion_files[mid]
        raw_len = raw_data["pose"].shape[0] - start_id
        data = {
            "body_pose": raw_data["pose"].clone()[start_id:, :],  # (F, 63)
            "betas": raw_data["beta"].clone().repeat(raw_len,1),  # (10)
            "global_orient": raw_data["pose"].clone()[start_id:, :3],  # (F, 3)
            "transl": raw_data["trans"].clone()[start_id:],  # (F, 3)
            "vid": mid,  # (F, 3)
        }
        
        return data

    def _process_data(self, data, idx):
        return {}

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data

class Dataset(BaseDataset):
    def __init__(self,root="inputs/amass/joints3d.pth", limit_size=None):
        self.root = root
        self.limit_size = limit_size
        super().__init__(root=root,limit_size=limit_size)
    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = trusted_torch_load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        mid, start_id,start,end = meta
        joints3d = self.motion_files[mid]["joints3d"]
        joints3d = joints3d[start:end]
        return_data = {"joints3d": joints3d, "vid": mid}
        return return_data
    def _get_idx2meta(self):
        self.idx2meta = []
        #max_motion_len = (self.max_motion_time - 4) * self.train_fps
        max_motion_len = self.max_motion_time * self.train_fps
        min_motion_len = self.min_motion_time * self.train_fps
        # Skip too-long idle-prefix
        #motion_start_id = {}
        for vid in self.motion_files:
            if self.skip_moyo and "moyo_smplxn" in vid:
                continue
            seq_length = self.motion_files[vid]["joints3d"].shape[0]
            if seq_length < min_motion_len:  # Skip clips that are too short
                continue
            s_motion = 0
            slice_ = seq_length//max_motion_len
            for i in range(slice_):
                start_ = s_motion+i*max_motion_len
                end_ = start_+max_motion_len
                self.idx2meta.extend([(vid, s_motion,start_,end_)])
            if slice_==0:
                end_ = 0
            if seq_length-1-end_>=min_motion_len:
                self.idx2meta.extend([(vid, s_motion,end_,seq_length-1)])
    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)
    
    def _process_data(self, data, idx):
        NotImplementedError("_process_data is not implemented")

class MVRichPointMapDataset(Dataset):
    def __init__(
        self,
        is_cam_rel2human=False,
        img_H=224,
        img_W=224,
        is_notext=False,
        is_uniform_views=False,
        N_views=5,
        train_fps=20,
        distance=1.0,
        min_motion_time=2,
        max_motion_time=10,
        max_text_len=20,
        is_ignore_transl=True,
        unit_length=4,
        is_root_next=False,
        is_pinhole=False,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        eleva_angle=None,
        patch_size=224,
        l_factor=1.5,  # speed augmentation
        pm_W=224,
        skip_moyo=True,  # not contained in the ICCV19 released version
        cam_augmentation="v11",
        random1024=False,  # DEBUG
        limit_size=None,
        **kwargs,
    ):
        self.is_cam_rel2human = is_cam_rel2human    
        self.img_H = img_H
        self.img_W = img_W
        self.is_uniform_views = is_uniform_views
        self.N_views = N_views
        self.is_notext = is_notext
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_root_next = is_root_next
        self.is_pinhole = is_pinhole
        self.required_text = required_text
        self.anti_text = anti_text
        self.img_H = img_H
        self.img_W = img_W
        self.eleva_angle = eleva_angle
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size
        self.pm_W = pm_W

        self.rich_cam = trusted_torch_load("hmr4d/dataset/rich/resource/rich_cam.pth")
        #self.rich_cam = torch.load("hmr4d/dataset/rich/resource/rich_cam_sort.pth")
        self.pointmaps = self.get_pointmaps()

        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        self.l_factor = l_factor
        self.random1024 = random1024
        self.skip_moyo = skip_moyo
        self.dataset_name = "AMASS"
        super().__init__(limit_size=limit_size)

        if self.is_notext:
            Log.info(f"Do not use text input!")
        else:
            Log.info(f"Required text: {self.required_text}; Anti text: {self.anti_text}")
        if self.is_pinhole:
            self.K = generate_camera_intrinsics(self.patch_size, self.patch_size)
            Log.info(f"Use K with patch_size={self.patch_size} and distance={self.distance}")
        else:
            self.K = torch.tensor(
                [
                    [1.0000e00, 0.0000e00, 0.0000e00],
                    [0.0000e00, 1.0000e00, 0.0000e00],
                    [0.0000e00, 0.0000e00, 1.0000e00],
                ]
            )
            Log.info(f"Use identity K")
        
    
    #speed up
    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        T_0 = compute_T_move0(joints_pos[0][None], inverse=True)  
        joints_pos = apply_T_on_points(joints_pos, T_0)  #
        
        caption = "" #if self.is_notext else caption
        # pad with "unk"
        tokens = ["sos/OTHER"] + ["eos/OTHER"]
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

        d_s = 0.5
        N_views = self.N_views
        while True:
            ran_seq_idx = random.randint(0,len(self.rich_cam)-1)
            seq_name = list(self.rich_cam.keys())[ran_seq_idx]
            cam_keys = self.rich_cam[seq_name]
            pad = 100
            start = 1000
            end = -1
            Ts_w2c, Ks ,Ks_global = [], [], []
            normed_motion2ds = []
            normed_motion2ds_global = []
            temp_motion2ds = []
            orig_K = []
            angle,eleva_angle,z_angle,distance,x_t,y_t,z_t = [], [], [], [], [], [], []
            cam_ = []
            ran_cam_idx = random.randint(0,len(cam_keys)-1)
            cam_key0 = list(cam_keys.keys())[(ran_cam_idx)]
            cam_Ekeys = cam_keys[cam_key0]['cam_Eay']
            T_ayfz2eay = cam_keys[cam_key0]['T_ayfz2eay']
            distance_ = cam_keys[cam_key0]['distance']
            Y_r = cam_keys[cam_key0]['Y_r']
            #random rot 3dpose
            if random.random()<0.9:
                T_rotY = compute_T_rotY(inverse=True)[0] 
                joints_pos_ = apply_T_on_points(joints_pos, T_rotY) 
                T_moveXZ = compute_T_move2(distance_,d_s,Y_r,inverse=True)[0] 
                joints_pos_ = apply_T_on_points(joints_pos_, T_moveXZ)  #
                joints_pos_ = apply_T_on_points(joints_pos_.clone(), T_ayfz2eay)  #
            else:
                joints_pos_ = apply_T_on_points(joints_pos.clone(), T_ayfz2eay)  #
            joints_pos_, ori_joints_pos_ = self._process_motion(joints_pos_)
            length = joints_pos_.shape[0]
            ori_length = length
            F, J, _ = joints_pos_.shape
            _, J_ori, _ = ori_joints_pos_.shape


            for cam_idx in range(len(cam_Ekeys)):
                cam_key = list(cam_Ekeys.keys())[cam_idx]
                T_w2c = cam_Ekeys[cam_key]['T_w2c']  
                K = cam_Ekeys[cam_key]['K']  
                gt_T_Eayfz2c = cam_Ekeys[cam_key]['gt_T_Eayfz2c'] 
                cam_.append(cam_key)
                
                c_motion = matrix.get_relative_position_to(joints_pos_, torch.inverse(gt_T_Eayfz2c)[None])  
                i_motion2d = project_p2d(c_motion, K[None], is_pinhole=self.is_pinhole)  
                
                x_0 = (i_motion2d[:,:,0]<0-pad).sum(axis=1)
                x_1 = (i_motion2d[:,:,0]>self.img_W+pad).sum(axis=1)
                y_0 = (i_motion2d[:,:,1]<0-pad).sum(axis=1)
                y_1 = (i_motion2d[:,:,1]>self.img_H+pad).sum(axis=1)
                sum_ = x_0+x_1+y_0+y_1
                first_idx = 1000
                last_idx = -1
                positive_indices = np.where(sum_ > 0)[0]
                if len(positive_indices)!=0:
                    first_idx = positive_indices[0]
                    last_idx = positive_indices[-1]
                start = min(start,first_idx)
                end = max(end,last_idx)
                temp_motion2ds.append(i_motion2d)

                Ts_w2c.append(gt_T_Eayfz2c)
                orig_K.append(K)
                
            if (start==1000 and end==-1) or start>0 or end<length-1:
                break
        
        pointmaps = []
        for i in range(len(temp_motion2ds)):
            pointmap_info = self.pointmaps[seq_name][cam_key0][cam_[i]]
            map_ = pointmap_info['map']
            map_idx = pointmap_info['idx']
            pointmap_refi = torch.zeros((self.pm_W*self.pm_W,3))
            pointmap_refi[map_idx,:] = map_
            pointmap_refi = pointmap_refi.reshape(self.pm_W,self.pm_W,3)
            pointmaps.append(pointmap_refi)
            cam_mat_ =torch.inverse(Ts_w2c[i]).numpy()#gt_T_ayfz2c.numpy()
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
    
            i_motion2d_global = temp_motion2ds[i].clone()
            i_motion2d_global[:,:,1] = i_motion2d_global[:,:,1]+(self.img_W-self.img_H)//2 #4112
            K_global = orig_K[i].clone()
            K_global[1,-1] = K_global[1,-1]+(self.img_W-self.img_H)//2 #4112
            if self.is_pinhole:
                normed_motion2d_global = normalize_keypoints_to_patch(i_motion2d_global, crop_size=self.img_W)#4112 -1 1
            else:
                normed_motion2d_global = i_motion2d_global
            normed_pred_motion2d,_,bbox =normalize_kp_2d_linear(temp_motion2ds[i]) #224 -1 1
            K_adj = adjust_K(K[None].repeat(bbox.size()[0], 1, 1) , bbox)
            w_h = normalize_keypoints_to_patch(bbox[:,2] * 200, crop_size=self.img_W)[:,None,None]#s 1 1
            w_h = torch.cat([w_h,w_h],dim=-1)#s 1 2
            normed_pred_motion2d = torch.cat([normed_pred_motion2d,w_h],dim=-2) #s j+1 2
            normed_motion2d_global = torch.cat([normed_motion2d_global,w_h],dim=-2)

            Ks.append(K_adj)
            Ks_global.append(K_global)
            normed_motion2ds.append(normed_pred_motion2d)
            normed_motion2ds_global.append(normed_motion2d_global)
        

        pointmaps = torch.stack(pointmaps)
        normed_motion2ds = torch.stack(normed_motion2ds)
        normed_motion2ds_global = torch.stack(normed_motion2ds_global)
        Ts_w2c = torch.stack(Ts_w2c)
        Ks = torch.stack(Ks)
        Ks_global = torch.stack(Ks_global)
        cam_mat = torch.inverse(Ts_w2c)
        angle = torch.tensor(angle)
        eleva_angle = torch.tensor(eleva_angle)
        z_angle = torch.tensor(z_angle)
        distance = torch.tensor(distance)
        x_t = torch.tensor(x_t)
        y_t = torch.tensor(y_t)
        z_t = torch.tensor(z_t)
        if start!=1000 and end!=-1:
            if start>=length-end-1:
                slice_idx = start
                length = slice_idx
                normed_motion2ds = normed_motion2ds[:,:length]
                normed_motion2ds_global = normed_motion2ds_global[:,:length]
                ori_joints_pos_ = ori_joints_pos_[:length]
                joints_pos_ = joints_pos_[:length]
            else:
                slice_idx = length-end-1
                length = slice_idx
                normed_motion2ds = normed_motion2ds[:,end+1:]
                normed_motion2ds_global = normed_motion2ds_global[:,end+1:]
                ori_joints_pos_ = ori_joints_pos_[end+1:]
                joints_pos_ = joints_pos_[end+1:]

        normed_motion2ds[:,:,0,:] = normed_motion2ds_global[:,:,0,:]
        J_2D = normed_motion2ds.shape[-2]

        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            joints_pos_ = torch.cat([joints_pos_, torch.zeros((pad_length, J, 3))], dim=0)
            normed_motion2ds = torch.cat([normed_motion2ds, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            normed_motion2ds_global = torch.cat([normed_motion2ds_global, torch.zeros((N_views, pad_length, J_2D, 2))], dim=1)
            ori_joints_pos_ = torch.cat([ori_joints_pos_, torch.zeros((pad_length, J_ori, 3))], dim=0)
            
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
            "gt_motion": ori_joints_pos_.float(),
            "T_w2c": Ts_w2c.float(),
            "is_pinhole": self.is_pinhole,  
            "Ks": Ks_global.float(),
            "patch_size": self.img_W,
            "RT_emb": RT_emb.float(),
            "K_emb": K_emb.float(),
            "pointmap": pointmaps.float(),  
            "word_embs": word_embeddings.astype(np.float32),
            "pos_onehot": pos_one_hots.astype(np.float32),
            "text": caption, 
            "text_len": sent_len,
            "task": "3D",
            "data_info": seq_name,
        }
        return return_data
    def get_pointmaps(self):
        pointmaps = {}
        for seq_name in self.rich_cam.keys():
            seq_cam = trusted_torch_load('inputs/RICH/WW_pointmap'+str(self.pm_W)+'/'+str(seq_name)+'.pth')
            pointmaps[seq_name] =seq_cam
        return pointmaps
    
    def _process_motion(self, joints_pos):
        # if self.train_fps != 20:
        #     joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        
        return joints_pos, ori_joints_pos
    
    
