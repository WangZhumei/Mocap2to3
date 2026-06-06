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
from hmr4d.utils.smplx_utils import make_smplx
import random
from scipy.spatial.transform import Rotation as R
from hmr4d.dataset.rich.rich_utils import (
    get_cam2params,
    remove_extra_rules,
    squared_crop_and_resize,
    sample_idx2meta,
    remove_bbx_invisible_frame,
    get_w2az_sahmr,
    parse_seqname_info,
    get_seqnames_of_split,
    get_seq_cam_fn,
    get_img_fn,
    get_img_key,
    get_augmented_square_bbx,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay,compute_T_eay2ayfz,compute_T_rotY,compute_T_move,compute_T_move2
from hmr4d.dataset.HumanML3D.utils import load_motion_files, swap_left_right, upsample_motion
from hmr4d.dataset.motionx.utils import generate_camera_intrinsics, normalize_keypoints_to_patch
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface,get_camera_mat_zface_wzm, get_gt2genCamera4_90_mat_wzm,get_gt2genCamera4_Y_mat_wzm,cartesian_to_spherical
from hmr4d.network.evaluator.word_vectorizer import WordVectorizer
from hmr4d.utils.o3d_utils import o3d_skeleton_animation
from hmr4d.utils.plt_utils import plt_skeleton_animation
from hmr4d.utils.hml3d.utils import standardize_motion
from hmr4d.dataset.motionx.utils import normalize_kp_2d,normalize_kp_2d_linear,adjust_K
from hmr4d.utils.net_utils import trusted_torch_load
import math

# For exporting joints3d.pth
class BaseDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/amass/smplhg_raw",
        split="train",  # No use here
        index_path="./inputs/hml3d/index.csv",
        humanact_path="./inputs/humanact12",
        target_fps=20,
        limit_size=None,
        export_data=False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.index_path = index_path
        self.humanact_path = humanact_path
        Log.info(f"Loading HML3D {split}...")

        self._load_dataset()
        # Options
        # HumanML3D uses 20 fps
        self.target_fps = target_fps
        self.limit_size = limit_size  # common usage: making validation faster

        if export_data:
            self._build_body_models()  # NOTE: this is only necessary for building joints3d.pth

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.motion_files))
        # Original and mirrored
        return len(self.motion_files) * 2

    def _build_body_models(self):
        body_models = {
            "male": make_smplx("rich-smplh", gender="male"),
            "female": make_smplx("rich-smplh", gender="female"),
        }
        self.smpl = body_models

    def _load_dataset(self):
        self.motion_files, self.new_names, self.start_frames, self.end_frames = load_motion_files(
            self.root, self.humanact_path, self.index_path
        )

    def _load_data(self, idx):
        motion_files = self.motion_files[idx // 2]
        data = np.load(motion_files)
        return data

    def _process_data(self, data, idx):
        
        return {}

    
    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


# for training
class Dataset(BaseDataset):
    def __init__(
        self,
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
        **kwargs,
    ):
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_root_next = is_root_next
        self.is_pinhole = is_pinhole
        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text

        self.is_notext = is_notext
        self.eleva_angle = eleva_angle
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size

        super().__init__(**kwargs)
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

    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = trusted_torch_load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
        # Dict of {"joints3d": tensor(F, J, C), "name": str}
        self.idx2meta = self.prepare_meta(self.split)
        Log.info(f"Loaded {len(self.idx2meta)} data from {self.root}")

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start_frame, end_frame, text_list = meta
        seq_name_ = seq_name + ".npy"
        joints3d = self.motion_files[seq_name_]["joints3d"]
        joints3d = joints3d[start_frame:end_frame]
        return_data = {"joints3d": joints3d, "text_list": text_list}
        return return_data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        J = joints_pos.shape[1]
        J_ori = ori_joints_pos.shape[1]
        length = joints_pos.shape[0]

        distance = torch.ones((1,)) * self.distance
        angle = torch.rand((1,)) * 2 * torch.pi
        # [-30, 30] eleva
        if self.eleva_angle is not None:
            eleva_angle = torch.ones((1,)) * self.eleva_angle / 180.0 * torch.pi
        else:
            eleva_angle = (torch.rand((1,)) * 2 - 1) * 30.0 / 180.0 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle, eleva_angle)  
        T_w2c = torch.inverse(cam_mat)[0]  # 4, 4
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # F, J, 3
        i_motion2d = project_p2d(c_motion, self.K[None], is_pinhole=self.is_pinhole)  # (F, J, 2)
        c_motion2d = cvt_p2d_from_i_to_c(i_motion2d, self.K[None])  # (F, J, 2)
        if self.is_pinhole:
            normed_motion2d = normalize_keypoints_to_patch(i_motion2d, crop_size=self.patch_size)
        else:
            normed_motion2d = i_motion2d

       
        # remove root as it is always at (0, 0)
        normed_motion2d = normed_motion2d[:, 1:]
        J_2D = normed_motion2d.shape[1] 

        max_motion_len = self.max_motion_time * self.train_fps
        if length < max_motion_len:
            # pad
            pad_length = max_motion_len - length
            normed_motion2d = torch.cat([normed_motion2d, torch.zeros((pad_length, J_2D, 2))], dim=0)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        
        # Return
        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  # (F, 22 - 1, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            # "T_w2c": T_w2c.float(),  # (4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            # "word_embeddings": word_embeddings,
            # "pos_one_hots": pos_one_hots,
            "text": caption,
            # "sent_len": sent_len,
            # "tokens": "_".join(tokens),
            "task": "2D",
        }
        return return_data

    def _process_motion(self, joints_pos):
        if self.train_fps != 20:
            joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        return joints_pos, ori_joints_pos

    def prepare_meta(self, split):
        meta = []
        split_file = f"./inputs/hml3d/{split}.txt"
        txt_path = f"./inputs/hml3d/texts"

        # Filter data by required_text
        def is_contain_required_text(text, required_text, anti_text):
            flag = False
            if required_text is None:
                flag = True
            else:
                for r_t in required_text:
                    if r_t in text:
                        flag = True
            if anti_text is not None:
                for a_t in anti_text:
                    if a_t in text:
                        flag = False
            return flag

        max_motion_len = self.max_motion_time * 20
        min_motion_len = self.min_motion_time * 20

        # https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/data/dataset.py#L225
        # Original hml vec has F - 1 frames, so slightly different number of data.
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                seq_name = line.strip()
                if seq_name + ".npy" in self.motion_files.keys():
                    motion = self.motion_files[seq_name + ".npy"]["joints3d"]
                    motion_len = motion.shape[0]
                    # Follow MDM, only uses [2s ~ 10s]
                    if motion_len < min_motion_len or motion_len > max_motion_len:
                        continue
                    text_data = []
                    flag = False
                    with cs.open(os.path.join(txt_path, seq_name + ".txt")) as text_f:
                        for text_line in text_f.readlines():
                            text_dict = {}
                            line_split = text_line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    flag = True
                                    text_data.append(text_dict)
                            else:
                                start_frame = int(f_tag * 20)
                                end_frame = int(to_tag * 20)
                                n_motion = motion[start_frame:end_frame]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) > max_motion_len):
                                    continue
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    meta.append([seq_name, start_frame, end_frame, [text_dict]])
                    if flag:
                        meta.append([seq_name, 0, motion_len, text_data])
        return meta
    
class PointMapTrainDataset(BaseDataset):
    def __init__(
        self,
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
        pm_W = 224,
        **kwargs,
    ):
        self.min_motion_time = min_motion_time
        self.max_motion_time = max_motion_time
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_root_next = is_root_next
        self.is_pinhole = is_pinhole
        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text
        self.img_H = img_H
        self.img_W = img_W
        self.pm_W = pm_W
        self.is_notext = is_notext
        self.eleva_angle = eleva_angle
        self.train_fps = train_fps
        self.distance = distance
        self.patch_size = patch_size
        self.rich_cam = trusted_torch_load("hmr4d/dataset/rich/resource/rich_cam.pth")
        #self.rich_cam = torch.load("hmr4d/dataset/rich/resource/rich_cam_sort.pth")
        self.pointmaps = self.get_pointmaps()

        super().__init__(**kwargs)
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
        
       
    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = trusted_torch_load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
        # Dict of {"joints3d": tensor(F, J, C), "name": str}
        self.idx2meta = self.prepare_meta(self.split)
        Log.info(f"Loaded {len(self.idx2meta)} data from {self.root}")

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        seq_name, start_frame, end_frame, text_list = meta
        seq_name_ = seq_name + ".npy"
        joints3d = self.motion_files[seq_name_]["joints3d"]
        joints3d = joints3d[start_frame:end_frame]
        return_data = {"joints3d": joints3d, "text_list": text_list}
        return return_data

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)
    #xywh
    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        ground = torch.min(joints_pos[:,:,1])
        thick = 0.05
        if ground>0:
            joints_pos[:,:,1] = joints_pos[:,:,1]-ground+thick
        else:
            joints_pos[:,:,1] = joints_pos[:,:,1]+abs(ground)+thick
           
        text_list = data["text_list"]
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption
        d_s = 0.5
        
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
                
                #random rot and move 3dpose 
                if random.random()<0.9:
                    T_rotY = compute_T_rotY(inverse=True)[0] 
                    joints_pos_ = apply_T_on_points(joints_pos, T_rotY) 
                    T_moveXZ = compute_T_move2(distance_,d_s,Y_r,inverse=True)[0] 
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
                T_w2c = torch.inverse(cam_mat)[0]  # N, 4, 4
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
        w_h = torch.cat([w_h,w_h],dim=-1)#s 1 2
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
        # Return
        pred_masks = torch.ones((max_motion_len, J_2D), dtype=torch.float32)
        return_data = {
            "length": length,  # Value = F
            "gt_motion2d": normed_motion2d.float(),  
            "gt_motion": ori_joints_pos_.float(),  
            # "T_w2c": T_w2c.float(), 
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
        if self.train_fps != 20:
            joints_pos = upsample_motion(joints_pos, 20, self.train_fps)
        ori_joints_pos = joints_pos.clone()
        
        return joints_pos, ori_joints_pos

    def get_pointmaps(self):
        pointmaps = {}
        for seq_name in self.rich_cam.keys():
            seq_cam = trusted_torch_load('inputs/RICH/WW_pointmap'+str(self.pm_W)+'/'+str(seq_name)+'.pth')
            pointmaps[seq_name] =seq_cam
        return pointmaps      
    def prepare_meta(self, split):
        meta = []
        split_file = f"./inputs/hml3d/{split}.txt"
        txt_path = f"./inputs/hml3d/texts"

        # Filter data by required_text
        def is_contain_required_text(text, required_text, anti_text):
            flag = False
            if required_text is None:
                flag = True
            else:
                for r_t in required_text:
                    if r_t in text:
                        flag = True
            if anti_text is not None:
                for a_t in anti_text:
                    if a_t in text:
                        flag = False
            return flag

        max_motion_len = self.max_motion_time * 20
        min_motion_len = self.min_motion_time * 20

        # https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/data/dataset.py#L225
        # Original hml vec has F - 1 frames, so slightly different number of data.
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                seq_name = line.strip()
                if seq_name + ".npy" in self.motion_files.keys():
                    motion = self.motion_files[seq_name + ".npy"]["joints3d"]
                    motion_len = motion.shape[0]
                    # Follow MDM, only uses [2s ~ 10s]
                    if motion_len < min_motion_len or motion_len > max_motion_len:
                        continue
                    text_data = []
                    flag = False
                    with cs.open(os.path.join(txt_path, seq_name + ".txt")) as text_f:
                        for text_line in text_f.readlines():
                            text_dict = {}
                            line_split = text_line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    flag = True
                                    text_data.append(text_dict)
                            else:
                                start_frame = int(f_tag * 20)
                                end_frame = int(to_tag * 20)
                                n_motion = motion[start_frame:end_frame]
                                if (len(n_motion)) < min_motion_len or (len(n_motion) > max_motion_len):
                                    continue
                                if is_contain_required_text(caption, self.required_text, self.anti_text):
                                    meta.append([seq_name, start_frame, end_frame, [text_dict]])
                    if flag:
                        meta.append([seq_name, 0, motion_len, text_data])
        return meta
    

# Multi-view dataset for MAS inference
class MVDataset(Dataset):
    def __init__(
        self,
        is_uniform_views=False,
        N_views=5,
        is_mm=False,
        **kwargs,
    ):
        self.is_uniform_views = is_uniform_views
        self.N_views = N_views
        self.w_vectorizer = WordVectorizer("./inputs/checkpoints/glove", "our_vab")
        if is_mm:
            self.mm_idx = np.random.permutation(len(self.idx2meta))
        super().__init__(**kwargs)

    def _process_data(self, data, idx):
        
        return {}


# Multi-view dataset for training multi-view generation
class MVCamDataset(MVDataset):
    def __init__(
        self,
        is_cam_rel2human=False,
        **kwargs,
    ):
        self.is_cam_rel2human = is_cam_rel2human

        super().__init__(**kwargs)

    def _process_data(self, data, idx):
        
        return {}

# Multi-view dataset for training multi-view generation
class MVRichPointMapDataset(MVCamDataset):
    def __init__(
        self,
        is_cam_rel2human=False,
        img_H=224,
        img_W=224,
        pm_W=224,
        **kwargs,
    ):
        self.is_cam_rel2human = is_cam_rel2human    
        self.img_H = img_H
        self.img_W = img_W
        self.pm_W = pm_W
        self.rich_cam = trusted_torch_load("hmr4d/dataset/rich/resource/rich_cam.pth")
        #self.rich_cam = torch.load("hmr4d/dataset/rich/resource/rich_cam_sort.pth")

        self.pointmaps = self.get_pointmaps()

        super().__init__(**kwargs)
    #speed up 2
    def _process_data(self, data, idx):
        joints_pos = data["joints3d"]#.copy()
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)

        ground = torch.min(joints_pos[:,:,1])
        thick = 0.05
        if ground>0:
            joints_pos[:,:,1] = joints_pos[:,:,1]-ground+thick
        else:
            joints_pos[:,:,1] = joints_pos[:,:,1]+abs(ground)+thick


        text_list = data["text_list"]
        # random.seed(idx)
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        caption = "" if self.is_notext else caption

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
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
            #distance.append(dist)
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
    
