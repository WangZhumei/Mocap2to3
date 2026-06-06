import os
from pathlib import Path
from hmr4d.utils.pylogger import Log
from tqdm import tqdm
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

from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from hmr4d.dataset.motionx.utils import load_motion_files, normalize_kp_2d
from hmr4d.utils.geo_transform import (
    apply_T_on_points,
    compute_T_ayf2az,
    project_p2d,
    cvt_to_bi01_p2d,
    cvt_p2d_from_i_to_c,
    cvt_from_bi01_p2d,
)
import hmr4d.utils.matrix as matrix
from hmr4d.utils.camera_utils import get_camera_mat_zface, cartesian_to_spherical
from hmr4d.network.evaluator.word_vectorizer import WordVectorizer
from hmr4d.utils.video_io_utils import read_video_np
from hmr4d.utils.vis.renderer import Renderer
import imageio
from hmr4d.utils.o3d_utils import o3d_skeleton_animation, vis_smpl_forward_animation
from hmr4d.utils.plt_utils import plt_skeleton_animation
from hmr4d.utils.hml3d.utils import standardize_motion


# For exporting joints3d.pth
class BaseDataset(data.Dataset):
    def __init__(
        self,
        root="inputs/motionx/idea400_smplx_HPE/idea400_smplx_HPE",
        split="train",  # No use here
        target_fps=30,
        limit_size=None,
        export_data=False,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.split = split
        Log.info(f"Loading MotionX-IDEA400 {split}...")

        self._load_dataset()
        # Options
        self.target_fps = target_fps
        self.limit_size = limit_size  # common usage: making validation faster

        if export_data:
            self._build_body_models()  # NOTE: this is only necessary for building joints3d.pth
            self.max_error = -1  # check cam_R error with identity
            self._notfound_video = []

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.motion_files))
        # Original and mirrored
        return len(self.motion_files)

    def _build_body_models(self):
        # TODO:
        body_models = {
            "male": make_smplx("rich-smplx", gender="male"),
            "female": make_smplx("rich-smplx", gender="female"),
            "neutral": make_smplx("rich-smplx", gender="neutral"),
        }
        self.smpl = body_models

    def _load_dataset(self):
        self.motion_files = load_motion_files(self.root)

    def _load_data(self, idx):
        motion_files = self.motion_files[idx]
        data = np.load(motion_files, allow_pickle=True)
        return data

    def _process_data(self, data, idx):
        name = self.motion_files[idx]
        video_path = name.replace("idea400_smplx_HPE", "idea400_video")[:-4] + ".mp4"
        if not os.path.exists(video_path):
            print("Video not found: ", video_path)
            self._notfound_video.append(video_path)

        transl = torch.tensor(data["trans"][0], dtype=torch.float32)  # F, 3
        global_orient = torch.tensor(data["root_orient"][0], dtype=torch.float32)  # F, 3
        body_pose = torch.tensor(data["pose_body"][0], dtype=torch.float32).flatten(1)  # (F, 63)
        F = global_orient.shape[0]
        betas = torch.tensor(data["betas"], dtype=torch.float32).expand(F, -1)

        # almost identity
        cam_R = torch.tensor(data["cam_R"][0], dtype=torch.float32)  # F, 3, 3

        # check identity
        error_identity = (cam_R - torch.eye(3)[None]).abs().max()
        print(f"{error_identity} / {self.max_error}")
        self.max_error = max(error_identity.item(), self.max_error)

        # has values
        cam_t = torch.tensor(data["cam_t"][0], dtype=torch.float32)  # F, 3

        cam_K = torch.tensor(data["intrins"], dtype=torch.float32)  # 4

        # this is real transl
        transl_c = transl + cam_t

        scale = data["world_scale"][0, 0]  # value

        # seems no gender, uses neutral
        smplx = self.smpl["neutral"]
        smpl_params_c = {
            "body_pose": body_pose,
            "betas": betas,
            "transl": transl_c,
            "global_orient": global_orient,
        }

        smpl_out = smplx(**smpl_params_c)
        joints = smpl_out.joints[:, :22] * scale  # (F, J, 3)

        # vis_smpl_forward_animation(transl_c, torch.cat([global_orient, body_pose], dim=-1))

        if False:  # cam-overlay
            # ----- Overlay ----- #
            # load video
            images = read_video_np(video_path)

            fps = 30
            K = torch.zeros((1, 3, 3), dtype=torch.float32)
            K[0, 0, 0] = cam_K[0]
            K[0, 1, 1] = cam_K[1]
            K[0, 0, 2] = cam_K[2]
            K[0, 1, 2] = cam_K[3]
            height, width = images.shape[1:3]
            faces = smplx.bm.faces
            renderer = Renderer(width, height, focal_length=None, device="cuda", faces=faces, K=K)
            writer = imageio.get_writer("tmp_debug.mp4", fps=fps, mode="I", format="FFMPEG", macro_block_size=1)

            kp_2d = project_p2d(joints, K.expand(F, -1, -1), is_pinhole=True)  # (F, J, 2)
            normed_kp2d, _, _ = normalize_kp_2d(kp_2d)
            normed_kp2d = normed_kp2d[:, :44].reshape(F, -1, 2)

            for i in tqdm(range(F)):
                img = images[i]
                img = renderer.render_mesh(scale * smpl_out.vertices[i].cuda(), img)
                writer.append_data(img)
            writer.close()

            plt_skeleton_animation(normed_kp2d, text=video_path, skeleton_type="smpl")

        # Return
        return_data = {
            "length": F,  # Value = F
            "incam_joints3d": joints.float(),  # (F, 22, 3)
            "name": name,  # str
            "video_path": video_path,  # str
            "cam_K": cam_K,  # (4)
        }

        return return_data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data


# for training
class Dataset(BaseDataset):
    def __init__(
        self,
        min_motion_len=40,
        max_motion_len=200,
        max_text_len=20,
        is_ignore_transl=True,
        unit_length=4,
        is_root_next=False,
        required_text=None,  # filter data by required_text
        anti_text=None,  # filter out data by anti_text
        **kwargs,
    ):
        self.min_motion_len = min_motion_len
        self.max_motion_len = max_motion_len
        self.max_text_len = max_text_len
        self.is_ignore_transl = is_ignore_transl
        self.unit_length = unit_length
        self.is_root_next = is_root_next
        # filter data by required_text
        self.required_text = required_text
        # filter data by anti_text
        self.anti_text = anti_text

        super().__init__(**kwargs)
        Log.info(f"Required text: {self.required_text}; Anti text: {self.anti_text}")

    def _load_dataset(self):
        if ".pth" in self.root:
            self.motion_files = torch.load(self.root)
        else:
            self.motion_files = np.load(self.root, allow_pickle=True)
        # Dict of {"incam_joints3d": tensor(F, J, C), "video_path": str, "cam_K": (4)}
        self.idx2meta = self.prepare_meta(self.split)

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

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        J = joints_pos.shape[1]
        J_ori = ori_joints_pos.shape[1]
        length = joints_pos.shape[0]

        distance = torch.ones((1,))
        angle = torch.rand((1,)) * 2 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle)  # 1, 4, 4
        T_w2c = torch.inverse(cam_mat)[0]  # 4, 4
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # F, J, 3
        i_motion2d = project_p2d(c_motion, self.K[None], is_pinhole=self.is_pinhole)  # (F, J, 2)
        bbx_lurb = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
        bi01_motion2d = cvt_to_bi01_p2d(i_motion2d, bbx_lurb[None])  # (F, J, 2)
        # plt_skeleton_animation(bi01_motion2d * 300, "smpl")

        if length < self.max_motion_len:
            # pad
            pad_length = self.max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            bi01_motion2d = torch.cat([bi01_motion2d, torch.zeros((pad_length, J, 2))], dim=0)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        # DEBUG
        # i_p2d = cvt_from_bi01_p2d(bi01_motion2d, bbx_lurb[None])  # (F, J, 2)
        # c_p2d = cvt_p2d_from_i_to_c(i_p2d[None], self.K[None])  # (1, F, J, 2)
        # o3d_skeleton_animation(joints_pos[None], c_p2d.reshape(1, 1, -1, 2), T_w2c[None], self.is_pinhole, caption)

        # Return
        return_data = {
            "length": length,  # Value = F
            "bbx_lurb": bbx_lurb.float(),  # (4)
            "gt_motion2d": bi01_motion2d.float(),  # (F, 22, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "T_w2c": T_w2c.float(),  # (4, 4)
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
        ori_joints_pos = joints_pos.clone()
        if self.is_root_next:
            # Add root velocity
            root_next = joints_pos[1:, :1]
            root_next = torch.cat([root_next, root_next[-1:]], dim=0)  # F, 1, C
            joints_pos = torch.cat([joints_pos, root_next], dim=1)  # F, J + 1, C

        # MDM does not move the character to the first frame when sample with another start_frame
        if self.is_ignore_transl:
            # move the every frame pelvis to origin
            joints_pos = joints_pos - joints_pos[:, :1, :]  # F, J, C
        else:
            # move the first frame pelvis to origin
            joints_pos = joints_pos - joints_pos[0, :1, :]  # F, J, C

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        m_length = joints_pos.shape[0]
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(joints_pos) - m_length)
        joints_pos = joints_pos[idx : idx + m_length]
        ori_joints_pos = ori_joints_pos[idx : idx + m_length]
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

        # https://github.com/GuyTevet/motion-diffusion-model/blob/main/data_loaders/humanml/data/dataset.py#L225
        # Original hml vec has F - 1 frames, so slightly different number of data.
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                seq_name = line.strip()
                if seq_name + ".npy" in self.motion_files.keys():
                    motion = self.motion_files[seq_name + ".npy"]["joints3d"]
                    motion_len = motion.shape[0]
                    # Follow MDM, only uses [2s ~ 10s]
                    if motion_len < self.min_motion_len or motion_len > self.max_motion_len:
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
                                if (len(n_motion)) < self.min_motion_len or (len(n_motion) > self.max_motion_len):
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
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        # random.seed(idx)
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

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

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        length = joints_pos.shape[0]

        F, J, _ = joints_pos.shape
        _, J_ori, _ = ori_joints_pos.shape

        N_views = self.N_views

        distance = torch.ones((N_views,))
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * torch.pi / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4
        joints_pos = joints_pos.reshape(1, F * J, 3)  # 1, F*J, 3
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # N, FJ, 3

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        i_motion2d = project_p2d(c_motion, Ks, is_pinhole=self.is_pinhole)  # (N, F*J, 2)
        bbx_lurb = torch.tensor([0, 0, 1, 1], dtype=torch.float32)  # 4
        bbx_lurb = bbx_lurb.reshape(1, 4).repeat(N_views, 1)  # N, 4
        bi01_motion2d = cvt_to_bi01_p2d(i_motion2d, bbx_lurb)  # (N, F*J, 2)
        bi01_motion2d = bi01_motion2d.reshape(N_views, F, J, 2)  # N, F, J, 2
        joints_pos = joints_pos.reshape(F, J, 3)  # F, J, 3

        if length < self.max_motion_len:
            # pad
            pad_length = self.max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            bi01_motion2d = torch.cat([bi01_motion2d, torch.zeros((N_views, pad_length, J, 2))], dim=1)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        # DEBUG
        # i_p2d = cvt_from_bi01_p2d(bi01_motion2d, bbx_lurb[None])  # (F, J, 2)
        # c_p2d = cvt_p2d_from_i_to_c(i_p2d[None], self.K[None])  # (1, F, J, 2)
        # o3d_skeleton_animation(joints_pos[None], c_p2d.reshape(1, 1, -1, 2), T_w2c[None], self.is_pinhole, caption)

        # Return
        return_data = {
            "length": length,  # Value = F
            "bbx_lurb": bbx_lurb.float(),  # (N, 4)
            "gt_motion2d": bi01_motion2d.float(),  # (N, F, 22, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "Ks": Ks.float(),  # N, 3, 3
            "word_embs": word_embeddings.astype(np.float32),
            "pos_onehot": pos_one_hots.astype(np.float32),
            "text": caption,
            "text_len": sent_len,
            # "tokens": "_".join(tokens),
            "task": "3D",
        }
        return return_data


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
        joints_pos = data["joints3d"]
        if isinstance(joints_pos, np.ndarray):
            joints_pos = torch.tensor(joints_pos, dtype=torch.float32)
        text_list = data["text_list"]
        # random.seed(idx)
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

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

        joints_pos, ori_joints_pos = self._process_motion(joints_pos)
        length = joints_pos.shape[0]

        F, J, _ = joints_pos.shape
        _, J_ori, _ = ori_joints_pos.shape

        N_views = self.N_views

        distance = torch.ones((N_views,))
        if self.is_uniform_views:
            start = torch.rand((1,)) * 2 * torch.pi
            interval = 2 * torch.pi / N_views
            angle = [start + i * interval for i in range(N_views)]
            angle = torch.cat((angle), dim=-1)
        else:
            angle = torch.rand((N_views,)) * 2 * torch.pi
        cam_mat = get_camera_mat_zface(matrix.identity_mat()[None], distance, angle)  # N, 4, 4
        T_w2c = torch.inverse(cam_mat)  # N, 4, 4
        joints_pos = joints_pos.reshape(1, F * J, 3)  # 1, F*J, 3
        c_motion = matrix.get_relative_position_to(joints_pos, cam_mat)  # N, FJ, 3

        Ks = self.K[None].repeat(N_views, 1, 1)  # N, 3, 3
        i_motion2d = project_p2d(c_motion, Ks, is_pinhole=self.is_pinhole)  # (N, F*J, 2)
        bbx_lurb = torch.tensor([0, 0, 1, 1], dtype=torch.float32)  # 4
        bbx_lurb = bbx_lurb.reshape(1, 4).repeat(N_views, 1)  # N, 4
        bi01_motion2d = cvt_to_bi01_p2d(i_motion2d, bbx_lurb)  # (N, F*J, 2)
        bi01_motion2d = bi01_motion2d.reshape(N_views, F, J, 2)  # N, F, J, 2
        joints_pos = joints_pos.reshape(F, J, 3)  # F, J, 3

        if length < self.max_motion_len:
            # pad
            pad_length = self.max_motion_len - length
            joints_pos = torch.cat([joints_pos, torch.zeros((pad_length, J, 3))], dim=0)
            bi01_motion2d = torch.cat([bi01_motion2d, torch.zeros((N_views, pad_length, J, 2))], dim=1)
            ori_joints_pos = torch.cat([ori_joints_pos, torch.zeros((pad_length, J_ori, 3))], dim=0)

        spherical_coord = cartesian_to_spherical(matrix.get_position(cam_mat))  # N, 3
        theta, azimuth, z = spherical_coord[..., :1], spherical_coord[..., 1:2], spherical_coord[..., 2:3]

        if self.is_cam_rel2human:
            d_T = torch.cat([theta, torch.sin(azimuth), torch.cos(azimuth), z], dim=-1)  # N, 4
        else:
            # delta = target - condition
            d_theta = theta - theta[:1]  # N, 1
            d_azimuth = (azimuth - azimuth[:1]) % 2 * torch.pi  # N, 1
            d_z = z - z[:1]  # N, 1
            d_T = torch.cat([d_theta, torch.sin(d_azimuth), torch.cos(d_azimuth), d_z], dim=-1)  # N, 4

        # DEBUG
        # i_p2d = cvt_from_bi01_p2d(bi01_motion2d, bbx_lurb[None])  # (F, J, 2)
        # c_p2d = cvt_p2d_from_i_to_c(i_p2d[None], self.K[None])  # (1, F, J, 2)
        # o3d_skeleton_animation(joints_pos[None], c_p2d.reshape(1, 1, -1, 2), T_w2c[None], self.is_pinhole, caption)

        # Return
        return_data = {
            "length": length,  # Value = F
            "bbx_lurb": bbx_lurb.float(),  # (N, 4)
            "gt_motion2d": bi01_motion2d.float(),  # (N, F, 22, 2)
            "gt_motion": ori_joints_pos.float(),  # (F, 22, 3)
            "T_w2c": T_w2c.float(),  # (N, 4, 4)
            "is_pinhole": self.is_pinhole,  # Value = False or True
            "Ks": Ks.float(),  # N, 3, 3
            "cam_emb": d_T.float(),  # N, 4
            "word_embs": word_embeddings.astype(np.float32),
            "pos_onehot": pos_one_hots.astype(np.float32),
            "text": caption,
            "text_len": sent_len,
            # "tokens": "_".join(tokens),
            "task": "3D",
        }
        return return_data
