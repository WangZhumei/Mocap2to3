import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines
from hmr4d.utils.hml3d.utils_reverse import convert_hmlvec263_to_motion
import hmr4d.utils.matrix as matrix
from hmr4d.dataset.supermotion.collate import pad_to_max_len
from einops import einsum, rearrange, repeat
import numpy as np

from hmr4d.utils.o3d_utils import o3d_skeleton_animation, pos_2dto3d, get_good_z_for_2dvis


class Motion2Dto3DVisualizer(pl.Callback):
    def __init__(self, skeleton_type="smpl"):
        """Visualizing final motion."""
        super().__init__()
        self.skeleton_type = skeleton_type
        self.cur_batch = 0

        self.on_test_batch_end = self.on_predict_batch_end

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)
        pred_motion_prog = outputs["pred_progress"]["pred_motion"]  # (B, progress, L, J, 3)
        pred_motion2d_prog = outputs["pred_progress"]["pred_motion2d"]  # (B, progress, V, L, J, 2)
        gt_motion = batch["gt_motion"]  # (B, L, J, 3)
        is_pinhole = batch["is_pinhole"][0]
        T_w2c = batch["T_w2c"]  # (B, V, 4, 4)
        #cam_RT = batch["T_w2c_gt"]  # (B, V, 4, 4)
        #T_w2c[:,0,:,:] = cam_RT#[:,0,:,:]
        J = pred_motion2d_prog.shape[-2]
        P = pred_motion2d_prog.shape[1]

        for b in range(B):
            l = length[b]
            gt_motion_ = gt_motion[b][:l]  # L, J, 3
            pred_m_prog = pred_motion_prog[b][:, :l]  # progress, L, J, 3
            pred_m2d_prog = pred_motion2d_prog[b][:, :, :l]  # progress, V, L, J, 2
            T_w2c_ = T_w2c[b]
            txt = text[b] if text is not None else ""

            # FIXME: hardcode joints number here
            if J == 23:
                # Assume last joints is next frame root
                accum_root = torch.cumsum(pred_m_prog[..., -1:, :], dim=-3)  # (progress, L, 1, 3)
                # Convert accumlated world pose to true world pose
                accum_pred_m_prog = pred_m_prog.clone()
                # Add accumulated root to each joints instead of virtual next frame root
                accum_pred_m_prog[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
                # Assign virtual next frame root
                accum_pred_m_prog[..., -1:, :] = accum_root

                cam_mat = torch.inverse(T_w2c_)  # V, 4, 4
                cam_mat = repeat(cam_mat, "v c d -> p l v c d", p=P, l=l)  # (progress, l, V, 4, 4)
                cam_pos = matrix.get_position(cam_mat)  # progress, l, V, 3
                accum_cam_pos = cam_pos + accum_root  # (progress, l, V, 3)
                accum_cam_mat = matrix.set_position(cam_mat, accum_cam_pos)  # (progress, l, V, 4, 4)
                T_w2c_ = torch.inverse(accum_cam_mat)  # (progress, l, V, 4, 4)

                o3d_skeleton_animation(
                    accum_pred_m_prog,
                    pos_2d=pred_m2d_prog,
                    w2c=T_w2c_,
                    gt_pos=gt_motion_,
                    is_pinhole=is_pinhole,
                    name="Global-" + txt,
                    skeleton_type=self.skeleton_type,
                )
            else:
                o3d_skeleton_animation(
                    pred_m_prog,
                    pos_2d=pred_m2d_prog,
                    w2c=T_w2c_,
                    gt_pos=gt_motion_,
                    is_pinhole=is_pinhole,
                    name="Local-" + txt,
                    skeleton_type=self.skeleton_type,
                )

            self.cur_batch += 1


class Wis3DMotion2Dto3DVisualizer(pl.Callback):
    def __init__(self, name, max_len=200, max_count=100):
        super().__init__()
        self.wis3d = make_wis3d(name=name)
        self.max_len = max_len
        self.max_count = max_count  # wis3d costs a lot cpu
        self.on_test_batch_end = self.on_predict_batch_end
        self.counter = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        text = batch.get("text", None)
        pred_motion_prog = outputs["pred_progress"]["pred_motion"]  # (B, progress, L, J, 3)
        pred_motion2d_prog = outputs["pred_progress"]["pred_motion2d"]  # (B, progress, V, L, J, 2)
        gt_motion = batch["gt_motion"]  # (B, L, J, 3)
        is_pinhole = batch["is_pinhole"][0]
        T_w2c = batch["T_w2c"]  # (B, V, 4, 4)
        J = pred_motion2d_prog.shape[-2]
        P = pred_motion2d_prog.shape[1]

        for b in range(B):
            if self.counter > self.max_count:
                continue

            l = length[b]
            gt_motion_ = gt_motion[b][:l]  # L, J, 3
            pred_m_prog = pred_motion_prog[b][:, :l]  # progress, L, J, 3
            pred_m2d_prog = pred_motion2d_prog[b][:, :, :l]  # progress, V, L, J, 2
            T_w2c_ = T_w2c[b]
            txt = text[b] if text is not None else ""

            # FIXME: hardcode joints number here
            if J == 23:
                # Assume last joints is next frame root
                accum_root = torch.cumsum(pred_m_prog[..., -1:, :], dim=-3)  # (progress, L, 1, 3)
                # Convert accumlated world pose to true world pose
                accum_pred_m_prog = pred_m_prog.clone()
                # Add accumulated root to each joints instead of virtual next frame root
                accum_pred_m_prog[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
                # Assign virtual next frame root
                accum_pred_m_prog[..., -1:, :] = accum_root

                cam_mat = torch.inverse(T_w2c_)  # V, 4, 4
                cam_mat = repeat(cam_mat, "v c d -> p l v c d", p=P, l=l)  # (progress, l, V, 4, 4)
                cam_pos = matrix.get_position(cam_mat)  # progress, l, V, 3
                accum_cam_pos = cam_pos + accum_root  # (progress, l, V, 3)
                accum_cam_mat = matrix.set_position(cam_mat, accum_cam_pos)  # (progress, l, V, 4, 4)
                T_w2c_ = torch.inverse(accum_cam_mat)  # (progress, l, V, 4, 4)

                # remove next root joint
                pred_m = accum_pred_m_prog[-1, :, :-1]  # (L, 22, 3)
                pred_m2d = pred_m2d_prog[-1, :, :, :-1]  # (V, L, 22, 2)
            else:
                pred_m = pred_m_prog[-1]  # (L, J, 3)
                pred_m2d = pred_m2d_prog[-1]  # V, L, J, 2

            T_w2c_last = rearrange(T_w2c_[-1], "l v c d -> v l c d")  # V, L, 4, 4
            root = pred_m[None, :, 0, :]  # 1, L, 3
            _, min_z = get_good_z_for_2dvis(T_w2c_last, root, is_pinhole=is_pinhole)
            pred_m2din3d = pos_2dto3d(pred_m2d, T_w2c_last, min_z, is_pinhole=is_pinhole)  # V, L, J, 3

            mid = txt
            pred = pad_to_max_len(pred_m, self.max_len)
            gt = pad_to_max_len(gt_motion_, self.max_len)

            add_motion_as_lines(gt, self.wis3d, name=mid + " ::: gt", skeleton_type="smpl22")
            add_motion_as_lines(pred, self.wis3d, name=mid + " ::: pred", skeleton_type="smpl22")
            for i in range(pred_m2din3d.shape[0]):
                pred_m2din3d_v = pad_to_max_len(pred_m2din3d[i], self.max_len)
                add_motion_as_lines(pred_m2din3d_v, self.wis3d, name=f"{mid} ::: view-{i}", skeleton_type="smpl22")

            self.counter += 1
