import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.o3d_utils import o3d_skeleton_animation


class MotionLogger(pl.Callback):
    def __init__(self, name, skeleton_type="smpl", scale=1.0, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.skeleton_type = skeleton_type
        self.scale = scale  # scale the results for visualization
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"]
        device = batch["length"].device
        is_pinhole = batch["is_pinhole"][0]
        Ts_w2c = batch["Ts_w2c"]
        text = batch.get("text", None)
        J = outputs["pred_motion_progress"].shape[3]
        P = outputs["pred_motion_progress"].shape[1]

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            l = length[b]
            w_p3d = outputs["pred_motion_progress"][b][:, :l]  # (progress, L, J, 3)
            w_p3d = w_p3d * self.scale
            c_p2d = outputs["pred_c_p2d_progress"][b][:, :, : l * J]  # (progress, V, L*J, 3)
            c_p2d = c_p2d * self.scale
            if "pred_3d_progress" in outputs.keys():
                p_w_p3d = outputs["pred_3d_progress"][b][:, :, : l * J]  # (progress, V, L*J , 3)
                p_w_p3d = p_w_p3d * self.scale
            else:
                p_w_p3d = None
            txt = text[b] if text is not None else ""
            T_w2c = Ts_w2c[b]
            o3d_skeleton_animation(
                w_p3d,
                c_p2d,
                T_w2c,
                p_w_p3d,
                is_pinhole,
                txt,
                skeleton_type=self.skeleton_type,
            )
            # FIXME: hardcode joints number here
            if J == 23:
                # Assume last joints is next frame root
                accum_root = torch.cumsum(w_p3d[..., -1:, :], dim=-3)  # (progress, L, 1, 3)
                # Convert accumlated world pose to true world pose
                accum_w_p3d = w_p3d.clone()
                # Add accumulated root to each joints instead of virtual next frame root
                accum_w_p3d[..., 1:, :-1, :] += accum_root[..., :-1, :, :]
                # Assign virtual next frame root
                accum_w_p3d[..., -1:, :] = accum_root
                o3d_skeleton_animation(
                    accum_w_p3d,
                    c_p2d,
                    T_w2c,
                    p_w_p3d,
                    is_pinhole,
                    "Global: " + txt,
                    skeleton_type=self.skeleton_type,
                )

            self.cur_batch += 1
