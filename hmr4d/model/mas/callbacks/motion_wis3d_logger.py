import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.wis3d_utils import make_wis3d, get_gradient_colors, get_const_colors, add_joints22_motion_as_lines


class MotionLogger(pl.Callback):
    def __init__(self, name, time_postfix=True, max_batches=1000):
        """Visualizing final motion."""
        super().__init__()
        self.wis3d = make_wis3d(name=name, time_postfix=time_postfix)
        self.max_batches = max_batches  # max number of batches to visualize, exceed will raise error
        self.cur_batch = 0

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        B = batch["length"].shape[0]
        length = batch["length"][0]
        device = batch["length"].device
        J = outputs["pred_motion"].shape[2]

        green_const = torch.tensor([0, 1.0, 0, 1.0])[None].to(device)
        red_gradients = get_gradient_colors("red", num_points=length, alpha=0.6).to(device)
        green_gradients = get_gradient_colors("green", num_points=length, alpha=0.6).to(device)

        from einops import rearrange
        from hmr4d.utils.geo_transform import apply_T_on_points

        Ts_c2w = torch.inverse(batch["Ts_w2c"][:, 0])  # (B, 4, 4)
        c_p2d_controled = outputs["pred_c_p2d_progress"][:, 0, 0]  # (B, (l j), c)
        c_p3d_controled = F.pad(c_p2d_controled, (0, 1), value=1)  # assume z=1
        w_p3d_controled = apply_T_on_points(c_p3d_controled, Ts_c2w)  # (B, (l j), c)
        w_p3d_controled = rearrange(w_p3d_controled, "b (l j) c -> b l j c", l=length, j=J, c=3)

        for b in range(B):
            if self.cur_batch > self.max_batches:
                raise ValueError("Exceed max_batches, stop visualization.")
            record_name = f"sample_{self.cur_batch:03d}"

            pred_motion = outputs["pred_motion"][b]  # (L, J, 3)
            add_joints22_motion_as_lines(pred_motion, self.wis3d, name=f"{record_name}")

            # Add a w_p2d
            control_view = w_p3d_controled[b]  # (L, J, 3)
            add_joints22_motion_as_lines(control_view, self.wis3d, name=f"{record_name}_controlview")

            self.cur_batch += 1
