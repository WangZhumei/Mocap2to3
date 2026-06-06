import numpy as np
import os
import torch
from hmr4d.utils.phc.joints2smpl.smplify_loc2rotinitvit import joints2smpl


@torch.inference_mode(mode=False)
#def convert_pos2d_to_smpl(pos, device=0, cuda=True, num_smplify_iters=30):
def convert_pos2d_to_smpl_init_vit(pos,init, device=0, cuda=True, num_smplify_iters=150):
    """_summary_

    Args:
        pos (_type_): [F, J, 3]
        device (int, optional): _description_. Defaults to 0.
        cuda (bool, optional): _description_. Defaults to True.
        num_smplify_iters (int, optional): _description_. Defaults to 150.

    Returns:
        motion: [F, 3 + 22 * 3]
    """
    pos = pos.detach().clone()
    smplify = joints2smpl(num_frames=pos.shape[0], device_id=device, cuda=cuda, num_smplify_iters=num_smplify_iters)
    v,j,t,global_orient,body_pose,betas = smplify.joint2smpl(pos,init)  # [F, 3 + 24 * 3]
    return v,j,t,global_orient,body_pose,betas
