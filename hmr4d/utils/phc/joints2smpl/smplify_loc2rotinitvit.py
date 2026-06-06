import os
import torch
from hmr4d.utils.phc.joints2smpl.src import config
import smplx
import h5py
from hmr4d.utils.phc.joints2smpl.src.smplifyGVvit import SMPLify2D_GV_vit
import hmr4d.utils.phc.joints2smpl.rotation_conversions as geometry
import argparse
#from hmr4d.utils.smplx_utils import make_smplx


class joints2smpl:
    def __init__(self, num_frames, device_id, cuda=True, num_smplify_iters=150):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        self.batch_size = num_frames
        self.num_joints = 17  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = num_smplify_iters
        self.fix_foot = False
        # self.fit_h36m = False
        # self.h36m_cond = [1,2,3,9,10,11,13,14,15]
        print(config.SMPL_MODEL_DIR)
        smplmodel = smplx.create(
           config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=self.batch_size
        ).to(self.device)
        #smplmodel = make_smplx("rich-smplx", gender="neutral").to(self.device)
        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE

        file = h5py.File(smpl_mean_file, "r")
        self.init_mean_pose = (
            torch.from_numpy(file["pose"][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        )
        self.init_mean_shape = (
            torch.from_numpy(file["shape"][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        )
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify2D_GV_vit(
            smplxmodel=smplmodel,
            batch_size=self.batch_size,
            joints_category=self.joint_category,
            num_iters=self.num_smplify_iters,
            device=self.device,
        )

    def joint2smpl(self, input_joints, init_params=None):
        _smplify = self.smplify  # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 69).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_2d = torch.zeros(self.batch_size, self.num_joints, 2).to(self.device)

        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_2d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params["betas"]
            pred_pose[:,:63] = init_params["body_pose"]
            pred_orient = init_params["global_orient"]
            pred_cam_t = init_params["transl"]
            focal_length = init_params['focal_length']
            camera_center = init_params['camera_center']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.batch_size,self.num_joints)
            
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss,global_orient,body_pose,betas = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            pred_orient.detach(),
            focal_length.detach(),
            camera_center.detach(),
            keypoints_2d,
            conf_2d=confidence_input.to(self.device),
        )

        # thetas = new_opt_pose.reshape(self.batch_size, -1)
        # root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        # thetas = torch.cat([root_loc, thetas], dim=1)

        return new_opt_vertices,new_opt_joints,new_opt_cam_t,global_orient,body_pose,betas

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Blender file or dir with blender files")
    parser.add_argument("--cuda", type=bool, default=True, help="")
    parser.add_argument("--device", type=int, default=0, help="")
    params = parser.parse_args()

    simplify = joints2smpl(device_id=params.device, cuda=params.cuda)

    if os.path.isfile(params.input_path) and params.input_path.endswith(".npy"):
        simplify.npy2smpl(params.input_path)
    elif os.path.isdir(params.input_path):
        files = [os.path.join(params.input_path, f) for f in os.listdir(params.input_path) if f.endswith(".npy")]
        for f in files:
            simplify.npy2smpl(f)
