import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hmr4d.utils.net_utils import trusted_torch_load
from hmr4d.utils.phc.pos2smpl import convert_pos_to_smpl
from hmr4d.utils.smplx_utils import make_smplx


def align_pcl(y, x, weight=None, fixed_scale=False):
    """Align X to Y with a similarity transform on the current tensor device."""
    *dims, n_points, _ = y.shape
    n_points = torch.ones(*dims, 1, 1, device=y.device, dtype=y.dtype) * n_points

    if weight is not None:
        y = y * weight
        x = x * weight
        n_points = weight.sum(dim=-2, keepdim=True)

    mean_y = y.sum(dim=-2) / n_points[..., 0]
    mean_x = x.sum(dim=-2) / n_points[..., 0]
    y0 = y - mean_y[..., None, :]
    x0 = x - mean_x[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    corr = torch.matmul(y0.transpose(-1, -2), x0) / n_points
    u, d, vh = torch.linalg.svd(corr)

    s_mat = torch.eye(3, device=y.device, dtype=y.dtype).reshape(*(1,) * len(dims), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(u) * torch.det(vh.transpose(-1, -2)) < 0
    s_mat[neg, 2, 2] = -1

    rot = torch.matmul(u, torch.matmul(s_mat, vh))
    if fixed_scale:
        scale = torch.ones(*dims, 1, device=y.device, dtype=y.dtype)
    else:
        d = torch.diag_embed(d)
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / n_points
        scale = torch.diagonal(torch.matmul(d, s_mat), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]

    trans = mean_y - scale * torch.matmul(rot, mean_x[..., None])[..., 0]
    return scale, rot, trans


def align_joints(gt_joints, pred_joints, pred_vertices):
    """Align one fitted frame back to the predicted joints of the same frame."""
    scale, rot, trans = align_pcl(gt_joints[:1].reshape(1, -1, 3), pred_joints[:1].reshape(1, -1, 3))
    pred_joints = scale * torch.einsum("tij,tnj->tni", rot, pred_joints) + trans[:, None]
    pred_vertices = scale * torch.einsum("tij,tnj->tni", rot, pred_vertices) + trans[:, None]
    return pred_joints, pred_vertices


def to_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def sanitize_seq_name(seq_name):
    return seq_name.replace("\\", "/").strip("/")


def make_camera_subseq_name(seq_name):
    seq_name = sanitize_seq_name(seq_name)
    parts = seq_name.split("/")

    if len(parts) >= 2 and parts[-1].startswith("cam_"):
        return f"{parts[-2]}_{parts[-1]}"

    if len(parts) >= 2:
        seq_base = parts[0]
        cam_part = parts[-1]
        if "_" in cam_part:
            cam_idx = cam_part.split("_")[-1]
            if cam_idx.isdigit():
                return f"{seq_base}_cam_{int(cam_idx):02d}"

    return seq_name.replace("/", "_")


def fit_subseq_vertices(joints, smplh_model, num_smplify_iters, device_id):
    joints3d = joints.float().cuda(device_id)
    smpl_pose = convert_pos_to_smpl(joints3d, device=device_id, cuda=True, num_smplify_iters=num_smplify_iters)
    transl = smpl_pose[:, :3]
    global_orient = smpl_pose[:, 3:6]
    body_pose = smpl_pose[:, 6:]

    smpl_out = smplh_model(transl=transl, global_orient=global_orient, body_pose=body_pose)
    vertices = smpl_out.vertices
    fit_joints = smpl_out.joints[:, :22]

    vertices[..., :2] = vertices[..., :2] / 1.2
    fit_joints[..., :2] = fit_joints[..., :2] / 1.2

    floor = vertices[:, :, 1].min()
    vertices[..., 1] -= floor
    fit_joints[..., 1] -= floor

    aligned_vertices = []
    for frame_idx in range(joints3d.shape[0]):
        _, aligned_v = align_joints(
            joints3d[frame_idx : frame_idx + 1],
            fit_joints[frame_idx : frame_idx + 1],
            vertices[frame_idx : frame_idx + 1],
        )
        aligned_vertices.append(aligned_v[0])

    return torch.stack(aligned_vertices, dim=0)


def iter_subseq_entries(pred):
    for seq_name, subseq_dict in pred.items():
        for subseq_key, sample in subseq_dict.items():
            yield seq_name, subseq_key, sample


def main():
    parser = argparse.ArgumentParser(
        description="Fit SMPLH vertices for each RICH sub-seq and export camera_subseq npy files."
    )
    parser.add_argument("--input", default="./res/rich_smpl.pth", help="Prediction file that stores sub-seq joints.")
    parser.add_argument(
        "--output-dir",
        default="./render_result/npy/camera_subseq/ours",
        help="Directory for exported sub-seq vertex npy files.",
    )
    parser.add_argument("--num-smplify-iters", type=int, default=150, help="Number of SMPLify optimization steps.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id used for SMPL fitting.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tools/mas/joints2smpl.py")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = trusted_torch_load(args.input, map_location="cpu")
    smplh_model = make_smplx("rich-smplh", gender="neutral").cuda(args.device)
    total_subseq = sum(len(subseq_dict) for subseq_dict in pred.values())

    for seq_name, subseq_key, sample in tqdm(iter_subseq_entries(pred), total=total_subseq, desc="Export sub-seq vertices"):
        if "pred_ayfz_motion" not in sample:
            raise KeyError(f"Missing `pred_ayfz_motion` in seq `{seq_name}`, sub-seq `{subseq_key}`.")

        joints = sample["pred_ayfz_motion"]
        start_frame = to_int(sample.get("start_frame", subseq_key))
        file_name = make_camera_subseq_name(seq_name)

        vertices = fit_subseq_vertices(joints, smplh_model, args.num_smplify_iters, args.device)
        save_path = output_dir / f"{file_name}_{start_frame}.npy"
        np.save(save_path, vertices.cpu().numpy())


if __name__ == "__main__":
    main()
