import argparse
import re
import shutil
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from hmr4d.utils.smplx_utils import make_smplx


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def parse_device_list(raw):
    if not raw:
        return [0]
    if isinstance(raw, list):
        return raw
    return [int(item) for item in str(raw).split(",") if item.strip()]


def collect_npy_paths(npy_path=None, dir_path=None):
    if (npy_path is None) == (dir_path is None):
        raise ValueError("Specify exactly one of `--npy` or `--dir`.")

    if npy_path is not None:
        path = Path(npy_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]

    input_dir = Path(dir_path)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    return sorted(input_dir.glob("*.npy"), key=natural_key)


def ensure_faces_path(faces_path=None):
    if faces_path is not None:
        path = Path(faces_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    assets_dir = Path("inputs/joints2smpl/smpl_models")
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / "smplh_faces.npy"
    if path.is_file():
        return path

    faces = np.asarray(make_smplx("rich-smplh", gender="neutral").bm.faces, dtype=np.int32)
    np.save(path, faces)
    return path


def resolve_output_root(args, npy_paths):
    if args.output_dir:
        root = Path(args.output_dir)
    elif args.npy:
        root = npy_paths[0].parent
    else:
        root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def whiten_alpha(frame):
    if frame.ndim == 3 and frame.shape[-1] == 4:
        mask = frame[:, :, 3] < 1
        frame = frame.copy()
        frame[mask] = 255
        frame = frame[:, :, :3]
    return frame


def save_video_from_frames(frames_dir, out_path, fps):
    frame_paths = sorted(frames_dir.glob("frame_*.png"), key=natural_key)
    if not frame_paths:
        raise FileNotFoundError(f"No rendered frames found in {frames_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame_path in frame_paths:
            frame = whiten_alpha(imageio.imread(frame_path))
            writer.append_data(frame)
    finally:
        writer.close()


def build_blender_command(args, blender_script, faces_path):
    cmd = [
        args.blender,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--mode",
        args.mode,
        "--joint-type",
        args.joint_type,
        "--output-dir",
        str(args.output_root),
        "--faces-path",
        str(faces_path),
        "--res",
        args.res,
        "--accelerator",
        args.accelerator,
        "--device",
        ",".join(str(x) for x in args.device),
    ]
    if args.npy:
        cmd.extend(["--npy", args.npy])
    else:
        cmd.extend(["--dir", args.dir])
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.canonicalize:
        cmd.append("--no-canonicalize")
    if args.always_on_floor:
        cmd.append("--always-on-floor")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Render motion npy files with Blender and save final videos.")
    parser.add_argument("--blender", default="blender", help="Path to the Blender executable.")
    parser.add_argument("--npy", default=None, help="Path to a single motion npy file.")
    parser.add_argument("--dir", default=None, help="Directory that contains motion npy files.")
    parser.add_argument("--output-dir", default=None, help="Directory for rendered outputs. Defaults to the input location.")
    parser.add_argument("--faces-path", default=None, help="Optional precomputed SMPLH faces .npy file.")
    parser.add_argument("--mode", default="video", choices=["video", "sequence"], help="Render target.")
    parser.add_argument("--joint-type", default="HumanML3D", choices=["HumanML3D", "COCO"], help="Skeleton type for joint npy rendering.")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS used when encoding videos.")
    parser.add_argument("--res", default="high", choices=["low", "med", "high", "ultra"], help="Render resolution preset.")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"], help="Blender Cycles device type.")
    parser.add_argument("--device", default="0", help="Visible Blender device ids, e.g. `0` or `0,1`.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render outputs even if they already exist.")
    parser.add_argument("--keep-frames", action="store_true", help="Keep rendered frame folders after mp4 encoding.")
    parser.add_argument("--always-on-floor", action="store_true", help="Force each frame to stay on the floor.")
    parser.add_argument("--no-canonicalize", dest="canonicalize", action="store_false", help="Disable canonicalization for joint inputs.")
    parser.set_defaults(canonicalize=True)
    args = parser.parse_args()

    args.device = parse_device_list(args.device)
    npy_paths = collect_npy_paths(args.npy, args.dir)
    args.output_root = resolve_output_root(args, npy_paths)
    faces_path = ensure_faces_path(args.faces_path)

    blender_script = Path(__file__).resolve().with_name("render_blender.py")
    cmd = build_blender_command(args, blender_script, faces_path)
    subprocess.run(cmd, check=True)

    if args.mode != "video":
        return

    for npy_path in npy_paths:
        frames_dir = args.output_root / f"{npy_path.stem}_frames"
        video_path = args.output_root / f"{npy_path.stem}.mp4"

        if video_path.exists() and not args.overwrite:
            continue
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Expected frame directory not found: {frames_dir}")

        save_video_from_frames(frames_dir, video_path, fps=args.fps)
        if not args.keep_frames:
            shutil.rmtree(frames_dir)


if __name__ == "__main__":
    main()
