import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
except ImportError as exc:
    raise ImportError("This script must be executed inside Blender.") from exc


HUMANML3D_JOINTS = [
    "root", "RH", "LH", "BP", "RK", "LK", "BT", "RMrot", "LMrot", "BLN", "RF", "LF",
    "BMN", "RSI", "LSI", "BUN", "RS", "LS", "RE", "LE", "RW", "LW",
]
HUMANML3D_KINEMATIC_TREE = [
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
]
COCO_JOINTS = [
    "root", "BUN", "REY", "LEY", "REA", "LEA", "RS", "LS", "RE", "LE",
    "RW", "LW", "RH", "LH", "RK", "LK", "RMrot", "LMrot", "BMN",
]
COCO_KINEMATIC_TREE = [
    [0, 18, 1],
    [18, 7, 9, 11],
    [18, 6, 8, 10],
    [0, 13, 15, 17],
    [0, 12, 14, 16],
]
MMM_TO_SMPLH_SCALING_FACTOR = 0.75 / 480


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render motion npy files inside Blender.")
    parser.add_argument("--npy", default=None, help="Path to a single motion npy file.")
    parser.add_argument("--dir", default=None, help="Directory that contains motion npy files.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered outputs.")
    parser.add_argument("--faces-path", required=True, help="SMPLH faces .npy file for mesh rendering.")
    parser.add_argument("--mode", default="video", choices=["video", "sequence"], help="Render target.")
    parser.add_argument("--joint-type", default="HumanML3D", choices=["HumanML3D", "COCO"], help="Skeleton type for joint inputs.")
    parser.add_argument("--res", default="high", choices=["low", "med", "high", "ultra"], help="Render resolution preset.")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"], help="Blender Cycles device type.")
    parser.add_argument("--device", default="0", help="Visible Blender device ids, e.g. `0` or `0,1`.")
    parser.add_argument("--overwrite", action="store_true", help="Re-render outputs even if they already exist.")
    parser.add_argument("--always-on-floor", action="store_true", help="Force each frame to stay on the floor.")
    parser.add_argument("--no-canonicalize", dest="canonicalize", action="store_false", help="Disable canonicalization for joint inputs.")
    parser.set_defaults(canonicalize=True)
    return parser.parse_args(argv)


def parse_device_list(raw):
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


def mesh_detect(data):
    return data.shape[1] > 1000


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def colored_material_diffuse_bsdf(r, g, b, a=1, roughness=0.127451):
    material = bpy.data.materials.new(name="body")
    material.use_nodes = True
    clear_material(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (r, g, b, a)
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def colored_material_reflection_bsdf(r, g, b, a=1, roughness=0.127451, saturation_factor=1.0):
    material = bpy.data.materials.new(name="body")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type="ShaderNodeOutputMaterial")
    diffuse = nodes["Principled BSDF"]
    diffuse.inputs["Base Color"].default_value = (
        r * saturation_factor,
        g * saturation_factor,
        b * saturation_factor,
        a,
    )
    diffuse.inputs["Roughness"].default_value = roughness
    links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def body_material(r, g, b, a=1):
    return colored_material_diffuse_bsdf(r, g, b, a=a)


def floor_material(color=(0.1, 0.1, 0.1, 1), roughness=0.127451):
    return colored_material_diffuse_bsdf(color[0], color[1], color[2], a=color[3], roughness=roughness)


GEN_SMPL = body_material(0.2365, 0.05, 0.9686)
JOINT_MATERIALS = [
    colored_material_reflection_bsdf(0.7647, 0.1843, 0.1529, saturation_factor=1.1),
    colored_material_reflection_bsdf(1.0, 0.7647, 0.0, saturation_factor=1.1),
    colored_material_reflection_bsdf(0.6863, 0.9882, 0.2549, saturation_factor=1.1),
    colored_material_reflection_bsdf(0.016, 0.3, 0.884, saturation_factor=1.1),
    colored_material_reflection_bsdf(0.102, 0.459, 0.624, saturation_factor=1.1),
    colored_material_reflection_bsdf(0.3, 0.3, 0.3, saturation_factor=1.1),
]


class ndarray_pydata(np.ndarray):
    def __bool__(self):
        return len(self) > 0


def delete_objects(names):
    if not isinstance(names, list):
        names = [names]
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        for name in names:
            if obj.name.startswith(name) or obj.name.endswith(name):
                obj.select_set(True)
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action="DESELECT")


def load_numpy_vertices_into_blender(vertices, faces, name, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces.view(ndarray_pydata))
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    obj.active_material = material
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    bpy.ops.object.select_all(action="DESELECT")


def setup_renderer(denoising=True, accelerator="gpu", device=None):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.data.scenes[0].render.engine = "CYCLES"
    if accelerator.lower() == "gpu":
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for idx, dev in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
            dev["use"] = 1 if idx in device else 0

    if denoising:
        bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.samples = 64


def setup_scene(res="high", denoising=True, accelerator="gpu", device=None):
    scene = bpy.data.scenes["Scene"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 640
        scene.render.resolution_y = 512
    elif res == "low":
        scene.render.resolution_x = 320
        scene.render.resolution_y = 256
    elif res == "ultra":
        scene.render.resolution_x = 2560
        scene.render.resolution_y = 2048

    scene.render.film_transparent = True
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    if "Sun" not in bpy.data.objects:
        bpy.ops.object.light_add(type="SUN", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = 1.5
    setup_renderer(denoising=denoising, accelerator=accelerator, device=device)


class Camera:
    def __init__(self, first_root, mode, is_mesh):
        camera = bpy.data.objects["Camera"]
        camera.location.x = 7.36
        camera.location.y = -6.93
        camera.location.z = 5.6 if is_mesh else 5.2

        if mode == "sequence":
            camera.data.lens = 65 if is_mesh else 85
        elif mode == "frame":
            camera.data.lens = 130 if is_mesh else 85
        else:
            camera.data.lens = 110 if is_mesh else 85

        self.camera = camera
        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]
        self.root = first_root

    def update(self, new_root):
        delta_root = new_root - self.root
        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]
        self.root = new_root


def get_frame_indices(mode, nframes):
    if mode in {"video", "sequence"}:
        return range(0, nframes)
    raise ValueError(f"Unsupported mode: {mode}")


def plot_floor(data, big_plane=True):
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    location = ((maxx + minx) / 2, (maxy + miny) / 2, 0)
    scale = (1.08 * (maxx - minx) / 2, 1.08 * (maxy - miny) / 2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align="WORLD", location=location, scale=(1, 1, 1))
    bpy.ops.transform.resize(value=scale)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"
    obj.active_material = floor_material(color=(0.5, 0.5, 0.5, 1) if big_plane else (0.35, 0.35, 0.35, 1))

    if big_plane:
        location = ((maxx + minx) / 2, (maxy + miny) / 2, -0.02)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align="WORLD", location=location, scale=(1, 1, 1))
        bpy.ops.transform.resize(value=[1.2 * x for x in scale])
        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        obj.active_material = floor_material(color=(0.15, 0.15, 0.15, 1))


def show_trajectory(coords):
    curve_data = bpy.data.curves.new("myCurve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2
    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(coords) - 1)
    for idx, coord in enumerate(coords):
        x, y = coord
        polyline.points[idx].co = (x, y, 0.001, 1)
    curve_obj = bpy.data.objects.new("myCurve", curve_data)
    curve_data.bevel_depth = 0.01
    orange_mat = bpy.data.materials.new(name="OrangeMaterial")
    orange_mat.diffuse_color = (1.0, 0.5, 0.0, 1.0)
    curve_obj.data.materials.append(orange_mat)
    bpy.context.collection.objects.link(curve_obj)


def render_current_frame(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(use_viewport=True, write_still=True)


def prepare_meshes(data, always_on_floor=False):
    data = data[..., [2, 0, 1]]
    data[..., 2] -= data[..., 2].min()
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]
    return data


class Meshes:
    def __init__(self, data, faces_path, mode, always_on_floor=False):
        self.data = prepare_meshes(data, always_on_floor=always_on_floor)
        self.faces = np.load(faces_path)
        self.mode = mode
        self.trajectory = self.data[:, :, [0, 1]].mean(1)
        self.material = GEN_SMPL

    def __len__(self):
        return len(self.data)

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index):
        name = f"{index:04d}"
        load_numpy_vertices_into_blender(self.data[index], self.faces, name, self.material)
        return name


def softmax(x, softness=1.0, axis=None):
    maxi = x.max(axis=axis)
    mini = x.min(axis=axis)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, softness=1.0, axis=0):
    return -softmax(-x, softness=softness, axis=axis)


def matrix_of_angles(cos, sin, inv=False):
    sin = -sin if inv else sin
    return np.stack((np.stack((cos, -sin), axis=-1), np.stack((sin, cos), axis=-1)), axis=-2)


def get_forward_direction(poses, joint_type):
    if joint_type == "humanml3d":
        joints = HUMANML3D_JOINTS
    elif joint_type == "coco":
        joints = COCO_JOINTS
    else:
        raise ValueError(joint_type)
    ls, rs = joints.index("LS"), joints.index("RS")
    lh, rh = joints.index("LH"), joints.index("RH")
    across = (poses[..., rh, :] - poses[..., lh, :] + poses[..., rs, :] - poses[..., ls, :])
    forward = np.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    return forward


def get_floor(poses, joint_type):
    joints = HUMANML3D_JOINTS if joint_type == "humanml3d" else COCO_JOINTS
    if joint_type == "coco":
        lm, rm = joints.index("LMrot"), joints.index("RMrot")
        foot_heights = poses[..., (lm, rm), 1].min(-1)
    else:
        lm, rm = joints.index("LMrot"), joints.index("RMrot")
        lf, rf = joints.index("LF"), joints.index("RF")
        foot_heights = poses[..., (lm, lf, rm, rf), 1].min(-1)
    floor_height = softmin(foot_heights, softness=0.5, axis=-1)
    ndim = len(poses.shape)
    return floor_height[tuple((ndim - 2) * [None])].T


def canonicalize_joints(joints, joint_type):
    poses = joints.copy()
    translation = joints[..., 0, :].copy()
    translation[..., 1] = 0
    trajectory = translation[..., [0, 2]]
    poses[..., 1] -= get_floor(poses, joint_type)
    poses[..., [0, 2]] -= trajectory[..., None, :]
    trajectory = trajectory - trajectory[..., 0, :]
    forward = get_forward_direction(poses[..., 0, :, :], joint_type)
    sin, cos = forward[..., 0], forward[..., 1]
    rotations_inv = matrix_of_angles(cos, sin, inv=True)
    trajectory_rotated = np.einsum("...j,...jk->...k", trajectory, rotations_inv)
    poses_rotated = np.einsum("...lj,...jk->...lk", poses[..., [0, 2]], rotations_inv)
    poses_rotated = np.stack((poses_rotated[..., 0], poses[..., 1], poses_rotated[..., 1]), axis=-1)
    poses_rotated[..., (0, 2)] += trajectory_rotated[..., None, :]
    return poses_rotated


def prepare_joints(joints, joint_type, canonicalize=True, always_on_floor=False):
    data = canonicalize_joints(joints, joint_type) if canonicalize else joints.copy()
    data = data * MMM_TO_SMPLH_SCALING_FACTOR
    data = data[..., [2, 0, 1]]
    data -= data[[0], [0], :]
    data[..., 2] -= data[..., 2].min()
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]
    return data


def sphere(radius, location, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=50, ring_count=50, radius=radius, location=location)
    bpy.context.object.active_material = material


def sphere_between(t1, t2, material, factor=1.0):
    x1, y1, z1 = t1
    x2, y2, z2 = t2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * factor
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=50,
        ring_count=50,
        radius=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1),
    )
    bpy.context.object.active_material = material


def cylinder_between(t1, t2, radius, material):
    x1, y1, z1 = t1
    x2, y2, z2 = t2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=dist, location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1))
    phi = math.atan2(dy, dx)
    theta = math.acos(dz / dist)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    bpy.context.object.active_material = material
    sphere(radius, (x1, y1, z1), material)
    sphere(radius, (x2, y2, z2), material)


def cylinder_sphere_between(t1, t2, radius, material):
    x1, y1, z1 = t1
    x2, y2, z2 = t2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2) - 0.2 * radius
    phi = math.atan2(dy, dx)
    theta = math.acos(dz / (dist + 0.2 * radius))
    sphere(radius * 0.9, t1, material)
    sphere(radius * 0.9, t2, material)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=dist,
        location=(dx / 2 + x1, dy / 2 + y1, dz / 2 + z1),
        enter_editmode=True,
    )
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_face_by_sides(number=32, extend=False)
    bpy.ops.mesh.bevel(offset=radius, segments=8)
    bpy.ops.object.editmode_toggle(False)
    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi
    bpy.context.object.active_material = material


class Joints:
    def __init__(self, data, joint_type, mode, canonicalize=True, always_on_floor=False):
        self.joint_type = joint_type
        self.data = prepare_joints(data, joint_type, canonicalize=canonicalize, always_on_floor=always_on_floor)
        self.mode = mode
        self.trajectory = self.data[:, 0, [0, 1]]
        if joint_type == "humanml3d":
            self.kinematic_tree = HUMANML3D_KINEMATIC_TREE
            self.joints = HUMANML3D_JOINTS
        else:
            self.kinematic_tree = COCO_KINEMATIC_TREE
            self.joints = COCO_JOINTS

    def __len__(self):
        return len(self.data)

    def get_root(self, index):
        return self.data[index][0]

    def get_mean_root(self):
        return self.data[:, 0].mean(0)

    def load_in_blender(self, index):
        skeleton = self.data[index]
        head_mat = JOINT_MATERIALS[0]
        body_mat = JOINT_MATERIALS[-1]
        for chain, material in zip(self.kinematic_tree, JOINT_MATERIALS):
            for j1, j2 in zip(chain[:-1], chain[1:]):
                joint_name = self.joints[j2]
                if joint_name in ["BUN"]:
                    if self.joint_type == "coco":
                        sphere(0.08, skeleton[self.joints.index("BUN")], head_mat)
                    else:
                        sphere_between(skeleton[j1], skeleton[j2], head_mat)
                elif joint_name in ["LE", "RE", "LW", "RW"]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040, material)
                elif joint_name in ["LMrot", "RMrot", "RK", "LK"]:
                    cylinder_sphere_between(skeleton[j1], skeleton[j2], 0.040, material)
                elif joint_name in ["LS", "RS", "LF", "RF", "LH", "RH"]:
                    cylinder_between(skeleton[j1], skeleton[j2], 0.040, material)

        if self.joint_type == "coco":
            bmn_index = self.joints.index("BMN")
            root_index = self.joints.index("root")
            cylinder_between(skeleton[bmn_index], skeleton[root_index], 0.06, body_mat)
        else:
            sphere(0.14, skeleton[self.joints.index("BLN")], body_mat)
            sphere_between(
                skeleton[self.joints.index("BLN")],
                skeleton[self.joints.index("root")],
                body_mat,
                factor=0.28,
            )
            sphere(0.11, skeleton[self.joints.index("root")], body_mat)
        return ["Cylinder", "Sphere"]


def render_motion(data, output_target, mode, faces_path, joint_type, canonicalize, always_on_floor):
    is_mesh = mesh_detect(data)
    if is_mesh:
        motion = Meshes(data, faces_path=faces_path, mode=mode, always_on_floor=always_on_floor)
    else:
        motion = Joints(
            data,
            joint_type=joint_type.lower(),
            mode=mode,
            canonicalize=canonicalize,
            always_on_floor=always_on_floor,
        )

    show_trajectory(motion.trajectory)
    plot_floor(motion.data, big_plane=True)
    camera = Camera(first_root=motion.get_root(0), mode=mode, is_mesh=is_mesh)
    frame_indices = get_frame_indices(mode=mode, nframes=len(motion))
    imported_names = []

    if mode == "video":
        output_target.mkdir(parents=True, exist_ok=True)

    for render_index, frame_index in enumerate(frame_indices):
        if mode == "video":
            camera.update(motion.get_root(frame_index))
        objname = motion.load_in_blender(frame_index)
        frame_path = output_target / f"frame_{render_index:04d}.png" if mode == "video" else output_target
        render_current_frame(frame_path)
        delete_objects(objname)
        if mode == "sequence":
            imported_names.extend(objname)

    delete_objects(imported_names)
    delete_objects(["Plane", "SmallPlane", "BigPlane", "myCurve", "Cylinder", "Sphere"])


def main():
    args = parse_argv()
    device = parse_device_list(args.device)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    npy_paths = collect_npy_paths(args.npy, args.dir)

    setup_scene(res=args.res, accelerator=args.accelerator, device=device)

    for npy_path in npy_paths:
        target = output_root / (f"{npy_path.stem}_frames" if args.mode == "video" else f"{npy_path.stem}.png")
        if args.mode == "video":
            has_output = target.is_dir() and any(target.glob("frame_*.png"))
        else:
            has_output = target.is_file()
        if has_output and not args.overwrite:
            continue

        data = np.load(npy_path)
        render_motion(
            data,
            output_target=target,
            mode=args.mode,
            faces_path=args.faces_path,
            joint_type=args.joint_type,
            canonicalize=args.canonicalize,
            always_on_floor=args.always_on_floor,
        )


if __name__ == "__main__":
    main()
