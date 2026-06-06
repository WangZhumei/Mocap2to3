import torch
import hmr4d.utils.matrix as matrix
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def get_camera_mat(mat, distance, angle):
    """_summary_
    We assume camera always rotate with distance and angle, points at the mat.
    We also assume z-axis is upward, x-axis is facing.

    Args:
        mat (Tensor): [*, 4, 4]
        pos (Tensor): [*]
        angle (Tensor):[*]
    """
    #  FIXME: not opencv coordinate
    # put z-axis on the ground (-x axis)
    y_axis = torch.zeros(angle.shape[:-1] + (3,), device=angle.device)
    y_axis[..., 1] = 1.0
    cam_rot = matrix.quat_from_angle_axis(-torch.pi / 2 + torch.zeros_like(angle), y_axis)  # [*, 4]
    cam_rotmat = matrix.rot_matrix_from_quaternion(cam_rot)  # [*, 3, 3]

    # now for camera, x-axis is upward
    x_axis = torch.zeros(angle.shape[:-1] + (3,), device=angle.device)
    x_axis[..., 0] = 1.0
    cam_rot_ = matrix.quat_from_angle_axis(angle, x_axis)  # [*, 4]
    cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
    cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    pos = torch.stack((torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), dim=-1)  # [*, 3]
    pos = pos * distance[..., None]
    cam_mat = matrix.get_TRS(cam_rotmat, pos)
    cam_mat = matrix.get_mat_BfromA(mat, cam_mat)
    return cam_mat
def get_camera_mat_zface_wzm(mat, distance, hor_angle,z_angle, elevation_angle=None, is_opencv=True):
    """_summary_
    We assume camera always rotate with distance, hor_angle, and elevation_angle, points at the mat.
    We also assume y-axis is upward, z-axis is facing.

    Args:
        mat (Tensor): [*, 4, 4]
        pos (Tensor): [*]
        hor_angle (Tensor):[*] #tensor([5.0621, 6.6329, 8.2037, 9.7745])
        elevation_angle (Tensor):[*] #tensor([0.0873])
    """
    # rotate to concentrate on human
    #print(hor_angle)
    #print(z_angle)
    #print(elevation_angle)
    #print(distance)
    #exit(0)
    y_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device) #tensor([0., 1., 0.]) #torch.Size([3])
    y_axis[..., 1] = 1.0
    cam_rot = matrix.quat_from_angle_axis(hor_angle, y_axis)  # [*, 4]
    cam_rotmat = matrix.rot_matrix_from_quaternion(cam_rot)  # [*, 3, 3]

    if elevation_angle is not None:
        x_axis = torch.zeros(elevation_angle.shape[:-1] + (3,), device=elevation_angle.device)
        x_axis[..., 0] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(elevation_angle, x_axis)  # [*, 4]
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    else:
        elevation_angle = torch.zeros_like(hor_angle)

    if is_opencv:
        # rotate to opencv
        z_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device)
        z_axis[..., 2] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(z_angle, z_axis)  # [*, 4] #tensor([3.1416, 3.1416, 3.1416, 3.1416])
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    xz_dist = torch.cos(elevation_angle).abs() * distance #([4.4829, 4.4829, 4.4829, 4.4829])
    pos = torch.stack(
        (-torch.sin(hor_angle) * xz_dist, torch.sin(elevation_angle) * distance, -torch.cos(hor_angle) * xz_dist),
        dim=-1,
    )  # [*, 3]
    cam_mat = matrix.get_TRS(cam_rotmat, pos)
    cam_mat = matrix.get_mat_BfromA(mat, cam_mat)
    return cam_mat
def get_gt2genCamera4_90_mat_wzm(mat,N_views=4):
    cam_mat = np.expand_dims(np.zeros_like(mat),axis=0).repeat(4,axis=0)
    R_m = mat[:3,:3]
    t_m = mat[:3,-1]
    tm_idx = [[0,1,2],[2,1,0],[0,1,2],[2,1,0]]
    tm_sym1 = [[1,1,1],[1,1,-1],[-1,1,-1],[-1,1,1]]
    #tm_sym2 = [[1,1,1],[-1,1,1],[-1,1,-1],[1,1,-1]]
    R_obj = R.from_matrix(R_m)
    #yaw, pitch, roll = R_obj.as_euler('ZYX', degrees=False)
    #yaw, pitch, roll = R_obj.as_euler('ZYX', degrees=False)
    pitch, roll ,yaw= R_obj.as_euler('YXZ', degrees=False)
    Z_r = [yaw for i in range(N_views)]
    interval = 2 * np.pi / N_views
    Y_r = [pitch + i * interval for i in range(N_views)] #增量固定
    X_r = [roll for i in range(N_views)]

    for i in range(N_views):
        #R_obj_recovered = R.from_euler('ZYX', [Z_r[i], Y_r[i], X_r[i]], degrees=False)
        R_obj_recovered = R.from_euler('YXZ', [Y_r[i], X_r[i],Z_r[i]], degrees=False)
        rotation_matrix_a = R_obj_recovered.as_matrix()
        cam_mat[i,:3,:3] = rotation_matrix_a
        t_n = t_m[tm_idx[i]]*tm_sym1[i]
        #if math.degrees(X_r[i])>110:#math.degrees(X_r)<-110 or  
        #    t_n = t_m[tm_idx[i]]*tm_sym1[i]
        #else:
        #    t_n = t_m[tm_idx[i]]*tm_sym2[i]
        cam_mat[i,:3,-1] = t_n
        cam_mat[i,-1,-1] = 1
    #for i in range(4):
    #    print(math.degrees(Z_r[i]), math.degrees(Y_r[i]), math.degrees(X_r[i]))
    return cam_mat
def rotate_point(x, y, theta):
    #theta = math.radians(theta)  # 将角度转换为弧度
    # 使用复数进行计算
    rotated_point = complex(x, y) * complex(math.cos(theta), math.sin(theta))
    return rotated_point.real, rotated_point.imag
def get_gt2genCamera4_Y_mat_wzm(mat,angle,angle_plus,eleva_angle,z_angle,distance,N_views=4): #逆时针增加
    cam_mat = np.expand_dims(np.zeros_like(mat),axis=0).repeat(N_views,axis=0)

    R_m = mat[:3,:3]
    t_m = mat[:3,-1]
    

    Z_r = z_angle.numpy()
    Y_r = angle.numpy()
    Y_r_p = angle_plus.numpy()
    X_r = eleva_angle.numpy()

    x, z = t_m[0], t_m[2]  # 原点为(0, 0)，我们选择点为(1, 0)

    for i in range(N_views):
        #R_obj_recovered = R.from_euler('ZYX', [Z_r[i], Y_r[i], X_r[i]], degrees=False)
        R_obj_recovered = R.from_euler('YXZ', [Y_r[i], X_r[i],Z_r[i]], degrees=False)
        rotation_matrix_a = R_obj_recovered.as_matrix()
        cam_mat[i,:3,:3] = rotation_matrix_a
        theta = -Y_r_p[i]#math.degrees(Y_r_p[i])
        new_x, new_z = rotate_point(x, z, theta)
        t_n = np.array([new_x,t_m[1],new_z])
        #t_n = t_m[tm_idx[i]]*tm_sym1[i]
        cam_mat[i,:3,-1] = t_n
        cam_mat[i,-1,-1] = 1
    return cam_mat
def get_camera_mat_zface(mat, distance, hor_angle, elevation_angle=None, is_opencv=True):
    """_summary_
    We assume camera always rotate with distance, hor_angle, and elevation_angle, points at the mat.
    We also assume y-axis is upward, z-axis is facing.

    Args:
        mat (Tensor): [*, 4, 4]
        pos (Tensor): [*]
        hor_angle (Tensor):[*] #tensor([5.0621, 6.6329, 8.2037, 9.7745])
        elevation_angle (Tensor):[*] #tensor([0.0873])
    """
    # rotate to concentrate on human
    y_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device) #tensor([0., 1., 0.]) #torch.Size([3])
    y_axis[..., 1] = 1.0
    cam_rot = matrix.quat_from_angle_axis(hor_angle, y_axis)  # [*, 4]
    cam_rotmat = matrix.rot_matrix_from_quaternion(cam_rot)  # [*, 3, 3]

    if elevation_angle is not None:
        x_axis = torch.zeros(elevation_angle.shape[:-1] + (3,), device=elevation_angle.device)
        x_axis[..., 0] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(elevation_angle, x_axis)  # [*, 4]
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    else:
        elevation_angle = torch.zeros_like(hor_angle)

    if is_opencv:
        # rotate to opencv
        z_axis = torch.zeros(hor_angle.shape[:-1] + (3,), device=hor_angle.device)
        z_axis[..., 2] = 1.0
        cam_rot_ = matrix.quat_from_angle_axis(torch.pi + torch.zeros_like(hor_angle), z_axis)  # [*, 4] #tensor([3.1416, 3.1416, 3.1416, 3.1416])
        cam_rotmat_ = matrix.rot_matrix_from_quaternion(cam_rot_)  # [*, 3, 3]
        cam_rotmat = matrix.get_mat_BfromA(cam_rotmat, cam_rotmat_)  # [*, 3, 3]
    xz_dist = torch.cos(elevation_angle).abs() * distance #([4.4829, 4.4829, 4.4829, 4.4829])
    pos = torch.stack(
        (-torch.sin(hor_angle) * xz_dist, torch.sin(elevation_angle) * distance, -torch.cos(hor_angle) * xz_dist),
        dim=-1,
    )  # [*, 3]
    '''
    tensor([[ 4.2115,  0.3922, -1.5360],
        [-1.5360,  0.3922, -4.2115],
        [-4.2115,  0.3922,  1.5360],
        [ 1.5360,  0.3922,  4.2115]])
    torch.Size([4, 3])
    '''
    cam_mat = matrix.get_TRS(cam_rotmat, pos)
    cam_mat = matrix.get_mat_BfromA(mat, cam_mat)
    return cam_mat
    '''
    [[[-3.4264e-01,  8.1880e-02, -9.3589e-01,  4.2115e+00],
         [-8.7090e-08, -9.9619e-01, -8.7156e-02,  3.9220e-01],
         [-9.3947e-01, -2.9863e-02,  3.4133e-01, -1.5360e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.3947e-01, -2.9863e-02,  3.4134e-01, -1.5360e+00],
         [-8.7090e-08, -9.9619e-01, -8.7156e-02,  3.9220e-01],
         [ 3.4264e-01, -8.1880e-02,  9.3589e-01, -4.2115e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 3.4264e-01, -8.1880e-02,  9.3589e-01, -4.2115e+00],
         [-8.7090e-08, -9.9619e-01, -8.7156e-02,  3.9220e-01],
         [ 9.3947e-01,  2.9863e-02, -3.4133e-01,  1.5360e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.3947e-01,  2.9863e-02, -3.4133e-01,  1.5360e+00],
         [-8.7090e-08, -9.9619e-01, -8.7156e-02,  3.9220e-01],
         [-3.4264e-01,  8.1880e-02, -9.3589e-01,  4.2115e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]
    '''


def cartesian_to_spherical(xyz):
    """_summary_
    From zero1to3: https://github.com/cvlab-columbia/zero123/blob/main/zero123/ldm/data/simple.py#L248
    Given a position in cartesian coordinates, return the spherical coordinates
    Because we assume camera is pointing at the center, so do not need camera rotation.
    Original code assumes z is upward, we modify it to y is upward.

    ###################### Original Code ################################
    xy = xyz[..., 0] ** 2 + xyz[..., 1] ** 2
    z = torch.sqrt(xy + xyz[..., 2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[..., 1], xyz[..., 0])
    #####################################################################

    Args:
        xyz (_tensor_): [..., 3]

    Returns:
        _(..., 3) spherical coordinate
    """
    xy = xyz[..., :1] ** 2 + xyz[..., 2:3] ** 2
    z = torch.sqrt(xy + xyz[..., 1:2] ** 2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[..., 1:2])  # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[..., 2:3], xyz[..., :1])
    return torch.cat([theta, azimuth, z], dim=-1)
