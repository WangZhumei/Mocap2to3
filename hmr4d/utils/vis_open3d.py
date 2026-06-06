
import numpy as np
import open3d
import copy

class ModelPose:
    def __init__(self):
        self.points3D = []
        self.__vis = None

    def read_model(self, joints_pos):
        self.points3D = joints_pos
        a = 1

    def add_points(self, min_track_len=3, remove_statistical_outlier=False):


        # 绘制open3d坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__vis.add_geometry(axis_pcd)

        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.points3D: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,0,1])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        
        a = 1
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # a = 1
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    
    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # render_options = self.__vis.get_render_option()
        self.__vis.get_render_option().point_size = 5
        # render_options.point_size = 1.5  # 设置点的大小

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

class ModelPose2:
    def __init__(self):
        self.points3D = []
        self.points3D_2 = []
        self.__vis = None

    def read_model(self, joints_pos,joints_pos_2):
        self.points3D = joints_pos
        self.points3D_2 = joints_pos_2
        a = 1

    def add_points(self, min_track_len=3, remove_statistical_outlier=False):


        # 绘制open3d坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__vis.add_geometry(axis_pcd)

        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.points3D: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,0,1])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        for point in self.points3D_2: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,1,0])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))
        
        a = 1
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # a = 1
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    
    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # render_options = self.__vis.get_render_option()
        self.__vis.get_render_option().point_size = 5
        # render_options.point_size = 1.5  # 设置点的大小

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()




class ModelPose3:
    def __init__(self):
        self.points3D = []
        self.points3D_2 = []
        self.points3D_3 = []
        self.__vis = None

    def read_model(self, joints_pos,joints_pos_2,joints_pos_3):
        self.points3D = joints_pos
        self.points3D_2 = joints_pos_2
        self.points3D_3 = joints_pos_3
        a = 1

    def add_points(self, min_track_len=3, remove_statistical_outlier=False):


        # 绘制open3d坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__vis.add_geometry(axis_pcd)

        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.points3D: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,0,1])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        for point in self.points3D_2: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,1,0])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        for point in self.points3D_3: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([1,0,0])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        a = 1
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # a = 1
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    
    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # render_options = self.__vis.get_render_option()
        self.__vis.get_render_option().point_size = 5
        # render_options.point_size = 1.5  # 设置点的大小

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()

class Model:
    def __init__(self):
        self.Ks = []
        self.T_w2cs = []
        self.points3D = []
        self.__vis = None

    def read_model(self, joints_pos,T_w2cs,Ks):
        self.Ks = Ks
        self.T_w2cs = T_w2cs
        self.points3D = joints_pos
        a = 1

    def add_points(self, min_track_len=3, remove_statistical_outlier=False):


        # 绘制open3d坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__vis.add_geometry(axis_pcd)

        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.points3D: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,0,1])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        
        a = 1
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # a = 1
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1,imgH=3008,imgW=4112):
        frames = []
        for idx in range(len(self.T_w2cs)):
            img = self.T_w2cs[idx] #c-w
            # rotation
            R = img[:3, :3]

            # translation
            t = img[:3, -1]
            # import pdb; pdb.set_trace()
            # invert

            t = -R.T @ t
            R = R.T

            scale_center = 1
            # intrinsics 内参数
            K = self.Ks[idx]
            fx = K[0,0]
            fy = K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramed geometries that will be drawn
            cam_model = draw_camera(K, R, t, imgW, imgH, scale)
            frames.extend(cam_model)

        # add geometries to visualizer
        for i in frames:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # render_options = self.__vis.get_render_option()
        self.__vis.get_render_option().point_size = 5
        # render_options.point_size = 1.5  # 设置点的大小

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()
#2
class Model2:
    def __init__(self):
        self.Ks = []
        self.T_w2cs = []
        self.points3D = []
        self.points3D_2 = []
        self.__vis = None

    def read_model(self, joints_pos,joints_pos_2,T_w2cs,Ks):
        self.Ks = Ks
        self.T_w2cs = T_w2cs
        self.points3D = joints_pos
        self.points3D_2 = joints_pos_2
        a = 1

    def add_points(self, min_track_len=3, remove_statistical_outlier=False):


        # 绘制open3d坐标系
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.__vis.add_geometry(axis_pcd)

        pcd = open3d.geometry.PointCloud()

        xyz = []
        rgb = []
        for point in self.points3D: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,0,1])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        for point in self.points3D_2: #L 22 3
            xyz.append(point) #22 3
            #rgb.append(np.array([0,0,255]) / 255)
            rgb.append([0,1,0])
            #rgb.append(np.array([0,0,1])[None].repeat(point.shape[0],axis=0))

        a = 1
        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # a = 1
        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_cameras(self, scale=1):
        frames = []
        for idx in range(len(self.T_w2cs)):
            img = self.T_w2cs[idx] #c-w
            # rotation
            R = img[:3, :3]

            # translation
            t = img[:3, -1]
            # import pdb; pdb.set_trace()
            # invert

            t = -R.T @ t
            R = R.T

            scale_center = 1
            # intrinsics 内参数
            K = self.Ks[idx]
            fx = K[0,0]
            fy = K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramed geometries that will be drawn
            cam_model = draw_camera(K, R, t, 4112, 3008, scale_center)
            frames.extend(cam_model)

        # add geometries to visualizer
        for i in frames:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window()
        # render_options = self.__vis.get_render_option()
        self.__vis.get_render_option().point_size = 5
        # render_options.point_size = 1.5  # 设置点的大小

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera(K, R, t, w, h,
                scale=1, color=[1, 0, 0]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    return [plane, line_set]
