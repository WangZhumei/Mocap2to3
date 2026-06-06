import numpy as np
import torch
import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .wis3d_utils import get_const_colors, color_schemes
from .skeleton_utils import SMPL_SKELETON, NBA_SKELETON, GYM_SKELETON, COLOR_NAMES
import hmr4d.utils.matrix as matrix


def get_kinematic_chain(skeleton_type="nba"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    else:
        raise NotImplementedError
    kinematic_chain = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["kinematic_chain"]
    ]
    return kinematic_chain


def get_skeleton_lines(skeleton_connections, ax):
    connections = []
    color_names = COLOR_NAMES
    for i, skel_con in enumerate(skeleton_connections):
        for _ in range(len(skel_con) - 1):
            c = np.array(color_schemes[color_names[i]][1]) / 255.0
            (line,) = ax.plot([], [], "k-", color=c)
            connections.append(line)
    return connections


def get_joints_color(pos, skeleton_type="nba"):
    if skeleton_type == "smpl":
        skeleton = SMPL_SKELETON
    elif skeleton_type == "nba":
        skeleton = NBA_SKELETON
    elif skeleton_type == "gym":
        skeleton = GYM_SKELETON
    else:
        raise NotImplementedError
    color_names = COLOR_NAMES
    joints_category = [
        [skeleton["joints"].index(skeleton_name) for skeleton_name in sub_skeleton_names]
        for sub_skeleton_names in skeleton["joints_category"]
    ]
    joints_color = []
    J = pos.shape[-2]
    for i in range(J):
        for j, joints_ in enumerate(joints_category):
            if i in joints_:
                joints_color.append(color_schemes[color_names[j]][1])
                break
    joints_color = np.array(joints_color) / 255.0
    return joints_color


class plt_skeleton_animation:
    def __init__(self, pos, text="", skeleton_type="nba"):
        """_summary_

        Args:
            NOTE: sometimes J may be >22 as we use virtual next frame root for global motions
            pos (tensor): (progress, T, J, 3) joints positions (x, y) and confidence (optional)
            text (str)
        """
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if len(pos.shape) == 3:
            pos = pos[None]
        pos = copy.deepcopy(pos)
        pos[..., 1] *= -1
        self.pos = pos
        self.text = text
        self.skeleton_type = skeleton_type
        fig, ax = plt.subplots()
        fig.suptitle(text)
        self.fig = fig
        self.ax = ax
        joints_color = get_joints_color(pos, skeleton_type)
        points = []
        for i in range(pos.shape[-2]):
            points.append(plt.plot([], [], "o", color=joints_color[i])[0])
        self.points = points

        self.skeleton_connections = get_kinematic_chain(skeleton_type)
        self.connections = get_skeleton_lines(self.skeleton_connections, ax)

        self.paused = False
        self.current_frame = 0
        self.current_progress = pos.shape[0] - 1

        self.animation = FuncAnimation(
            fig,
            self.update,
            frames=pos.shape[-3],
            init_func=self.plt_init,
            blit=True,
            interval=33,
        )
        fig.canvas.mpl_connect("key_press_event", self.on_key)

        plt.show()

    def plt_init(self):
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            x_min, y_min, _ = self.pos.min(axis=(0, 1, 2))
            x_max, y_max, _ = self.pos.max(axis=(0, 1, 2))
        else:
            # x, y
            x_min, y_min = self.pos.min(axis=(0, 1, 2))
            x_max, y_max = self.pos.max(axis=(0, 1, 2))
        x_lim_min = x_min * 0.8 if x_min > 0 else x_min * 1.2
        x_lim_max = x_max * 0.8 if x_max < 0 else x_max * 1.2
        y_lim_min = y_min * 0.8 if y_min > 0 else y_min * 1.2
        y_lim_max = y_max * 0.8 if y_max < 0 else y_max * 1.2
        self.ax.set_xlim(x_lim_min, x_lim_max)
        self.ax.set_ylim(y_lim_min, y_lim_max)
        return *self.points, *self.connections

    def update(self, frame):
        p = self.current_progress
        x, y = self.pos[p, frame, :, 0], self.pos[p, frame, :, 1]
        if self.pos.shape[-1] == 3:
            # x, y, confidence
            confidence = self.pos[p, frame, :, 2]
        else:
            # x, y
            confidence = np.ones_like(self.pos[p, frame, :, 0])
        for i in range(self.pos.shape[-2]):
            self.points[i].set_data(x[i], y[i])
            confidence[i] = min(confidence[i], 1.0)
            confidence[i] = max(confidence[i], 0.0)
            self.points[i].set_alpha(confidence[i])
            self.points[i].set_markersize(20 * confidence[i])
            # self.points[i].set_markersize(1)
        i = 0
        for skel_con in self.skeleton_connections:
            for j in range(len(skel_con) - 1):
                a = skel_con[j]
                b = skel_con[j + 1]
                self.connections[i].set_data([x[a], x[b]], [y[a], y[b]])
                i += 1
        return *self.points, *self.connections

    def on_key(self, event):
        if event.key == "p":
            if self.paused:
                self.animation.event_source.start()
            else:
                self.animation.event_source.stop()
            self.paused = ~self.paused
        if event.key == "u":
            self.current_progress = min(self.current_progress + 1, self.pos.shape[0] - 1)
        if event.key == "y":
            self.current_progress = max(self.current_progress - 1, 0)
