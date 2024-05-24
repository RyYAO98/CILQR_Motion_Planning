import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from utils import get_vehicle_front_and_rear_centers, get_ellipsoid_obstacle_scales
import numpy as np
import time


def create_vehicle_patches(ax, x, y, yaw, length, width, wheelbase):
    rect = patches.Rectangle((x - length / 2, y - width / 2), length, width, edgecolor='black', facecolor='none')
    transform = Affine2D().rotate_deg_around(x, y, np.degrees(yaw)) + ax.transData
    rect.set_transform(transform)

    front_center, rear_center = get_vehicle_front_and_rear_centers([x, y], yaw, wheelbase)
    front_circle = patches.Circle(front_center, 0.5 * width, edgecolor='r', facecolor='none')
    rear_circle = patches.Circle(rear_center, 0.5 * width, edgecolor='r', facecolor='none')

    return rect, front_circle, rear_circle


def create_obstacle_patches(ax, x, y, yaw, length, width, d_safe):
    rect = patches.Rectangle((x - length / 2, y - width / 2), length, width, edgecolor='black', facecolor='none')
    transform = Affine2D().rotate_deg_around(x, y, np.degrees(yaw)) + ax.transData
    rect.set_transform(transform)

    a, b = get_ellipsoid_obstacle_scales(0.0, width, length, d_safe)
    ellipse = patches.Ellipse((x, y), width=2 * a, height=2 * b, angle=np.degrees(yaw), edgecolor='b', facecolor='none')

    return rect, ellipse


def init_patches(ax, opti_x, obs_pred, obstacle_attr, ego_length, ego_width, ego_whba):
    ego_patches = []
    obs_patches = []

    ego_rect, front_circle, rear_circle = create_vehicle_patches(ax, opti_x[0, 0], opti_x[1, 0], opti_x[3, 0],
                                                                 ego_length, ego_width, ego_whba)
    ego_patches.extend([ego_rect, front_circle, rear_circle])
    ax.add_patch(ego_rect)
    ax.add_patch(front_circle)
    ax.add_patch(rear_circle)

    obs_rect, ellipse = create_obstacle_patches(ax, obs_pred[0, 0], obs_pred[1, 0], obs_pred[3, 0], obstacle_attr[1],
                                                obstacle_attr[0], obstacle_attr[2])
    obs_patches.extend([obs_rect, ellipse])
    ax.add_patch(obs_rect)
    ax.add_patch(ellipse)

    return ego_patches, obs_patches


def update_patches(frame, ax, opti_x, obs_pred, obstacle_attr, ego_length, ego_width, ego_whba, ego_patches,
                   obs_patches):
    ego_rect, front_circle, rear_circle = create_vehicle_patches(ax, opti_x[0, frame], opti_x[1, frame],
                                                                 opti_x[3, frame], ego_length, ego_width, ego_whba)
    obs_rect, ellipse = create_obstacle_patches(ax, obs_pred[0, frame], obs_pred[1, frame], obs_pred[3, frame],
                                                obstacle_attr[1], obstacle_attr[0], obstacle_attr[2])

    ego_patches.extend([ego_rect, front_circle, rear_circle])
    obs_patches.extend([obs_rect, ellipse])

    ax.add_patch(ego_rect)
    ax.add_patch(front_circle)
    ax.add_patch(rear_circle)
    ax.add_patch(obs_rect)
    ax.add_patch(ellipse)

    return ego_patches + obs_patches


def vis(cfg, ref_waypoints, obstacle_attr, obs_pred, opti_x, save_path=None, save_format='gif'):
    mpc_params = cfg['mpc']
    N = mpc_params['N']

    ego_veh_params = cfg['vehicle']
    ego_length, ego_width, ego_whba = ego_veh_params['length'], ego_veh_params['width'], ego_veh_params['wheelbase']

    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlim((ref_waypoints[0, 0] - 5, ref_waypoints[0, -1]))
    ax.plot(ref_waypoints[0, :], ref_waypoints[1, :], c='lime', label='reference path', zorder=0)

    ego_patches, obs_patches = init_patches(ax, opti_x, obs_pred, obstacle_attr, ego_length, ego_width, ego_whba)

    def update(frame):
        return update_patches(frame, ax, opti_x, obs_pred, obstacle_attr, ego_length, ego_width, ego_whba, ego_patches,
                              obs_patches)

    ani = FuncAnimation(fig, update, frames=N + 1, blit=False, repeat=False)

    plt.legend()

    if save_path:
        fps = int(1 / cfg['mpc']['dt'])

        if save_format == 'mp4':
            writer = FFMpegWriter(fps=fps)
            ani.save(save_path, writer=writer)
        elif save_format == 'gif':
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
        else:
            raise ValueError("Unsupported save format. Use 'mp4' or 'gif'.")

    else:
        plt.show()
