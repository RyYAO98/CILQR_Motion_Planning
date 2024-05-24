import numpy as np
from system_dynamics.vehicle_model import bicycle_model


# 初始条件和参考轨迹
def get_ref_waypoints(config):
    ref_traj = config['reference_trajectory']
    longit_ref = np.linspace(ref_traj['longit_ref'][0], ref_traj['longit_ref'][1], ref_traj['num_wpts'])
    lateral_ref = np.linspace(ref_traj['lateral_ref'][0], ref_traj['lateral_ref'][1], ref_traj['num_wpts'])

    x_ref = np.vstack((longit_ref, lateral_ref))  # 参考轨迹点
    return x_ref


def const_velo_prediction(x0, N, dt, wheelbase):
    cur_u = np.zeros(2)

    predicted_states = [x0]
    cur_x = x0
    for i in range(N):
        next_x = bicycle_model(cur_x, cur_u, dt, wheelbase)
        cur_x = next_x
        predicted_states.append(next_x)

    predicted_states = np.vstack(predicted_states).transpose()

    return predicted_states


def get_ref_exact_points(pos, ref_waypoints):
    ref_waypoints_reshaped = ref_waypoints.transpose()[:, :, np.newaxis]
    distances = np.sum((pos - ref_waypoints_reshaped) ** 2, axis=1)
    arg_min_dist_indices = np.argmin(distances, axis=0)
    ref_exact_points = ref_waypoints[:, arg_min_dist_indices]

    return ref_exact_points


def ellipsoid_safety_margin(pnt, elp_center, theta, a, b):
    diff = pnt - elp_center
    rotation_matrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pnt_std = diff @ rotation_matrx  # rotate by (-theta)

    result = 1 - ((pnt_std[0] ** 2) / (a ** 2) + (pnt_std[1] ** 2) / (b ** 2))

    return result


def ellipsoid_safety_margin_derivatives(pnt, elp_center, theta, a, b):
    diff = pnt - elp_center
    rotation_matrx = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pnt_std = diff @ rotation_matrx  # rotate by (-theta)

    # (1) constraint over standard point vec.:
    #       [c -> x_std, c -> y_std]
    res_over_pnt_std = np.array([-2 * pnt_std[0] / (a ** 2), -2 * pnt_std[1] / (b ** 2)])

    # (2) standard point vec. over difference vec.:
    #       [[x_std -> x_diff, x_std -> y_diff]
    #        [y_std -> x_diff, y_std -> y_diff]]
    pnt_std_over_diff = rotation_matrx.transpose()

    # (3) difference vec. over original point vec.:
    #       [[x_diff -> x, x_diff -> y]
    #        [y_diff -> x, y_diff -> y]]
    diff_over_pnt = np.eye(2)

    # chain (1)(2)(3) together:
    #       [c -> x, c -> y]
    res_over_pnt = res_over_pnt_std @ pnt_std_over_diff @ diff_over_pnt

    return res_over_pnt


def get_vehicle_front_and_rear_centers(pos, yaw, wheelbase):
    half_whba_vec = 0.5 * wheelbase * np.array([np.cos(yaw), np.sin(yaw)])
    front_pnt = pos + half_whba_vec
    rear_pnt = pos - half_whba_vec

    return front_pnt, rear_pnt


def get_vehicle_front_and_rear_center_derivatives(yaw, wheelbase):
    half_whba = 0.5 * wheelbase

    # front point over (center) state:
    #           [[x_fr -> x_c, x_fr -> y_c, x_fr -> v, x_fr -> yaw]
    #            [y_fr -> x_c, y_fr -> y_c, y_fr -> v, y_fr -> yaw]]
    front_pnt_over_state = np.array([
        [1., 0., 0., half_whba * (-np.sin(yaw))],
        [0., 1., 0., half_whba *   np.cos(yaw) ]
    ])

    # rear point over (center) state:
    #            <similarly...>
    rear_point_over_state = np.array([
        [1., 0., 0., -half_whba * (-np.sin(yaw))],
        [0., 1., 0., -half_whba *   np.cos(yaw) ]
    ])

    return front_pnt_over_state, rear_point_over_state


def get_ellipsoid_obstacle_scales(ego_pnt_radius, obs_width, obs_length, d_safe):
    a = 0.5 * obs_length + d_safe + ego_pnt_radius
    b = 0.5 * obs_width + d_safe + ego_pnt_radius

    return a, b
