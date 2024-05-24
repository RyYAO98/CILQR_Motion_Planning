import numpy as np


def bicycle_model(cur_x, cur_u, dt, wheelbase):
    # see: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357
    # implementation of <2.3. If the desired point is at the center of gravity or cg.>
    beta = np.arctan(np.tan(cur_u[1]) / 2)
    next_x = np.array(
        [
            cur_x[0] + cur_x[2] * np.cos(beta + cur_x[3]) * dt,
            cur_x[1] + cur_x[2] * np.sin(beta + cur_x[3]) * dt,
            cur_x[2] + cur_u[0] * dt,
            cur_x[3] + 2 * cur_x[2] * np.sin(beta) * dt / wheelbase
        ]
    )

    return next_x


def get_N_steps_bicycle_model_derivatives(x, u, dt, wheelbase, N):
    N_velo = x[2, :-1]
    N_yaw = x[3, :-1]
    N_beta = np.arctan(np.tan(u[1] / 2))
    N_beta_over_stl = 0.5 * (1 + np.tan(u[1])**2) / (1 + 0.25 * np.tan(u[1])**2)

    # f(state, ctrl) over state of t_0 to t_N
    # df_dx.shape: (N, 4, 4)
    # For t_k, df_dx[k] is organized by:
    #   [[x_k+1     -> x_k, x_k+1     -> y_k, x_k+1     -> v_k, x_k+1     -> theta_k]
    #    [y_k+1     -> x_k, y_k+1     -> y_k, y_k+1     -> v_k, y_k+1     -> theta_k]
    #    [v_k+1     -> x_k, v_k+1     -> y_k, v_k+1     -> v_k, v_k+1     -> theta_k]
    #    [theta_k+1 -> x_k, theta_k+1 -> y_k, theta_k+1 -> v_k, theta_k+1 -> theta_k]]
    df_dx = np.tile(np.eye(4), (N, 1, 1))
    df_dx[:, 0, 2] = np.cos(N_beta + N_yaw) * dt
    df_dx[:, 0, 3] = N_velo * (-np.sin(N_beta + N_yaw)) * dt
    df_dx[:, 1, 2] = np.sin(N_beta + N_yaw) * dt
    df_dx[:, 1, 3] = N_velo * np.cos(N_beta + N_yaw) * dt
    df_dx[:, 3, 2] = 2 * np.sin(N_beta) * dt / wheelbase

    # f(state, ctrl) over ctrl of t_0 to t_N
    # df_du.shape: (N, 4, 2)
    # For t_k, df_du[k] is organized by:
    #   [[x_k+1     -> a_k, x_k+1     -> delta_k]
    #    [y_k+1     -> a_k, y_k+1     -> delta_k]
    #    [v_k+1     -> a_k, v_k+1     -> delta_k]
    #    [theta_k+1 -> a_k, theta_k+1 -> delta_k]]
    df_du = np.zeros((N, 4, 2))
    df_du[:, 2, 0] = np.ones(N) * dt
    df_du[:, 0, 1] = N_velo * (-np.sin(N_beta + N_yaw)) * dt * N_beta_over_stl
    df_du[:, 1, 1] = N_velo * np.cos(N_beta + N_yaw) * dt * N_beta_over_stl
    df_du[:, 3, 1] = (2 * N_velo * dt / wheelbase) * np.cos(N_beta) * N_beta_over_stl

    # Put time horizon at the innermost dimension to satisfy the output formats
    return df_dx.transpose(1, 2, 0), df_du.transpose(1, 2, 0)
