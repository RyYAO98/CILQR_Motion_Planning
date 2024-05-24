import numpy as np
import sys
from system_dynamics.vehicle_model import bicycle_model, get_N_steps_bicycle_model_derivatives
from utils import const_velo_prediction, get_ref_exact_points
from solver.solver_utils import (
    exp_barrier,
    exp_barrier_derivative_and_Hessian,
    get_bound_constr,
    get_obstacle_avoidance_constr,
    get_obstacle_avoidance_constr_derivatives,
    get_bounded_ctrl
)


class CILQR:

    def __init__(self, cfg):
        # planning-related settings
        planner_params = cfg['mpc']
        self.N = planner_params['N']
        self.dt = planner_params['dt']
        self.nx = planner_params['nx']
        self.nu = planner_params['nu']
        self.state_weight = np.array(
            [[planner_params['w_pos'], 0., 0., 0.],
             [0., planner_params['w_pos'], 0., 0.],
             [0., 0., planner_params['w_vel'], 0.],
             [0., 0., 0., 0.]]
        )
        self.ctrl_weight = np.array(
            [[planner_params['w_acc'], 0.0],
             [0.0, planner_params['w_stl']]
             ]
        )
        self.exp_q1 = planner_params['exp_q1']
        self.exp_q2 = planner_params['exp_q2']

        # iteration-related settings
        iteration_params = cfg['iteration']
        self.max_iter = iteration_params['max_iter']
        self.init_lamb = iteration_params['init_lamb']
        self.lamb_decay = iteration_params['lamb_decay']
        self.lamb_amplify = iteration_params['lamb_amplify']
        self.max_lamb = iteration_params['max_lamb']
        self.alpha_options = iteration_params['alpha_options']
        self.tol = iteration_params['tol']

        # ego vehicle-related settings
        ego_veh_params = cfg['vehicle']
        self.wheelbase = ego_veh_params['wheelbase']
        self.width = ego_veh_params['width']
        self.length = ego_veh_params['length']
        self.velo_max = ego_veh_params['velo_max']
        self.velo_min = ego_veh_params['velo_min']
        self.yaw_lim = ego_veh_params['yaw_lim']
        self.acc_max = ego_veh_params['a_max']
        self.acc_min = ego_veh_params['a_min']
        self.stl_lim = ego_veh_params['stl_lim']

        # vector representations of ctrl and state bounds
        # self.ctrl_lo_bound = np.array([self.acc_min, -self.stl_lim])
        # self.ctrl_up_bound = np.array([self.acc_max, self.stl_lim])
        # self.state_lo_bound = np.array([-np.inf, -np.inf, self.velo_min, -np.inf])
        # self.state_up_bound = np.array([np.inf, np.inf, self.velo_max, np.inf])

    def get_nominal_traj(self, x0):
        # zero ctrl initialization
        nomi_u = np.zeros((self.nu, self.N))
        nomi_x = const_velo_prediction(x0, self.N, self.dt, self.wheelbase)

        return nomi_u, nomi_x

    def backward_pass(self, u, x, lamb, ref_waypoints, ref_velo, obs_attr, obs_pred):
        l_u, l_uu, l_x, l_xx, l_ux = self.get_total_cost_derivatives_and_Hessians(u, x, ref_waypoints, ref_velo,
                                                                                  obs_attr, obs_pred)
        df_dx, df_du = get_N_steps_bicycle_model_derivatives(x, u, self.dt, self.wheelbase, self.N)

        delt_V = 0
        V_x = l_x[:, -1]
        V_xx = l_xx[:, :, -1]

        k = np.zeros((self.nu, self.N))
        K = np.zeros((self.nu, self.nx, self.N))

        regu_I = lamb * np.eye(V_xx.shape[0])

        # Run a backwards pass from N-1 control step
        for i in range(self.N - 1, -1, -1):
            # This part of the implementation references:
            # https://github.com/Bharath2/iLQR/blob/main/ilqr/controller.py

            # Q_terms
            Q_x = l_x[:, i] + df_dx[:, :, i].T @ V_x
            Q_u = l_u[:, i] + df_du[:, :, i].T @ V_x
            Q_xx = l_xx[:, :, i] + df_dx[:, :, i].T @ V_xx @ df_dx[:, :, i]
            Q_uu = l_uu[:, :, i] + df_du[:, :, i].T @ V_xx @ df_du[:, :, i]
            Q_ux = l_ux[:, :, i] + df_du[:, :, i].T @ V_xx @ df_dx[:, :, i]

            # gains
            df_du_regu = df_du[:, :, i].T @ regu_I
            Q_ux_regu = Q_ux + df_du_regu @ df_dx[:, :, i]
            Q_uu_regu = Q_uu + df_du_regu @ df_du[:, :, i]
            Q_uu_inv = np.linalg.inv(Q_uu_regu)

            k[:, i] = -Q_uu_inv @ Q_u
            K[:, :, i] = -Q_uu_inv @ Q_ux_regu

            # Update value function for next time step
            V_x = Q_x + K[:, :, i].T @ Q_uu @ k[:, i] + K[:, :, i].T @ Q_u + Q_ux.T @ k[:, i]
            V_xx = Q_xx + K[:, :, i].T @ Q_uu @ K[:, :, i] + K[:, :, i].T @ Q_ux + Q_ux.T @ K[:, :, i]

            # expected cost reduction
            delt_V += 0.5 * k[:, i].T @ Q_uu @ k[:, i] + k[:, i].T @ Q_u

        return k, K, delt_V

    def forward_pass(self, u, x, k, K, alpha):
        new_u = np.zeros((self.nu, self.N))
        new_x = np.zeros((self.nx, self.N + 1))
        new_x[:, 0] = x[:, 0]

        for i in range(self.N):
            new_u_i = u[:, i] + alpha * k[:, i] + K[:, :, i] @ (new_x[:, i] - x[:, i])
            new_u[:, i] = get_bounded_ctrl(new_u_i, new_x[:, i], self.acc_max, self.acc_min, self.stl_lim,
                                           self.velo_min, self.velo_max)
            new_x[:, i + 1] = bicycle_model(new_x[:, i], new_u[:, i], self.dt, self.wheelbase)

        return new_u, new_x

    def get_total_cost(self, u, x, ref_waypoints, ref_velo, obs_attr, obs_pred):
        # part 1: costs included in the prime objective
        ref_exact_points = get_ref_exact_points(x[:2], ref_waypoints)
        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.N + 1, ref_velo),
            np.zeros(self.N + 1)
        ])

        states_devt = np.sum(((x - ref_states).T @ self.state_weight) * (x - ref_states).T)
        ctrl_energy = np.sum((u.T @ self.ctrl_weight) * u.T)

        J_prime = states_devt + ctrl_energy

        # part 2: costs of the barrier function terms
        J_barrier = 0.
        for k in range(1, self.N + 1):
            u_k_minus_1 = u[:, k - 1]
            x_k = x[:, k]  # there is no need to count for the current state cost
            obs_pred_k = obs_pred[:, k]

            # acceleration constraints
            acc_up_constr = get_bound_constr(u_k_minus_1[0], self.acc_max, bound_type='upper')
            acc_lo_constr = get_bound_constr(u_k_minus_1[0], self.acc_min, bound_type='lower')

            # steering angle constraints
            stl_up_constr = get_bound_constr(u_k_minus_1[1], self.stl_lim, bound_type='upper')
            stl_lo_constr = get_bound_constr(u_k_minus_1[1], - self.stl_lim, bound_type='lower')

            # velocity constraints
            velo_up_constr = get_bound_constr(x_k[2], self.velo_max, bound_type='upper')
            velo_lo_constr = get_bound_constr(x_k[2], self.velo_min, bound_type='lower')

            # obstacle avoidance constraints
            obs_avoid_front_constr, obs_avoid_rear_constr = get_obstacle_avoidance_constr(
                x_k, obs_pred_k, self.wheelbase, self.width, obs_attr
            )

            J_barrier_k = exp_barrier(acc_up_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(acc_lo_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(stl_up_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(stl_lo_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(velo_up_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(velo_lo_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(obs_avoid_front_constr, self.exp_q1, self.exp_q2) \
                          + exp_barrier(obs_avoid_rear_constr, self.exp_q1, self.exp_q2)

            J_barrier += J_barrier_k

        # Get the total cost
        J_tot = J_prime + J_barrier

        return J_tot

    def get_total_cost_derivatives_and_Hessians(self, u, x, ref_waypoints, ref_velo, obs_attr, obs_pred):
        ref_exact_points = get_ref_exact_points(x[:2], ref_waypoints)
        ref_states = np.vstack([
            ref_exact_points,
            np.full(self.N + 1, ref_velo),
            np.zeros(self.N + 1)
        ])

        # part 1: cost derivatives due to the prime objective
        l_u_prime = 2 * (u.T @ self.ctrl_weight).T
        l_uu_prime = np.repeat((2 * self.ctrl_weight)[:, :, np.newaxis], self.N, axis=2)
        l_x_prime = 2 * ((x - ref_states).T @ self.state_weight).T
        l_xx_prime = np.repeat((2 * self.state_weight)[:, :, np.newaxis], self.N + 1, axis=2)

        # part 2: cost derivatives due to the barrier terms
        l_u_barrier = np.zeros((self.nu, self.N))
        l_uu_barrier = np.zeros((self.nu, self.nu, self.N))
        l_x_barrier = np.zeros((self.nx, self.N + 1))
        l_xx_barrier = np.zeros((self.nx, self.nx, self.N + 1))

        for k in range(self.N + 1):
            # ---- Ctrl: only N steps ----
            if k < self.N:
                u_k = u[:, k]

                # acceleration constraints derivatives and Hessians
                acc_up_constr = get_bound_constr(u_k[0], self.acc_max, bound_type='upper')
                acc_up_constr_over_u = np.array([1., 0.])
                acc_up_barrier_over_u, acc_up_barrier_over_uu = exp_barrier_derivative_and_Hessian(
                    acc_up_constr, acc_up_constr_over_u, self.exp_q1, self.exp_q2
                )

                acc_lo_constr = get_bound_constr(u_k[0], self.acc_min, bound_type='lower')
                acc_lo_constr_over_u = np.array([-1., 0.])
                acc_lo_barrier_over_u, acc_lo_barrier_over_uu = exp_barrier_derivative_and_Hessian(
                    acc_lo_constr, acc_lo_constr_over_u, self.exp_q1, self.exp_q2
                )

                # steering angle constraints derivatives and Hessians
                stl_up_constr = get_bound_constr(u_k[1], self.stl_lim, bound_type='upper')
                stl_up_constr_over_u = np.array([0., 1.])
                stl_up_barrier_over_u, stl_up_barrier_over_uu = exp_barrier_derivative_and_Hessian(
                    stl_up_constr, stl_up_constr_over_u, self.exp_q1, self.exp_q2
                )

                stl_lo_constr = get_bound_constr(u_k[1], -self.stl_lim, bound_type='lower')
                stl_lo_constr_over_u = np.array([0., -1.])
                stl_lo_barrier_over_u, stl_lo_barrier_over_uu = exp_barrier_derivative_and_Hessian(
                    stl_lo_constr, stl_lo_constr_over_u, self.exp_q1, self.exp_q2
                )

                # fill the ctrl-related spaces
                l_u_barrier[:, k] = acc_up_barrier_over_u + acc_lo_barrier_over_u \
                                  + stl_up_barrier_over_u + stl_lo_barrier_over_u

                l_uu_barrier[:, :, k] = acc_up_barrier_over_uu + acc_lo_barrier_over_uu \
                                      + stl_up_barrier_over_uu + stl_lo_barrier_over_uu

            # ---- State: (N + 1) steps ----
            x_k = x[:, k]
            obs_pred_k = obs_pred[:, k]

            # velocity constraints derivatives and Hessians
            velo_up_constr = get_bound_constr(x_k[2], self.velo_max, bound_type='upper')
            velo_up_constr_over_x = np.array([0., 0., 1., 0.])
            velo_up_barrier_over_x, velo_up_barrier_over_xx = exp_barrier_derivative_and_Hessian(
                velo_up_constr, velo_up_constr_over_x, self.exp_q1, self.exp_q2
            )

            velo_lo_constr = get_bound_constr(x_k[2], self.velo_min, bound_type='lower')
            velo_lo_constr_over_x = np.array([0., 0., -1., 0.])
            velo_lo_barrier_over_x, velo_lo_barrier_over_xx = exp_barrier_derivative_and_Hessian(
                velo_lo_constr, velo_lo_constr_over_x, self.exp_q1, self.exp_q2
            )

            # obstacle avoidance constraints derivatives and Hessians
            obs_avoid_front_constr, obs_avoid_rear_constr = get_obstacle_avoidance_constr(
                x_k, obs_pred_k, self.wheelbase, self.width, obs_attr
            )
            obs_avoid_front_constr_over_x, obs_avoid_rear_constr_over_x = get_obstacle_avoidance_constr_derivatives(
                x_k, obs_pred_k, self.wheelbase, self.width, obs_attr
            )
            obs_avoid_front_barrier_over_x, obs_avoid_front_barrier_over_xx = exp_barrier_derivative_and_Hessian(
                obs_avoid_front_constr, obs_avoid_front_constr_over_x, self.exp_q1, self.exp_q2
            )
            obs_avoid_rear_barrier_over_x, obs_avoid_rear_barrier_over_xx = exp_barrier_derivative_and_Hessian(
                obs_avoid_rear_constr, obs_avoid_rear_constr_over_x, self.exp_q1, self.exp_q2
            )

            # fill the state-related spaces
            l_x_barrier[:, k] = velo_up_barrier_over_x + velo_lo_barrier_over_x \
                              + obs_avoid_front_barrier_over_x + obs_avoid_rear_barrier_over_x

            l_xx_barrier[:, :, k] = velo_up_barrier_over_xx + velo_lo_barrier_over_xx \
                                  + obs_avoid_front_barrier_over_xx + obs_avoid_rear_barrier_over_xx

        # Get the results by combining both components
        l_u = l_u_prime + l_u_barrier
        l_uu = l_uu_prime + l_uu_barrier
        l_x = l_x_prime + l_x_barrier
        l_xx = l_xx_prime + l_xx_barrier

        l_ux = np.zeros((self.nu, self.nx, self.N))

        return l_u, l_uu, l_x, l_xx, l_ux

    def iter_step(self, u, x, J, lamb, ref_waypoints, ref_velo, obs_attr, obs_pred):
        k, K, expc_redu = self.backward_pass(u, x, lamb, ref_waypoints, ref_velo, obs_attr, obs_pred)

        iter_effective_flag = False
        new_u, new_x, new_J = np.zeros((self.nu, self.N)), np.zeros((self.nx, self.N + 1)), sys.float_info.max

        for alpha in self.alpha_options:
            new_u, new_x = self.forward_pass(u, x, k, K, alpha)
            new_J = self.get_total_cost(new_u, new_x, ref_waypoints, ref_velo, obs_attr, obs_pred)

            if new_J < J:
                iter_effective_flag = True
                break

        return new_u, new_x, new_J, iter_effective_flag

    def solve(self, x0, ref_waypoints, ref_velo, obs_attr, obs_pred):
        nomi_u, nomi_x = self.get_nominal_traj(x0)
        J = self.get_total_cost(nomi_u, nomi_x, ref_waypoints, ref_velo, obs_attr, obs_pred)
        u, x = nomi_u, nomi_x

        lamb = self.init_lamb

        for itr in range(self.max_iter):
            new_u, new_x, new_J, iter_effective_flag = self.iter_step(
                u, x, J, lamb, ref_waypoints, ref_velo, obs_attr, obs_pred
            )

            if iter_effective_flag:
                x = new_x
                u = new_u
                J_temp = J
                J = new_J

                if abs(J - J_temp) < self.tol:
                    print('Tolerance condition satisfied.')
                    break

                lamb *= self.lamb_decay

            else:
                lamb *= self.lamb_amplify

                if lamb > self.max_lamb:
                    print('Regularization parameter reached the maximum.')
                    break

        return u, x
