import numpy as np
import time
import yaml
import utils
from solver.CILQR import CILQR
from animation import vis


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config


def ego_state_loader(config):
    return np.array(config['initial_condition']['state'])


def ref_waypoints_loader(config):
    return utils.get_ref_waypoints(config)


def ref_velo_loader(config):
    return np.array(config['reference_trajectory']['velo_ref'])


def obstacle_attr_loader(config):
    obstacle_params = config['obstacle']
    obstacle_attrs = np.array([obstacle_params['width'], obstacle_params['length'], obstacle_params['d_safe']]).transpose()

    return obstacle_attrs # shape: (num_obs, 3)


def obstacle_pred_loader(config):
    mpc_params = config['mpc']
    N = mpc_params['N']
    dt = mpc_params['dt']

    obstacle_params = config['obstacle']
    obstacle_states = obstacle_params['init_state']
    obstacle_whbas = obstacle_params['wheelbase']

    # improvement is needed regarding parallel implementation
    num_obs = len(obstacle_states)
    obstacle_preds = []
    for i in range(num_obs):
        cur_obstacle_state = obstacle_states[i]
        cur_obstacle_whba = obstacle_whbas[i]
        cur_obstacle_pred = utils.const_velo_prediction(cur_obstacle_state, N, dt, cur_obstacle_whba)
        obstacle_preds.append(cur_obstacle_pred)

    obstacle_preds = np.array(obstacle_preds)

    return obstacle_preds # shape: (num_obs, 4, N+1)


def main():
    config = load_config('config.yaml')

    planner = CILQR(config)

    ego_state = ego_state_loader(config)
    ref_waypoints = ref_waypoints_loader(config)
    ref_velo = ref_velo_loader(config)
    obstacle_attrs = obstacle_attr_loader(config)
    obstacle_preds = obstacle_pred_loader(config)

    solver_start_t = time.process_time()
    opti_u, opti_x = planner.solve(ego_state,
                                   ref_waypoints,
                                   ref_velo,
                                   obstacle_attrs,
                                   obstacle_preds
                                   )
    print('----CILQR Solution Time: {} seconds----'.format(time.process_time() - solver_start_t))

    vis(config, ref_waypoints, obstacle_attrs, obstacle_preds, opti_x)


main()
