# CILQR for Autonomous Driving Motion Planning
This mini project implements a modified version of the Constrained iLQR for autonomous driving motion planning with soft constraints [1].

# Results of Example Settings
## Lane Change
    initial_condition:
      state: 
                - [0., 3., 5., 0.]
    obstacle:
      init_state: 
                - [2.5, 0., 3., 0.]
      action_gt:
                - [0., 0.]
      wheelbase:
                - 2.7
      width:
                - 1.5
      length:
                - 3.2
      d_safe:
                - 1.0
![LaneChange](./sim_record/Lane_Change.gif)

## Overtaking
    initial_condition:
      state: 
                - [0., 0., 5., 0.]
    
    obstacle:
      init_state: 
                - [5., 1.2, 3., 0.]
      action_gt:
                - [0., 0.]
      wheelbase:
                - 2.7
      width:
                - 1.5
      length:
                - 3.2
      d_safe:
                - 1.0
![Overtaking](./sim_record/Overtaking.gif)

## Car Following
    initial_condition:
      state: 
                - [0., 0., 4., 0.]
    
    obstacle:
      init_state: 
                - [5., 0., 3., 0.]
      action_gt:
                - [0., 0.]
      wheelbase:
                - 2.7
      width:
                - 1.5
      length:
                - 3.2
      d_safe:
                - 1.0
![CarFollowing](./sim_record/Car_Following.gif)

## Multi-obstacle Avoidance
    initial_condition:
      state: [0., 0., 5.0, 0.]
    
    obstacle:
      init_state:
                - [5., -1.2, 3.0, 0.]
                - [25.0, 2.4, 0., 0.]
      action_gt:
                - [0., 0.]
                - [0., 0.]
      wheelbase:
                - 2.7
                - 2.5
      width:
                - 1.5
                - 1.5
      length:
                - 3.2
                - 3.0
      d_safe:
                - 1.0
                - 1.0
![MultiObstacle](./sim_record/Multi_Obstacles.gif)

# References
[1] Chen, J., Zhan, W., & Tomizuka, M. (2017, October). Constrained iterative lqr for on-road autonomous driving motion planning. In 2017 IEEE 20th International conference on intelligent transportation systems (ITSC) (pp. 1-7). IEEE.


