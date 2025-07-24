
## Arguments
The gym.make command supports multiple arguments:

| Argument                    | Usage                                                                            | Values                                                                             |
| --------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| seed                        | The number to seed the random number generator with                              | None or int                                                                        |
| max_episode_steps           | The maximum number of steps per episode                                          | None or int                                                                        |
| use_one_hot                 | Whether the one hot wrapper should be use to add the task ID to the observation  | True or False                                                                      |
| num_tasks                   | The number of parametric variations to sample (default:50)                       | int                                                                                |
| terminate_on_success        | Whether to terminate the episode during training when the success signal is seen | True or False                                                                      |
| vector_strategy             | What kind of vector strategy the environments should be wrapped in               | 'sync' or 'async'                                                                  |
| task_select                 | How parametric variations should be selected                                     | "random" or "pseudorandom"                                                         |
| reward_function_version     | Use the original reward functions from Meta-World or the updated ones            | "v1" or "v2"                                                                       |
| reward_normalization_method | Apply a reward normalization wrapper                                             | None or 'gymnasium' or 'exponential'                                               |
| render_mode                 | The render mode of each environment                                              | None or 'human' or 'rgb_array' or 'depth_array'                                    |
| camera_name                 | The Mujoco name of the camera that should be used to render                      | 'corner' or 'topview' or 'behindGripper' or 'gripperPOV' or 'corner2' or 'corner3' |
| camera_id                   | The Mujoco ID of the camera that should be used to render                        | int                                                                                |

# State Space

Like the [action space](action_space), the state space among the tasks is maintains the same structure such that a single policy/model can be shared between tasks.
Meta-World contains tasks that either manipulate a single object with a potentially variable goal position (e.g., reach, push, pick place) or to manipulate two objects with a fixed goal position (e.g., hammer, soccer, shelf place).
To account for such variability, large parts of the observation space are kept as placeholders, e.g., for the second object, if only one object is available.

The observation array consists of the end-effector's 3D Cartesian position and the composition of a single object with its goal coordinates or the positions of the first and second object.
This always results in a 9D state vector.

| Indices | Description                                                                   |
| ------- | ----------------------------------------------------------------------------- |
| 0:2     | the XYZ coordinates of the end-effector                                       |
| 3       | a scalar value that represents how open/closed the gripper is                 |
| 4:6     | the XYZ coordinates of the first object                                       |
| 7:10    | the quaternion describing the spatial orientations and rotations of object #1 |
| 11:13   | the XYZ coordinates of the second object                                      |
| 14:17   | the quaternion describing the spatial orientations and rotations of object #2 |
