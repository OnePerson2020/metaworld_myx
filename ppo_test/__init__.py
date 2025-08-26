import pickle
import numpy as np
from .sawyer_peg_insertion_side_v4 import SawyerPegInsertionSideEnvV4
from .types import Task
import gymnasium as gym

def make_env(seed=None, render_mode=None, max_steps=None, print_flag=False, pos_action_scale=0.01):
    env = SawyerPegInsertionSideEnvV4(render_mode=render_mode, print_flag=print_flag, pos_action_scale=pos_action_scale)
    # 创建一个随机任务：每次 reset 时都会重新随机初始化
    task_data = {
        "env_cls": SawyerPegInsertionSideEnvV4,
        "partially_observable": False,
        "freeze": False,
        "seeded_rand_vec": True,
    }
    env.set_task(Task(env_name="peg-insert-side-v4", data=pickle.dumps(task_data)))

    if seed is not None:
        env.seed(seed)

    # 如果需要限制最大步数
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env
