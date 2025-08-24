import pickle
import numpy as np
from .sawyer_peg_insertion_side_v4 import SawyerPegInsertionSideEnvV4
from .types import Task
import gym

def make_env(seed=None, render_mode=None, max_steps=None, print_falg=False, pos_action_scale=0.01):
    env = SawyerPegInsertionSideEnvV4(render_mode=render_mode, print_falg=print_falg, pos_action_scale=pos_action_scale)
    # 创建一个固定任务：随机初始化一次
    rand_vec = np.array([0.0, 0.6, 0.02, -0.3, 0.6, 0.0]) # peg 和 box 位置
    task_data = {
        "env_cls": SawyerPegInsertionSideEnvV4,
        "rand_vec": rand_vec,
        "partially_observable": False
    }
    env.set_task(Task(env_name="peg-insert-side-v4", data=pickle.dumps(task_data)))
    # 如果需要限制最大步数
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    if seed is not None:
        env.seed(seed)
    return env
