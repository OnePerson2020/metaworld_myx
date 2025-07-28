import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random
import pandas as pd

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

env_name = 'peg-insert-side-v3'
env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
env = env_class(render_mode='rgb_array', width=1280, height=720) # human or rgb_array
benchmark = ppo_test.MT1(env_name)
task = random.choice(benchmark.train_tasks)
env.set_task(task)
timestep = env.model.opt.timestep

print(f"Current simulation timestep: {timestep}")

obs, info = env.reset()

done = False
count = 0
while count < 500 and not done:
    env.render()
    action = np.zeros(7)
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    if info.get('success', 0.0) > 0.5:
        print("任务成功！")
        # time.sleep(1)
        break
    # time.sleep(0.01)
    count += 1

env.close()