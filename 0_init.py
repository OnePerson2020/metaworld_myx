import gymnasium as gym
import ppo_test
import time
import random
import mujoco

env_name = 'peg-insert-side-v3'

# env = metaworld.make_mt_envs(
#     'peg-insert-side-v3',
#     render_mode='human',
#     width=1080,
#     height=1920
# )

env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
env = env_class(render_mode='human', width=1080, height=1920)
benchmark = ppo_test.MT1(env_name)
task = benchmark.train_tasks[0]  # 使用第一个训练任务
env.set_task(task)

# env = gym.make('Meta-World/MT1', env_name=env_name, render_mode='human', width=1080, height=1920)

from ppo_test.policies import SawyerPegInsertionSideV3Policy
policy = SawyerPegInsertionSideV3Policy()

obs, info = env.reset()

# 6. 循环执行直到任务成功
done = False
count = 0

mujoco_env = env.unwrapped
mujoco_env.mujoco_renderer.viewer.cam.azimuth = 135
# mujoco_env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
# mujoco_env.mujoco_renderer.viewer.cam.fixedcamid = 2

while count < 500 and not done:
    # 渲染环境
    env.render()

    # 根据当前观测值获取动作
    action = policy.get_action(obs)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 检查任务是否成功
    if info['success'] > 0.5:
        print("任务成功！")
        done = True
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
env.close()

