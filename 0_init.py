import gymnasium as gym
import metaworld
import time
import random

ml1 = metaworld.ML1('peg-insert-side-v3')
env = metaworld.make_mt_envs(
    'peg-insert-side-v3',
    render_mode='human',
    width=1080,
    height=1920
)

# 5. 实例化专家策略
# 注意：策略类的名称也可能随着版本更新
from metaworld.policies import SawyerPegInsertionSideV3Policy
policy = SawyerPegInsertionSideV3Policy()

obs, info = env.reset()

# 6. 循环执行直到任务成功
done = False
count = 0
mujoco_env = env.unwrapped
# mujoco_env.mujoco_renderer.viewer.cam.azimuth = 135
mujoco_env.mujoco_renderer.viewer.cam.fixedcamid = 2

while count < 500 and not done:
    # 渲染环境
    env.render()

    # 根据当前观测值获取动作
    action = policy.get_action(obs)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 检查任务是否成功
    # info['success'] 在成功时会返回 1.0
    if info['success'] > 0.5:
        print("任务成功！")
        done = True
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
env.close()