import gymnasium as gym
import metaworld
import time
import random

# 建议使用 gymnasium.make() 来创建环境，这是官方推荐的标准方式。
# 这样做代码更简洁，并且可以自动处理环境的内部设置。
# render_mode='human' 可以在这里直接指定。
# camera_name 可以指定为 'corner', 'corner2', 'corner3', 'topview', 'behindGripper', 'gripperPOV' 等。
env = gym.make('Meta-World/MT1', env_name='peg-insert-side-v3', render_mode='human', seed=42, camera_name='gripperPOV')


# 实例化专家策略
from metaworld.policies import SawyerPegInsertionSideV3Policy
policy = SawyerPegInsertionSideV3Policy()

# gymnasium 的 reset() 方法返回 obs 和 info
obs, info = env.reset()

# 循环执行直到任务成功或 episode 结束
count = 0
while count < 500:
    # 渲染环境
    # 在新版 gymnasium 中，render() 不需要再被调用，如果 render_mode='human'
    env.render() 

    # 根据当前观测值获取动作
    action = policy.get_action(obs)
    
    # 执行动作
    # step() 返回 obs, reward, terminated, truncated, info
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 检查任务是否成功
    # info['success'] 在成功时会返回 1.0
    if info['success'] > 0.5:
        print("任务成功！")
        break
        
    # 如果环境因为达到最大步数或其他原因结束，也应退出循环
    if terminated or truncated:
        print("Episode 结束。")
        break
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
env.close()