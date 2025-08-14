import ppo_test
from stable_baselines3 import PPO
import time
import numpy as np

# --- 1. 加载训练好的模型 ---
# model_path = "rl_models/ppo_peg_insert_v3_final.zip"
model_path = "rl_models/ppo_approach_stage_final.zip"
try:
    model = PPO.load(model_path)
    print(f"从 {model_path} 加载模型成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 {model_path}。请先运行 train_rl.py 进行训练。")
    exit()


# --- 2. 创建用于评估的环境 ---
# 注意这里我们打开了渲染模式 'human'
env = ppo_test.make_mt_envs(
    name='peg-insert-side-v3',
    render_mode='human',
    width=1000,
    height=720
)

# --- 3. 运行评估 ---
num_episodes = 10
success_count = 0
for i in range(num_episodes):
    print(f"\n--- 开始评估轮次 {i+1}/{num_episodes} ---")
    obs, info = env.reset()
    done = False
    
    # 尝试设置一个好看的视角
    # env.unwrapped 返回最原始的环境实例，绕过所有包装器
    mujoco_env = env.unwrapped 
    if hasattr(mujoco_env, 'mujoco_renderer') and mujoco_env.mujoco_renderer.viewer is not None:
        mujoco_env.mujoco_renderer.viewer.cam.azimuth = 245
        mujoco_env.mujoco_renderer.viewer.cam.elevation = -20

    episode_reward = 0
    while not done:
        # 使用RL模型根据观测值预测动作
        # a. 这里的 model.predict() 代替了你之前写的 CorrectedPolicyV2.get_action()
        # b. deterministic=True 表示使用模型的确定性策略进行评估，而不是带噪声的探索性策略
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        episode_reward += reward
        env.render()
        
        # 检查是否成功
        # info 字典来自环境的 evaluate_state 函数
        if info.get('success', 0.0) > 0.5:
            print("任务成功！")
            success_count += 1
            time.sleep(1.5) # 暂停观察
            break # 成功后直接开始下一轮
        
        done = terminated or truncated

    print(f"本轮次结束。累计奖励: {episode_reward:.2f}")

print("\n--- 评估完成 ---")
print(f"总轮数: {num_episodes}")
print(f"成功次数: {success_count}")
print(f"成功率: {success_count / num_episodes * 100:.2f}%")
env.close()