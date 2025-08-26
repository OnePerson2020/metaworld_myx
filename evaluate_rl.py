import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import ppo_test

# --- 1. 设置命令行参数解析 ---
parser = argparse.ArgumentParser(description="Evaluate a PPO model with an option to enable plotting.")
parser.add_argument('--plot', action='store_true', help='Enable real-time plotting of actions and observations.')
args = parser.parse_args()

# --- 2. 根据参数决定是否初始化绘图 ---
if args.plot:
    print("绘图功能已开启。")
    plt.ion()  # 开启交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    # 动作和观测历史记录
    history_steps = []
    history_actions = []
    history_obs = []
    history_rewards = []
else:
    print("绘图功能已关闭。如需开启，请使用 --plot 参数运行。")

# --- 3. 加载训练好的模型 ---
model_path = "models/best_model.zip"
stats_path = "models/vec_normalize.pkl"
try:
    model = PPO.load(model_path)
    print(f"从 {model_path} 加载模型成功！")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 {model_path}，请先运行 train_rl.py。")
    exit()

# --- 4. 创建评估环境 ---
MAX_STEPS = 200
eval_env_raw = ppo_test.make_env(
    seed=999,
    render_mode="human",
    max_steps=MAX_STEPS,
    print_flag=True
)

# 使用加载的统计数据封装环境
try:
    eval_env_vec = DummyVecEnv([lambda: eval_env_raw])

    # 现在传入向量化后的环境
    env = VecNormalize.load(stats_path, eval_env_vec)
    print(f"从 {stats_path} 加载归一化统计数据成功！")
    
    env.training = False
    env.norm_reward = False
except FileNotFoundError:
    print(f"错误: 找不到归一化文件 {stats_path}，请先完整运行 train_rl.py。")
    exit()

# --- 5. 运行评估 ---
num_episodes = 10
success_count = 0

for ep in range(num_episodes):
    print(f"\n=== 开始评估第 {ep+1}/{num_episodes} 局 ===")
    obs = env.reset()

    # 如果开启绘图，则重置数据
    if args.plot:
        history_steps.clear()
        history_actions.clear()
        history_obs.clear()
        history_rewards.clear()

    # 调整摄像机
    mujoco_env = env.envs[0] 
    if hasattr(mujoco_env, 'mujoco_renderer') and mujoco_env.mujoco_renderer.viewer:
        mujoco_env.mujoco_renderer.viewer.cam.azimuth = 245
        mujoco_env.mujoco_renderer.viewer.cam.elevation = -20

    episode_reward = 0
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Step {step}, Action: {action}")

        obs, reward, done, info = env.step(action)
        original_obs = env.get_original_obs()
        episode_reward += reward[0] 
        info_dict = info[0]

        # --- 仅在开启绘图时记录数据和更新图表 ---
        if args.plot:
            # 记录数据
            history_steps.append(step)
            history_actions.append(action.copy())
            history_obs.append(original_obs.copy())
            history_rewards.append(episode_reward)

            # 实时绘图更新
            if step % 2 == 0:  # 每2步更新一次，避免太卡
                ax1.clear()
                ax2.clear()

                # 转为 NumPy 数组便于处理
                arr_actions = np.array(history_actions)
                arr_obs = np.array(history_obs)

                # 绘制动作
                for i in range(arr_actions.shape[1]):
                    ax1.plot(history_steps, arr_actions[:, i], label=f'Action {i}')
                ax1.set_title(f"Episode {ep+1} - Actions Over Time")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Action Value")
                ax1.legend()
                ax1.grid(True)

                # 绘制观测（取前几维或关键维度）
                obs_dim = arr_obs.shape[1]
                plot_dims = min(6, obs_dim)  # 最多画前6维
                for i in range(plot_dims):
                    ax2.plot(history_steps, arr_obs[:, i], label=f'Obs {i}')
                ax2.set_title("Observations Over Time")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Obs Value")
                ax2.legend()
                ax2.grid(True)

                # 刷新图像
                fig.tight_layout()
                plt.pause(0.01)  # 短暂暂停以刷新

        env.render()

        # 检查成功
        if info_dict.get("success", 0.0) > 0.5:
            print("✅ 成功插入 Peg！")
            success_count += 1
            time.sleep(1.5)
            break

        step += 1

    print(f"该局累计奖励: {episode_reward:.2f}")

    # 如果绘图，则暂停一下保持图表
    if args.plot:
        time.sleep(1)

print("\n=== 评估结束 ===")
print(f"总回合数: {num_episodes}")
print(f"成功次数: {success_count}")
print(f"成功率: {success_count / num_episodes * 100:.1f}%")

# --- 6. 关闭环境和图形 ---
env.close()
if args.plot:
    plt.ioff()
    plt.show()  # 保持最终图像显示