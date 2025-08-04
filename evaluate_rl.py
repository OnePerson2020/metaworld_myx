import ppo_test
from stable_baselines3 import PPO
import time

# --- 1. 加载训练好的模型 ---
model_path = "rl_models/ppo_peg_insert_v3.zip"
model = PPO.load(model_path)
print("模型加载成功！")

# --- 2. 创建用于评估的环境 ---
# 注意这里我们打开了渲染模式 'human'
env = ppo_test.make_mt_envs(
    name='peg-insert-side-v3',
    render_mode='human',
    width=1000,
    height=720
)

# --- 3. 运行评估 ---
num_episodes = 5
for i in range(num_episodes):
    print(f"\n--- 开始评估轮次 {i+1}/{num_episodes} ---")
    obs, info = env.reset()
    done = False
    
    # 设置一个好看的视角
    mujoco_env = env.unwrapped
    if hasattr(mujoco_env, 'mujoco_renderer'):
        mujoco_env.mujoco_renderer.viewer.cam.azimuth = 245
        mujoco_env.mujoco_renderer.viewer.cam.elevation = -20

    while not done:
        # 使用模型根据观测值预测动作
        # a. 这里的 model.predict() 代替了您之前写的 policy.get_action()
        # b. deterministic=True 表示使用模型的确定性策略进行评估，而不是带噪声的探索性策略
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if info.get('success', 0.0) > 0.5:
            print("任务成功！")
            time.sleep(1) # 暂停1秒观察
            break
        
        done = terminated or truncated

print("评估完成。")
env.close()