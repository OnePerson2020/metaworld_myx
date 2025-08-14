import os
import ppo_test
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. 定义常量和配置 ---
ENV_NAME = 'peg-insert-side-v3'

# 日志和模型保存路径
LOG_DIR = "rl_logs"
MODEL_SAVE_DIR = "rl_models"
BEST_MODEL_SAVE_PATH = "rl_models" # 对应 evaluate_rl.py 中的路径
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

# 训练总步数 (这是一个关键的超参数，需要根据训练效果调整)
# 对于第一阶段，可以先从一个较小的值开始，如 20万 或 50万
TOTAL_TIMESTEPS = 500_000

# --- 2. 创建训练和评估环境 ---
print("正在创建环境...")
# 训练环境，不需要渲染以加快速度
train_env = ppo_test.make_mt_envs(name=ENV_NAME)

# 评估环境，用于在训练过程中周期性地测试模型性能
# EvalCallback 将使用这个环境来确定哪个模型是 "best_model"
eval_env = ppo_test.make_mt_envs(name=ENV_NAME)
print("环境创建完成。")

# --- 3. 设置评估回调 ---
# EvalCallback 是一个强大的工具，它会:
# 1. 定期 (eval_freq) 在评估环境 (eval_env) 上运行模型。
# 2. 评估 N 次 (n_eval_episodes)。
# 3. 如果当前模型效果是历史最佳，则将其保存到 best_model_save_path。
# 4. deterministic=True 表示使用确定性策略进行评估，更准确地反映模型学习到的策略。
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_SAVE_PATH,
    log_path=LOG_DIR,
    eval_freq=10000, # 每 10000 步评估一次
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# --- 4. 初始化或加载模型 ---
# 这里我们总是从头开始创建一个新模型，因为这是训练脚本
# "MlpPolicy" 适用于基于向量/数值的观察空间 (state-based observation)
print("正在初始化 PPO 模型...")
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,  # 打印训练过程中的信息
    tensorboard_log=LOG_DIR, # 启用 TensorBoard 日志
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.0,
    clip_range=0.2
)
print("模型初始化完成。")

# --- 5. 开始训练 ---
print("\n--- 开始训练 ---")
# model.learn 会执行整个训练过程
# callback 参数让我们能在训练中途执行特定操作，如保存最佳模型
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)
print("\n--- 训练完成 ---")

# --- 6. 保存最终模型 ---
final_model_path = os.path.join(MODEL_SAVE_DIR, f"ppo_{ENV_NAME}_final.zip")
model.save(final_model_path)
print(f"最终模型已保存至: {final_model_path}")
print(f"最佳模型已保存在 '{BEST_MODEL_SAVE_PATH}/' 目录下，可用于 evaluate_rl.py。")

# --- 7. 关闭环境 ---
train_env.close()
eval_env.close()