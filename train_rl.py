import ppo_test
from stable_baselines3 import PPO
# 导入 Monitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import torch

if __name__ == "__main__":
    # --- 1. 环境准备 ---
    env_name = 'peg-insert-side-v3'
    log_dir = "rl_models"
    
    os.makedirs(log_dir, exist_ok=True)

    # 创建训练环境
    # PPO/A2C等算法在内部处理统计，所以训练环境不强制要求Monitor，但加上也无妨
    train_env = Monitor(ppo_test.make_mt_envs(
        name=env_name,
        render_mode=None,
    ))

    # --- 2. 评估回调 (Best Practice) ---
    eval_env = Monitor(ppo_test.make_mt_envs(name=env_name, render_mode=None))
    
    # EvalCallback 会在训练过程中定期评估模型，并只保存表现最好的模型
    eval_callback = EvalCallback(
        eval_env, # 使用包装后的环境
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # --- 3. 模型定义 (使用优化的超参数) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",
        train_env, # 使用包装后的训练环境
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_tensorboard/",
        device=device
    )

    # --- 4. 开始训练 ---
    total_timesteps = 500000 
    print(f"开始在 {env_name} 环境中训练PPO模型...")
    print(f"总训练步数: {total_timesteps}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    print("训练完成！")

    # --- 5. 保存最终模型 ---
    final_model_path = os.path.join(log_dir, "ppo_peg_insert_v3_final")
    model.save(final_model_path)
    print(f"最好的模型已由Callback自动保存至: {log_dir}/best_model.zip")
    print(f"最终模型已保存至: {final_model_path}.zip")

    train_env.close()
    eval_env.close()