import ppo_test
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
# 导入矢量化环境相关的工具
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import torch

# 建议将创建单个环境的函数封装起来
# 这样 make_vec_env 才能正确调用它
def make_env(env_name, render_mode=None):
    # 假设 ppo_test.make_mt_envs 创建并返回一个环境实例
    env = ppo_test.make_mt_envs(name=env_name, render_mode=render_mode)
    # Monitor 最好在创建单个环境时就包装好
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # --- 1. 环境准备 ---
    env_name = 'peg-insert-side-v3'
    log_dir = "rl_models"
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取可用的CPU核心数，作为并行环境的数量
    num_cpu = os.cpu_count() or 4 # 如果获取失败，则默认为4
    print(f"使用 {num_cpu} 个并行环境进行训练。")

    # (核心修改) 创建矢量化训练环境
    # SubprocVecEnv 会在独立的进程中运行每个环境，实现真正的并行
    train_env = make_vec_env(
        make_env, 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv,
        # 传递给 make_env 的参数
        env_kwargs=dict(env_name=env_name, render_mode=None) 
    )

    # --- 2. 评估回调 ---
    # 评估环境通常只需要一个
    eval_env = make_vec_env(make_env, n_envs=1, env_kwargs=dict(env_name=env_name, render_mode=None))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        # eval_freq 需要根据并行环境数调整。
        # 原来是5000，现在总步数是 num_cpu 倍速，可以适当增加频率
        # eval_freq 的值是总时间步数，不是单个环境的步数，所以可以保持不变或按需调整
        eval_freq=max(5000 // num_cpu, 512),
        deterministic=True,
        render=False
    )
    
    # --- 3. 模型定义 (使用优化的超参数) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",
        train_env, # 使用矢量化环境
        verbose=1,
        n_steps=2048, # 每个环境收集 n_steps / num_cpu 步
        batch_size=256, # (关键修改) 增大 batch_size
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