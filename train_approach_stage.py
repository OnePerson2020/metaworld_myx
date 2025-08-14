import os
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_checker import check_env

import ppo_test

def make_env():
    """创建单个环境实例"""
    env = ppo_test.make_mt_envs(
        name='peg-insert-side-v3',
        seed=42,
        max_episode_steps=100,
        render_mode=None
    )
    return env

def main():
    # --- 1. 设置训练参数 ---
    TOTAL_TIMESTEPS = 200000  # 第一阶段可以先用较少的步数
    EVAL_FREQ = 10000  # 每10000步评估一次
    N_EVAL_EPISODES = 10
    SAVE_FREQ = 20000  # 每20000步保存一次检查点
    
    # --- 2. 创建训练和评估环境 ---
    print("创建训练环境...")
    train_env = make_env()
    # check_env(train_env)
    train_env = Monitor(train_env, "rl_logs/train_monitor.csv")
    train_env = DummyVecEnv([lambda: train_env])
    
    # 使用VecNormalize来标准化观察和奖励，这对PPO很重要
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    print("创建评估环境...")
    eval_env = make_env()
    # check_env(eval_env)
    eval_env = Monitor(eval_env, "rl_logs/eval_monitor.csv")
    eval_env = DummyVecEnv([lambda: eval_env])
    # 评估环境通常不需要标准化奖励，但需要标准化观察以保持一致性
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False,  # 评估时不标准化奖励
        clip_obs=10.0,
        training=False  # 评估环境不更新标准化参数
    )
    
    # --- 3. 配置 PPO 模型 ---
    print("初始化PPO模型...")
    
    # PPO 超参数配置（针对第一阶段优化）
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log="rl_logs/",
        policy_kwargs=dict(
            activation_fn=torch.nn.Tanh,  # 激活函数
            net_arch=[256, 256, 256],    # 网络架构
        ),
        verbose=1,
    )
    
    # --- 4. 设置回调函数 ---
    print("设置回调函数...")
    
    # 评估回调：定期评估模型性能并保存最佳模型
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="rl_models/",
        log_path="rl_logs/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=1
    )
    
    # 检查点回调：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path="rl_models/checkpoints/",
        name_prefix="ppo_approach_stage"
    )
    
    # 组合回调
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # --- 5. 开始训练 ---
    print(f"开始训练第一阶段（approach）...")
    print(f"总训练步数: {TOTAL_TIMESTEPS}")
    print(f"评估频率: 每{EVAL_FREQ}步")
    print(f"检查点保存频率: 每{SAVE_FREQ}步")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            log_interval=10,  # 每10次更新打印一次日志
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成！总用时: {training_time/3600:.2f} 小时")
        
        # --- 6. 保存最终模型 ---
        print("保存最终模型...")
        model.save("rl_models/ppo_approach_stage_final")
        
        # 保存环境标准化参数
        train_env.save("rl_models/vecnormalize_approach_stage.pkl")
        
        print("模型保存完成！")
        
        # --- 7. 快速测试 ---
        print("\n进行快速测试...")
        test_episodes = 3
        success_count = 0
        
        # 为测试环境应用训练时的标准化参数
        eval_env.load("rl_models/vecnormalize_approach_stage.pkl", train_env)
        
        for i in range(test_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                # 检查是否接近目标（第一阶段的成功标准）
                if info[0].get('success', 0.0) > 0.5:
                    success_count += 1
                    print(f"测试轮次 {i+1}: 成功! 奖励: {episode_reward:.3f}, 步数: {step_count}")
                    break
                
                if step_count >= 1000:  # 避免无限循环
                    print(f"测试轮次 {i+1}: 超时. 奖励: {episode_reward:.3f}, 步数: {step_count}")
                    break
        
        print(f"\n快速测试结果: {success_count}/{test_episodes} 成功")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("保存当前模型...")
        model.save("rl_models/ppo_approach_stage_interrupted")
        train_env.save("rl_models/vecnormalize_approach_stage_interrupted.pkl")
    
    finally:
        # 清理环境
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("PPO训练 - 第一阶段：接近目标物体 (Approach Stage)")
    print("=" * 60)
    main()