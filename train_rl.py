import ppo_test  # 导入您的环境包
from stable_baselines3 import PPO
import os

if __name__ == "__main__":
    # --- 1. 环境准备 ---
    # 使用您在 ppo_test/__init__.py 中定义的函数创建环境
    # 这个函数已经包含了所有必要的包装器，非常适合训练
    env_name = 'peg-insert-side-v3'
    env = ppo_test.make_mt_envs(
        name=env_name,
        render_mode=None,  # 训练时关闭渲染以提高速度
        # render_mode='human', # 如果想在训练时观察，取消这行注释
    )

    # --- 2. 模型定义 ---
    # 定义模型保存的路径
    log_dir = "rl_models"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "ppo_peg_insert_v3")

    # 定义PPO模型
    # "MlpPolicy": 使用标准的多层感知机作为策略网络
    # env: 告诉模型在哪个环境中学习
    # verbose=1: 在训练时打印学习进度
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/" # 可选：保存TensorBoard日志
    )

    # --- 3. 开始训练 ---
    # total_timesteps: 训练的总步数，可以先设小一点测试，例如 50000
    # 然后根据效果增加到 200000 或更多
    print(f"开始在 {env_name} 环境中训练PPO模型...")
    model.learn(total_timesteps=50000)
    print("训练完成！")

    # --- 4. 保存模型 ---
    model.save(model_path)
    print(f"模型已保存至: {model_path}.zip")

    env.close()