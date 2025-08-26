import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import ppo_test
from tqdm import tqdm


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class TensorboardPhaseCallback(BaseCallback):
    """
    自定义回调，用于记录环境中的阶段信息到TensorBoard
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.phase_counts = {
            'approach': 0,
            'grasp': 0,
            'lift': 0,
            'alignment': 0,
            'insertion': 0,
            'success': 0
        }
        self.episode_rewards = []
        self.current_episode_rewards = []
        self.episode_phases = []

    def _on_step(self) -> bool:
        # 获取环境信息
        infos = self.locals.get('infos', [])
        
        for i, info in enumerate(infos):
            if info is None:
                continue
                
            # 记录阶段信息
            if 'stage_rewards' in info and 'task_phase' in info['stage_rewards']:
                phase = info['stage_rewards']['task_phase']
                self.phase_counts[phase] += 1
                
            # 记录阶段奖励
            if 'stage_rewards' in info:
                for stage_name, stage_reward in info['stage_rewards'].items():
                    if stage_name != 'task_phase':  # 排除阶段名称本身
                        self.logger.record(f'stage_rewards/{stage_name}', stage_reward)
            
            # 记录插入深度
            if 'stage_rewards' in info and 'insertion_depth' in info['stage_rewards']:
                insertion_depth = info['stage_rewards']['insertion_depth']
                self.logger.record('metrics/insertion_depth', insertion_depth)
            
            # 记录成功次数
            if 'success' in info:
                self.phase_counts['success'] += int(info['success'])
                self.logger.record('metrics/success_rate', info['success'])
        
        # 每100步记录一次阶段统计
        if self.n_calls % 100 == 0:
            total_steps = sum(self.phase_counts.values())
            for phase, count in self.phase_counts.items():
                if total_steps > 0:
                    phase_ratio = count / total_steps
                    self.logger.record(f'phase_stats/{phase}_ratio', phase_ratio)
                    self.logger.record(f'phase_stats/{phase}_count', count)
            
            # 重置计数
            self.phase_counts = {k: 0 for k in self.phase_counts.keys()}
        
        return True
    
def make_env_fn(seed):
    def _init():
        env = ppo_test.make_env(seed=seed)
        return env
    return _init


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        self.pbar.close()


def main():
    LOG_DIR = "logs"
    MODEL_DIR = "models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "best_model.zip") 
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")

    num_envs = 16
    train_env_raw = SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
    
    if os.path.exists(stats_path):
        print(f"🔄 从 {stats_path} 加载环境统计数据...")
        train_env = VecNormalize.load(stats_path, train_env_raw)
        train_env.training = True # 确保设置为训练模式
    else:
        print("📊 未找到环境统计数据，将创建新的。")
        train_env = VecNormalize(train_env_raw, norm_obs=True, norm_reward=True)


    eval_env_raw = ppo_test.make_env(seed=999)
    eval_env = VecNormalize(DummyVecEnv([lambda: eval_env_raw]), norm_obs=True, norm_reward=False, training=False)


    eval_frequency_adjusted = max(1, 5000 // num_envs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=eval_frequency_adjusted,
        n_eval_episodes=5,
        deterministic=True,
    )
    total_train_steps = 1_000_000
    progress_callback = ProgressBarCallback(total_timesteps=total_train_steps)
    callback = CallbackList([eval_callback, progress_callback, TensorboardPhaseCallback()]) # 加入自定义的Tensorboard回调

    # --- ✨ 新增：加载模型 (如果存在) 或创建新模型 ---
    if os.path.exists(model_path):
        print(f"🔄 从 {model_path} 加载模型并继续训练...")
        model = PPO.load(model_path, env=train_env, tensorboard_log=LOG_DIR)
    else:
        print("🆕 未找到旧模型，将创建新模型。")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="cuda",
            n_steps=192,
            batch_size=128,
            ent_coef=0.01,
        )
    
    print(f"🚀 开始训练，n_steps = {model.n_steps}")

    # --- 4. 训练模型 ---
    # `reset_num_timesteps=False` 确保日志和回调中的步数是连续的
    model.learn(total_timesteps=total_train_steps, callback=callback, reset_num_timesteps=False)

    # --- 5. 保存最终结果 ---
    print("💾 训练完成，保存最终模型和统计数据...")
    train_env.save(stats_path)
    model.save(os.path.join(MODEL_DIR, "ppo_peg_insert_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()