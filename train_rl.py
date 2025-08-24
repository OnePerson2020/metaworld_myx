import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import ppo_test
from tqdm import tqdm


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

    num_envs = 16
    train_env = SubprocVecEnv([make_env_fn(i) for i in range(num_envs)])
    eval_env = ppo_test.make_env(seed=999)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )
    progress_callback = ProgressBarCallback(total_timesteps=200_000)

    # 全部收纳到 CallbackList
    callback = CallbackList([eval_callback, progress_callback])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="cuda",  # 用 GPU
        n_steps=2048,
        batch_size=4096,
    )

    model.learn(total_timesteps=200_000, callback=callback)

    model.save(os.path.join(MODEL_DIR, "ppo_peg_insert_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()
