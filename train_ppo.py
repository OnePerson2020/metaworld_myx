"""
PPO Training Script for Sawyer Peg Insertion Task
Author: AI Assistant
"""

import os
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import your custom environment
from ppo_test import make_env


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int,
        hidden_sizes: List[int] = [256, 256],
        activation: nn.Module = nn.ReLU,
        log_std_init: float = -0.5
    ):
        super().__init__()
        
        # Build shared layers
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size
        
        # Last hidden layer (shared)
        layers.append(nn.Linear(prev_size, hidden_sizes[-1]))
        layers.append(activation())
        layers.append(nn.LayerNorm(hidden_sizes[-1]))
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Critic head
        self.critic = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean, log_std, and value."""
        features = self.shared_net(obs)
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        value = self.critic(features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from the policy."""
        action_mean, action_log_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action).sum(-1, keepdim=True)
        else:
            action_std = action_log_std.exp()
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        # Ensure action is in [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, value
    
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given actions."""
        action_mean, action_log_std, value = self.forward(obs)
        action_std = action_log_std.exp()
        
        dist = Normal(action_mean, action_std)
        
        # Undo tanh for log_prob calculation
        action = torch.atanh(torch.clamp(action, -0.999, 0.999))
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        
        return log_prob, entropy, value


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.reset()
    
    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0
    
    def add(self, obs, action, reward, done, log_prob, value):
        assert self.ptr < self.buffer_size
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and GAE advantages."""
        advantages = np.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * (1 - next_done) * last_gae_lam
            advantages[t] = last_gae_lam
        
        self.advantages[:self.ptr] = advantages[:self.ptr]
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get_batches(self, batch_size: int):
        """Generate random batches for training."""
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.ptr, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield {
                'observations': torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                'actions': torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                'log_probs': torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[batch_indices]).to(self.device),
                'returns': torch.FloatTensor(self.returns[batch_indices]).to(self.device),
            }


class PPOTrainer:
    """PPO trainer for the peg insertion task."""
    
    def __init__(
        self,
        env_kwargs: dict = {},
        # Network parameters
        hidden_sizes: List[int] = [256, 256],
        # PPO parameters
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        # Training parameters
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_freq: int = 50_000,
        log_dir: str = './logs',
        save_dir: str = './models',
        device: str = 'auto',
        seed: int = 42,
        verbose: int = 1,
    ):
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create environments
        self.env = make_env(seed=seed, **env_kwargs)
        self.eval_env = make_env(seed=seed + 1000, **env_kwargs)
        
        # Get dimensions
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Initialize network
        self.actor_critic = ActorCritic(
            self.obs_dim, 
            self.action_dim, 
            hidden_sizes
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate, eps=1e-5)
        
        # Initialize buffer
        self.rollout_buffer = RolloutBuffer(
            self.obs_dim, 
            self.action_dim, 
            n_steps, 
            self.device
        )
        
        # Store hyperparameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_freq = save_freq
        self.verbose = verbose
        
        # Create directories
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir)
        
        # Training statistics
        self.num_timesteps = 0
        self.num_episodes = 0
        self.best_mean_reward = -np.inf
        
    def collect_rollouts(self) -> dict:
        """Collect rollout data."""
        self.rollout_buffer.reset()
        
        obs, _ = self.env.reset()
        episode_rewards = []
        episode_lengths = []
        episode_infos = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(self.n_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action(obs_tensor)
            
            action_np = action.cpu().numpy()[0]
            log_prob_np = log_prob.cpu().numpy()[0, 0]
            value_np = value.cpu().numpy()[0, 0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.rollout_buffer.add(obs, action_np, reward, done, log_prob_np, value_np)
            
            # Update statistics
            current_episode_reward += reward
            current_episode_length += 1
            self.num_timesteps += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_infos.append(info)
                
                obs, _ = self.env.reset()
                current_episode_reward = 0
                current_episode_length = 0
                self.num_episodes += 1
            else:
                obs = next_obs
        
        # Compute value for last observation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            _, _, last_value = self.actor_critic.get_action(obs_tensor)
            last_value_np = last_value.cpu().numpy()[0, 0]
        
        # Compute returns and advantages
        self.rollout_buffer.compute_returns_and_advantages(last_value_np, self.gamma, self.gae_lambda)
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_infos': episode_infos
        }
    
    def train_step(self) -> dict:
        """Perform one training step of PPO."""
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []
        clip_fractions = []
        
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get_batches(self.batch_size):
                # Evaluate actions
                log_probs, entropy, values = self.actor_critic.evaluate_action(
                    batch['observations'], 
                    batch['actions']
                )
                values = values.squeeze(-1)
                
                # Normalize advantages
                advantages = batch['advantages']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute policy loss
                ratio = torch.exp(log_probs.squeeze(-1) - batch['log_probs'])
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Compute value loss
                if self.clip_range_vf is not None:
                    values_pred = batch['values'] + torch.clamp(
                        values - batch['values'], -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss = F.mse_loss(values_pred, batch['returns'])
                else:
                    value_loss = F.mse_loss(values, batch['returns'])
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                with torch.no_grad():
                    approx_kl_div = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kl_divs.append(approx_kl_div.item())
                    
                    clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'approx_kl_div': np.mean(approx_kl_divs),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def evaluate(self, render: bool = False) -> dict:
        """Evaluate the policy."""
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_insertion_depths = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, _ = self.actor_critic.get_action(obs_tensor, deterministic=True)
                
                action_np = action.cpu().numpy()[0]
                obs, reward, terminated, truncated, info = self.eval_env.step(action_np)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.eval_env.render()
                
                if terminated or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_successes.append(info.get('success', 0.0))
                    episode_insertion_depths.append(info.get('insertion_depth', 0.0))
                    break
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean(episode_successes),
            'mean_insertion_depth': np.mean(episode_insertion_depths)
        }
    
    def save(self, path: str = None):
        """Save the model."""
        if path is None:
            path = os.path.join(self.save_dir, f'model_{self.num_timesteps}.pth')
        
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_episodes': self.num_episodes,
        }, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_episodes = checkpoint['num_episodes']
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting PPO training for {self.total_timesteps} timesteps")
        print(f"Using device: {self.device}")
        
        progress_bar = tqdm(total=self.total_timesteps, desc="Training")
        
        while self.num_timesteps < self.total_timesteps:
            # Collect rollouts
            rollout_info = self.collect_rollouts()
            
            # Train
            train_info = self.train_step()
            
            # Update progress bar
            progress_bar.update(self.n_steps)
            
            # Log training statistics
            if len(rollout_info['episode_rewards']) > 0:
                self.writer.add_scalar('train/episode_reward', 
                                      np.mean(rollout_info['episode_rewards']), 
                                      self.num_timesteps)
                self.writer.add_scalar('train/episode_length', 
                                      np.mean(rollout_info['episode_lengths']), 
                                      self.num_timesteps)
                
                # Log success rate if available
                successes = [info.get('success', 0) for info in rollout_info['episode_infos']]
                if successes:
                    self.writer.add_scalar('train/success_rate', np.mean(successes), self.num_timesteps)
                
                # Log insertion depth
                depths = [info.get('insertion_depth', 0) for info in rollout_info['episode_infos']]
                if depths:
                    self.writer.add_scalar('train/insertion_depth', np.mean(depths), self.num_timesteps)
            
            self.writer.add_scalar('train/policy_loss', train_info['policy_loss'], self.num_timesteps)
            self.writer.add_scalar('train/value_loss', train_info['value_loss'], self.num_timesteps)
            self.writer.add_scalar('train/entropy_loss', train_info['entropy_loss'], self.num_timesteps)
            self.writer.add_scalar('train/approx_kl_div', train_info['approx_kl_div'], self.num_timesteps)
            self.writer.add_scalar('train/clip_fraction', train_info['clip_fraction'], self.num_timesteps)
            
            # Evaluate
            if self.num_timesteps % self.eval_freq < self.n_steps:
                eval_info = self.evaluate()
                
                self.writer.add_scalar('eval/mean_reward', eval_info['mean_reward'], self.num_timesteps)
                self.writer.add_scalar('eval/success_rate', eval_info['success_rate'], self.num_timesteps)
                self.writer.add_scalar('eval/mean_insertion_depth', eval_info['mean_insertion_depth'], self.num_timesteps)
                
                if self.verbose:
                    print(f"\n[Eval] Step: {self.num_timesteps}, "
                          f"Reward: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}, "
                          f"Success: {eval_info['success_rate']*100:.1f}%, "
                          f"Depth: {eval_info['mean_insertion_depth']:.3f}")
                
                # Save best model
                if eval_info['mean_reward'] > self.best_mean_reward:
                    self.best_mean_reward = eval_info['mean_reward']
                    self.save(os.path.join(self.save_dir, 'best_model.pth'))
            
            # Save checkpoint
            if self.num_timesteps % self.save_freq < self.n_steps:
                self.save()
        
        progress_bar.close()
        self.writer.close()
        print("Training completed!")
        
        # Final evaluation
        print("\nFinal evaluation...")
        final_eval = self.evaluate()
        print(f"Final performance - Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}, "
              f"Success: {final_eval['success_rate']*100:.1f}%, "
              f"Depth: {final_eval['mean_insertion_depth']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='PPO training for Peg Insertion task')
    
    # Environment arguments
    parser.add_argument('--render-mode', type=str, default=None, help='Render mode (human/rgb_array)')
    parser.add_argument('--max-steps', type=int, default=300, help='Maximum steps per episode')
    parser.add_argument('--pos-action-scale', type=float, default=0.01, help='Position action scale')
    
    # PPO arguments
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Number of steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--n-epochs', type=int, default=10, help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    
    # Training arguments
    parser.add_argument('--total-timesteps', type=int, default=1_000_000, help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10_000, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=50_000, help='Model save frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    
    # Directories
    parser.add_argument('--log-dir', type=str, default='./logs/ppo_peg_insertion', help='Log directory')
    parser.add_argument('--save-dir', type=str, default='./models/ppo_peg_insertion', help='Save directory')
    
    # Load model
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate the model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PPOTrainer(
        env_kwargs={
            'render_mode': args.render_mode,
            'max_steps': args.max_steps,
            'pos_action_scale': args.pos_action_scale,
            'print_flag': False
        },
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Load model if specified
    if args.load_model:
        trainer.load(args.load_model)
    
    # Train or evaluate
    if args.evaluate_only:
        print("Evaluating model...")
        eval_results = trainer.evaluate(render=(args.render_mode == 'human'))
        print(f"Evaluation results: {eval_results}")
    else:
        trainer.train()


if __name__ == '__main__':
    main()