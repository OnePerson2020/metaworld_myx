"""
Test and Visualization Script for Trained PPO Model
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict, Any
import json
from datetime import datetime

from ppo_test import make_env
from train_ppo import ActorCritic


class PolicyTester:
    """Test and visualize trained PPO policy."""
    
    def __init__(
        self,
        model_path: str,
        env_kwargs: dict = {},
        device: str = 'auto',
        seed: int = 42
    ):
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Create environment
        self.env = make_env(seed=seed, **env_kwargs)
        
        # Load model
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.actor_critic = ActorCritic(
            self.obs_dim,
            self.action_dim,
            hidden_sizes=[256, 256]
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.actor_critic.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Trained for {checkpoint['num_timesteps']} timesteps")
    
    def test_episodes(
        self, 
        n_episodes: int = 10, 
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Test the policy for multiple episodes."""
        
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_successes': [],
            'episode_insertion_depths': [],
            'episode_phases': [],
            'episode_trajectories': []
        }
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'infos': []
            }
            
            if verbose:
                print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
            
            while True:
                # Store observation
                trajectory['observations'].append(obs.copy())
                
                # Get action
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.actor_critic.get_action(obs_tensor, deterministic=True)
                action_np = action.cpu().numpy()[0]
                
                # Store action
                trajectory['actions'].append(action_np.copy())
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action_np)
                
                # Store reward and info
                trajectory['rewards'].append(reward)
                trajectory['infos'].append(info)
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
                
                if verbose and episode_length % 50 == 0:
                    print(f"  Step {episode_length}: Reward={reward:.3f}, "
                          f"Phase={info.get('task_phase', 'unknown')}, "
                          f"Depth={info.get('insertion_depth', 0):.3f}")
                
                if terminated or truncated:
                    success = info.get('success', 0.0)
                    insertion_depth = info.get('insertion_depth', 0.0)
                    final_phase = info.get('task_phase', 'unknown')
                    
                    results['episode_rewards'].append(episode_reward)
                    results['episode_lengths'].append(episode_length)
                    results['episode_successes'].append(success)
                    results['episode_insertion_depths'].append(insertion_depth)
                    results['episode_phases'].append(final_phase)
                    results['episode_trajectories'].append(trajectory)
                    
                    if verbose:
                        print(f"  Episode finished: Reward={episode_reward:.2f}, "
                              f"Length={episode_length}, Success={bool(success)}, "
                              f"Depth={insertion_depth:.3f}, Phase={final_phase}")
                    break
        
        # Compute statistics
        results['statistics'] = {
            'mean_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'mean_length': np.mean(results['episode_lengths']),
            'success_rate': np.mean(results['episode_successes']),
            'mean_insertion_depth': np.mean(results['episode_insertion_depths']),
            'max_insertion_depth': np.max(results['episode_insertion_depths']),
        }
        
        return results
    
    def visualize_trajectory(
        self, 
        trajectory: Dict[str, List],
        save_path: str = None
    ):
        """Visualize a single episode trajectory."""
        
        observations = np.array(trajectory['observations'])
        actions = np.array(trajectory['actions'])
        rewards = trajectory['rewards']
        infos = trajectory['infos']
        
        # Extract key information
        tcp_positions = observations[:, :3]
        obj_positions = observations[:, 14:17]
        insertion_depths = [info.get('insertion_depth', 0) for info in infos]
        phases = [info.get('task_phase', 'unknown') for info in infos]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Episode Trajectory Analysis', fontsize=16)
        
        # 1. TCP trajectory in 3D space
        ax = axes[0, 0]
        ax.plot(tcp_positions[:, 0], tcp_positions[:, 1], 'b-', alpha=0.7)
        ax.scatter(tcp_positions[0, 0], tcp_positions[0, 1], c='g', s=100, label='Start')
        ax.scatter(tcp_positions[-1, 0], tcp_positions[-1, 1], c='r', s=100, label='End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('TCP Trajectory (XY plane)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Height over time
        ax = axes[0, 1]
        ax.plot(tcp_positions[:, 2], 'b-', label='TCP')
        ax.plot(obj_positions[:, 2], 'r-', label='Object')
        ax.set_xlabel('Step')
        ax.set_ylabel('Z position')
        ax.set_title('Height over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Insertion depth
        ax = axes[0, 2]
        ax.plot(insertion_depths, 'g-', linewidth=2)
        ax.axhline(y=0.04, color='r', linestyle='--', label='Success threshold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Insertion Depth')
        ax.set_title('Insertion Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Rewards
        ax = axes[1, 0]
        ax.plot(rewards, 'b-')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward over Time')
        ax.grid(True, alpha=0.3)
        
        # 5. Cumulative reward
        ax = axes[1, 1]
        cumulative_rewards = np.cumsum(rewards)
        ax.plot(cumulative_rewards, 'g-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Reward')
        ax.grid(True, alpha=0.3)
        
        # 6. Actions
        ax = axes[1, 2]
        for i in range(actions.shape[1]):
            ax.plot(actions[:, i], label=f'Action {i}', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Action Value')
        ax.set_title('Actions over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 7. Distance to object
        ax = axes[2, 0]
        tcp_to_obj_dist = np.linalg.norm(tcp_positions - obj_positions, axis=1)
        ax.plot(tcp_to_obj_dist, 'purple')
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance')
        ax.set_title('TCP to Object Distance')
        ax.grid(True, alpha=0.3)
        
        # 8. Task phases
        ax = axes[2, 1]
        phase_mapping = {phase: i for i, phase in enumerate(set(phases))}
        phase_numbers = [phase_mapping[phase] for phase in phases]
        ax.plot(phase_numbers, 'o-', markersize=2)
        ax.set_yticks(list(phase_mapping.values()))
        ax.set_yticklabels(list(phase_mapping.keys()))
        ax.set_xlabel('Step')
        ax.set_title('Task Phase Progression')
        ax.grid(True, alpha=0.3)
        
        # 9. Stage rewards breakdown
        ax = axes[2, 2]
        if 'stage_rewards' in infos[0]:
            stage_names = list(infos[0]['stage_rewards'].keys())
            stage_names = [name for name in stage_names 
                          if name not in ['insertion_depth', 'lateral_distance', 'task_phase']]
            
            for stage_name in stage_names[:5]:  # Plot top 5 reward components
                stage_values = [info['stage_rewards'].get(stage_name, 0) for info in infos]
                ax.plot(stage_values, label=stage_name, alpha=0.7)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward Component')
            ax.set_title('Stage Rewards Breakdown')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Trajectory visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def analyze_performance(
        self, 
        results: Dict[str, Any],
        save_path: str = None
    ):
        """Analyze and visualize overall performance."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Performance Analysis', fontsize=16)
        
        # 1. Episode rewards distribution
        ax = axes[0, 0]
        ax.hist(results['episode_rewards'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['episode_rewards']), color='r', 
                  linestyle='--', label=f"Mean: {np.mean(results['episode_rewards']):.2f}")
        ax.set_xlabel('Episode Reward')
        ax.set_ylabel('Count')
        ax.set_title('Episode Rewards Distribution')
        ax.legend()
        
        # 2. Episode lengths
        ax = axes[0, 1]
        ax.hist(results['episode_lengths'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(results['episode_lengths']), color='r', 
                  linestyle='--', label=f"Mean: {np.mean(results['episode_lengths']):.0f}")
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Count')
        ax.set_title('Episode Lengths Distribution')
        ax.legend()
        
        # 3. Insertion depths
        ax = axes[0, 2]
        ax.hist(results['episode_insertion_depths'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0.04, color='g', linestyle='--', label='Success threshold')
        ax.axvline(np.mean(results['episode_insertion_depths']), color='r', 
                  linestyle='--', label=f"Mean: {np.mean(results['episode_insertion_depths']):.3f}")
        ax.set_xlabel('Insertion Depth')
        ax.set_ylabel('Count')
        ax.set_title('Insertion Depths Distribution')
        ax.legend()
        
        # 4. Success rate pie chart
        ax = axes[1, 0]
        success_count = sum(results['episode_successes'])
        fail_count = len(results['episode_successes']) - success_count
        ax.pie([success_count, fail_count], labels=['Success', 'Failure'], 
               autopct='%1.1f%%', colors=['green', 'red'])
        ax.set_title(f'Success Rate: {results["statistics"]["success_rate"]*100:.1f}%')
        
        # 5. Final phases distribution
        ax = axes[1, 1]
        phases, counts = np.unique(results['episode_phases'], return_counts=True)
        ax.bar(phases, counts)
        ax.set_xlabel('Final Phase')
        ax.set_ylabel('Count')
        ax.set_title('Final Task Phases')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Performance metrics summary
        ax = axes[1, 2]
        ax.axis('off')
        metrics_text = f"""Performance Summary:
        
Mean Reward: {results['statistics']['mean_reward']:.2f} ± {results['statistics']['std_reward']:.2f}
Success Rate: {results['statistics']['success_rate']*100:.1f}%
Mean Length: {results['statistics']['mean_length']:.0f} steps
Mean Depth: {results['statistics']['mean_insertion_depth']:.3f}
Max Depth: {results['statistics']['max_insertion_depth']:.3f}
Total Episodes: {len(results['episode_rewards'])}
"""
        ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Performance analysis saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def save_results(self, results: Dict[str, Any], save_dir: str = './test_results'):
        """Save test results to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save statistics as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stats_path = os.path.join(save_dir, f'statistics_{timestamp}.json')
        
        stats_to_save = {
            'statistics': results['statistics'],
            'episode_rewards': results['episode_rewards'],
            'episode_lengths': results['episode_lengths'],
            'episode_successes': results['episode_successes'],
            'episode_insertion_depths': results['episode_insertion_depths'],
            'episode_phases': results['episode_phases']
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        print(f"Results saved to {stats_path}")
        
        return stats_path


def main():
    parser = argparse.ArgumentParser(description='Test and visualize trained PPO model')
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    
    # Environment arguments
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--max-steps', type=int, default=300, help='Maximum steps per episode')
    parser.add_argument('--pos-action-scale', type=float, default=0.01, help='Position action scale')
    
    # Test arguments
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Visualization arguments
    parser.add_argument('--visualize-trajectory', action='store_true', help='Visualize trajectory')
    parser.add_argument('--visualize-performance', action='store_true', help='Visualize performance')
    parser.add_argument('--save-results', action='store_true', help='Save test results')
    parser.add_argument('--save-dir', type=str, default='./test_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create tester
    tester = PolicyTester(
        model_path=args.model_path,
        env_kwargs={
            'render_mode': 'human' if args.render else None,
            'max_steps': args.max_steps,
            'pos_action_scale': args.pos_action_scale,
            'print_flag': False
        },
        device=args.device,
        seed=args.seed
    )
    
    # Run tests
    print(f"\nTesting policy for {args.n_episodes} episodes...")
    results = tester.test_episodes(
        n_episodes=args.n_episodes,
        render=args.render,
        verbose=args.verbose
    )
    
    # Print summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    for key, value in results['statistics'].items():
        print(f"{key}: {value:.3f}")
    
    # Visualize best trajectory
    if args.visualize_trajectory and results['episode_trajectories']:
        # Find best episode
        best_idx = np.argmax(results['episode_rewards'])
        print(f"\nVisualizing best episode (Episode {best_idx + 1}, Reward: {results['episode_rewards'][best_idx]:.2f})")
        
        save_path = None
        if args.save_results:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'best_trajectory.png')
        
        tester.visualize_trajectory(
            results['episode_trajectories'][best_idx],
            save_path=save_path
        )
    
    # Visualize overall performance
    if args.visualize_performance:
        save_path = None
        if args.save_results:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'performance_analysis.png')
        
        tester.analyze_performance(results, save_path=save_path)
    
    # Save results
    if args.save_results:
        tester.save_results(results, save_dir=args.save_dir)


if __name__ == '__main__':
    main()