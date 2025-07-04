import gymnasium as gym
import metaworld
import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import time
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PegInsertSideHybridEnv:
    """
    基于Meta-World的Peg Insert Side环境，集成力位混合控制
    专门用于研究方向选择矩阵对插入任务性能的影响
    """
    
    def __init__(self, render_mode: str = "human", record_data: bool = True):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式 ("human", "rgb_array", None)
            record_data: 是否记录实验数据
        """
        # 创建Meta-World环境，禁用观测空间检查
        self.env = gym.make('Meta-World/MT1', env_name='peg-insert-side-v3', 
                           render_mode=render_mode, disable_env_checker=True)
        
        # 获取环境信息
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # 数据记录
        self.record_data = record_data
        self.reset_data_logging()
        
        # 力位混合控制相关参数
        self.setup_force_position_control()
        
        # 任务相关参数
        self.peg_pos = None
        self.hole_pos = None
        self.contact_threshold = 1.0  # 接触力阈值(N)
        self.insertion_threshold = 0.02  # 插入成功阈值(m)
        
        # 控制频率和时间步长
        self.dt = 0.02  # 20ms控制周期
        self.max_episode_steps = 500
        self.current_step = 0
        
    def reset_data_logging(self):
        """重置数据记录"""
        if self.record_data:
            self.data_log = {
                'time': [],
                'peg_pos': [],
                'hole_pos': [],
                'eef_pos': [],
                'contact_force': [],
                'desired_pose': [],
                'desired_force': [],
                'control_action': [],
                'reward': [],
                'insertion_depth': [],
                'success': [],
                'strategy': []
            }
        
    def setup_force_position_control(self):
        """设置力位混合控制参数"""
        
        # 分配矩阵 S (6x6): 对角线元素决定每个自由度的控制模式
        # 1: 力控制, 0: 位置控制
        # 顺序: [x, y, z, rx, ry, rz]
        self.selection_matrices = {
            # 基线策略: 全位置控制
            'position_only': np.diag([0, 0, 0, 0, 0, 0]),
            
            # 策略1: X方向力控制 (插入方向)
            'force_x': np.diag([1, 0, 0, 0, 0, 0]),
            
            # 策略2: Y方向力控制 (垂直插入方向)  
            'force_y': np.diag([0, 1, 0, 0, 0, 0]),
            
            # 策略3: Z方向力控制 (垂直方向)
            'force_z': np.diag([0, 0, 1, 0, 0, 0]),
            
            # 策略4: X,Y方向力控制
            'force_xy': np.diag([1, 1, 0, 0, 0, 0]),
            
            # 策略5: X,Z方向力控制
            'force_xz': np.diag([1, 0, 1, 0, 0, 0]),
            
            # 策略6: Y,Z方向力控制
            'force_yz': np.diag([0, 1, 1, 0, 0, 0]),
            
            # 策略7: X,Y,Z方向力控制
            'force_xyz': np.diag([1, 1, 1, 0, 0, 0]),
            
            # 策略8: 自适应策略 (根据阶段动态切换)
            'adaptive': np.diag([0, 0, 0, 0, 0, 0])  # 初始值，动态更新
        }
        
        # 当前使用的分配矩阵
        self.current_strategy = 'position_only'
        self.S = self.selection_matrices[self.current_strategy].copy()
        
        # 控制增益 - 针对peg-insert-side任务优化
        self.kp_pos = np.array([800.0, 800.0, 800.0, 100.0, 100.0, 100.0])  # 位置控制增益
        self.kd_pos = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 10.0])        # 位置控制阻尼
        self.kp_force = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05])         # 力控制比例增益
        self.ki_force = np.array([0.02, 0.02, 0.02, 0.005, 0.005, 0.005])   # 力控制积分增益
        self.kd_force = np.array([0.01, 0.01, 0.01, 0.002, 0.002, 0.002])   # 力控制微分增益
        
        # 力限制
        self.max_force = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
        
        # 控制状态
        self.force_integral = np.zeros(6)
        self.force_error_prev = np.zeros(6)
        self.pose_prev = np.zeros(6)
        
        # 自适应控制参数
        self.insertion_phase = 'approach'  # 'approach', 'contact', 'insertion'
        self.contact_detected = False
        self.insertion_started = False
        
    def set_selection_strategy(self, strategy_name: str):
        """
        设置分配矩阵策略
        
        Args:
            strategy_name: 策略名称
        """
        if strategy_name in self.selection_matrices:
            self.current_strategy = strategy_name
            self.S = self.selection_matrices[strategy_name].copy()
            print(f"切换到分配策略: {strategy_name}")
            print(f"分配矩阵对角线: {np.diag(self.S)}")
            
            # 重置控制状态
            self.force_integral = np.zeros(6)
            self.force_error_prev = np.zeros(6)
        else:
            raise ValueError(f"未知策略: {strategy_name}. 可用策略: {list(self.selection_matrices.keys())}")
    
    def get_current_state(self) -> Dict:
        """获取当前环境状态信息"""
        obs = self.env.unwrapped._get_obs()
        
        # Meta-World peg-insert-side环境的观测空间分析
        # obs通常包含: [gripper_pos(3), gripper_state(1), object_pos(3), goal_pos(3), ...]
        eef_pos = obs[:3]  # 末端执行器位置
        gripper_state = obs[3]  # 夹爪状态
        
        # 获取peg位置 (被抓取物体的位置)
        if len(obs) > 4:
            peg_pos = obs[4:7] if len(obs) > 6 else eef_pos
        else:
            peg_pos = eef_pos
            
        # 获取目标孔洞位置
        if len(obs) > 7:
            hole_pos = obs[7:10] if len(obs) > 9 else np.array([0.1, 0.6, 0.15])
        else:
            hole_pos = np.array([0.1, 0.6, 0.15])  # 默认孔洞位置
        
        # 简化姿态处理
        eef_ori = np.array([0., 0., 0.])
        
        return {
            'eef_pos': eef_pos,
            'eef_ori': eef_ori,
            'eef_pose': np.concatenate([eef_pos, eef_ori]),
            'peg_pos': peg_pos,
            'hole_pos': hole_pos,
            'gripper_state': gripper_state,
            'full_obs': obs
        }
    
    def estimate_contact_forces(self, current_state: Dict) -> np.ndarray:
        """
        估算接触力 (由于Meta-World不直接提供力传感器，使用启发式方法)
        """
        eef_pos = current_state['eef_pos']
        hole_pos = current_state['hole_pos']
        
        # 计算到孔洞的距离
        distance_to_hole = np.linalg.norm(eef_pos - hole_pos)
        
        # 简化的接触力估算
        contact_forces = np.zeros(6)
        
        # 如果距离很近，假设存在接触力
        if distance_to_hole < 0.05:  # 5cm内认为可能接触
            # 基于距离和相对位置估算接触力
            direction_to_hole = (hole_pos - eef_pos) / (distance_to_hole + 1e-6)
            
            # 模拟接触阻力
            if distance_to_hole < 0.02:  # 2cm内认为强接触
                contact_forces[:3] = -direction_to_hole * (0.02 - distance_to_hole) * 50.0
                self.contact_detected = True
            else:
                contact_forces[:3] = -direction_to_hole * (0.05 - distance_to_hole) * 10.0
                self.contact_detected = False
        else:
            self.contact_detected = False
        
        return contact_forces
    
    def update_adaptive_strategy(self, current_state: Dict, contact_forces: np.ndarray):
        """更新自适应控制策略"""
        if self.current_strategy != 'adaptive':
            return
            
        eef_pos = current_state['eef_pos']
        hole_pos = current_state['hole_pos']
        distance_to_hole = np.linalg.norm(eef_pos - hole_pos)
        
        # 阶段判断
        if distance_to_hole > 0.05:
            self.insertion_phase = 'approach'
            # 接近阶段：全位置控制
            self.S = np.diag([0, 0, 0, 0, 0, 0])
        elif distance_to_hole > 0.02 and not self.contact_detected:
            self.insertion_phase = 'pre_contact'
            # 预接触阶段：X方向(插入方向)力控制
            self.S = np.diag([1, 0, 0, 0, 0, 0])
        elif self.contact_detected:
            self.insertion_phase = 'insertion'
            # 插入阶段：X,Y方向力控制，Z方向位置控制保持高度
            self.S = np.diag([1, 1, 0, 0, 0, 0])
    
    def compute_desired_trajectory(self, current_state: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算期望轨迹和力
        
        Returns:
            desired_pose: 期望位姿 [x, y, z, rx, ry, rz]
            desired_force: 期望力/力矩 [fx, fy, fz, mx, my, mz]
        """
        eef_pos = current_state['eef_pos']
        hole_pos = current_state['hole_pos']
        
        # 期望位置：朝向孔洞
        desired_pos = hole_pos.copy()
        desired_ori = np.array([0., 0., 0.])  # 保持水平姿态
        desired_pose = np.concatenate([desired_pos, desired_ori])
        
        # 期望力设计
        desired_force = np.zeros(6)
        
        distance_to_hole = np.linalg.norm(eef_pos - hole_pos)
        
        if self.current_strategy == 'adaptive':
            # 自适应力设计
            if self.insertion_phase == 'approach':
                desired_force[:3] = np.array([0., 0., 0.])
            elif self.insertion_phase == 'pre_contact':
                desired_force[:3] = np.array([2.0, 0., 0.])  # X方向插入力
            elif self.insertion_phase == 'insertion':
                desired_force[:3] = np.array([3.0, 0., -1.0])  # 插入力+向下压力
        else:
            # 基于当前策略的力设计
            if distance_to_hole < 0.05:
                if 'x' in self.current_strategy:
                    desired_force[0] = 2.0  # X方向插入力
                if 'y' in self.current_strategy:
                    desired_force[1] = 0.0  # Y方向保持0力避免偏移
                if 'z' in self.current_strategy:
                    desired_force[2] = -1.0  # Z方向轻微下压
        
        return desired_pose, desired_force
    
    def compute_hybrid_control(self, desired_pose: np.ndarray, 
                             desired_force: np.ndarray,
                             current_state: Dict) -> np.ndarray:
        """
        计算力位混合控制输出
        """
        current_pose = current_state['eef_pose']
        current_force = self.estimate_contact_forces(current_state)
        
        # 更新自适应策略
        self.update_adaptive_strategy(current_state, current_force)
        
        # 位置误差和速度
        pose_error = desired_pose - current_pose
        pose_velocity = (current_pose - self.pose_prev) / self.dt
        self.pose_prev = current_pose.copy()
        
        # 力误差和积分、微分项
        force_error = desired_force - current_force
        self.force_integral += force_error * self.dt
        force_derivative = (force_error - self.force_error_prev) / self.dt
        self.force_error_prev = force_error.copy()
        
        # 积分饱和限制
        self.force_integral = np.clip(self.force_integral, -10.0, 10.0)
        
        # 位置控制输出 (PD控制)
        u_pos = self.kp_pos * pose_error - self.kd_pos * pose_velocity
        
        # 力控制输出 (PID控制)
        u_force = (self.kp_force * force_error + 
                  self.ki_force * self.force_integral + 
                  self.kd_force * force_derivative)
        
        # 力控制限制
        u_force = np.clip(u_force, -self.max_force, self.max_force)
        
        # 力位混合控制: u = (I - S)u_pos + Su_force
        u_hybrid = (np.eye(6) - self.S) @ u_pos + self.S @ u_force
        
        # 转换为Meta-World动作空间 [delta_x, delta_y, delta_z, gripper]
        action = np.zeros(self.action_space.shape[0])
        
        # 位置增量控制 (限制单步移动距离)
        max_delta = 0.05  # 5cm最大单步移动
        action[:3] = np.clip(u_hybrid[:3] * self.dt, -max_delta, max_delta)
        
        # 夹爪控制：保持抓取
        action[3] = 1.0
        
        return action, u_hybrid, current_force
    
    def check_success(self, current_state: Dict) -> bool:
        """检查是否成功插入"""
        eef_pos = current_state['eef_pos']
        hole_pos = current_state['hole_pos']
        
        # 计算插入深度 (X方向)
        insertion_depth = hole_pos[0] - eef_pos[0]
        
        # 检查是否成功插入
        if insertion_depth < self.insertion_threshold and np.linalg.norm(eef_pos[1:] - hole_pos[1:]) < 0.02:
            return True
        return False
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置控制状态
        self.force_integral = np.zeros(6)
        self.force_error_prev = np.zeros(6)
        self.pose_prev = np.zeros(6)
        self.current_step = 0
        self.contact_detected = False
        self.insertion_started = False
        self.insertion_phase = 'approach'
        
        # 重置数据记录
        self.reset_data_logging()
        
        return obs, info
    
    def step_with_hybrid_control(self):
        """
        使用力位混合控制执行一步
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 获取当前状态
        current_state = self.get_current_state()
        
        # 计算期望轨迹
        desired_pose, desired_force = self.compute_desired_trajectory(current_state)
        
        # 计算混合控制动作
        action, control_output, contact_force = self.compute_hybrid_control(
            desired_pose, desired_force, current_state
        )
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查成功条件
        success = self.check_success(current_state)
        if success:
            reward += 100.0  # 成功奖励
            terminated = True
        
        # 步数限制
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # 记录数据
        if self.record_data:
            self.data_log['time'].append(self.current_step * self.dt)
            self.data_log['peg_pos'].append(current_state['peg_pos'].copy())
            self.data_log['hole_pos'].append(current_state['hole_pos'].copy())
            self.data_log['eef_pos'].append(current_state['eef_pos'].copy())
            self.data_log['contact_force'].append(contact_force.copy())
            self.data_log['desired_pose'].append(desired_pose.copy())
            self.data_log['desired_force'].append(desired_force.copy())
            self.data_log['control_action'].append(action.copy())
            self.data_log['reward'].append(reward)
            self.data_log['insertion_depth'].append(
                current_state['hole_pos'][0] - current_state['eef_pos'][0]
            )
            self.data_log['success'].append(success)
            self.data_log['strategy'].append(self.current_strategy)
        
        # 添加控制相关信息
        info.update({
            'control_strategy': self.current_strategy,
            'selection_matrix': self.S.copy(),
            'current_pose': current_state['eef_pose'],
            'contact_force': contact_force,
            'insertion_phase': self.insertion_phase,
            'contact_detected': self.contact_detected,
            'success': success,
            'insertion_depth': current_state['hole_pos'][0] - current_state['eef_pos'][0]
        })
        
        return obs, reward, terminated, truncated, info
    
    def run_episode(self, max_steps: int = None) -> Dict:
        """
        运行一个完整的episode
        
        Returns:
            episode_info: 包含episode统计信息的字典
        """
        if max_steps is None:
            max_steps = self.max_episode_steps
            
        obs, info = self.reset()
        total_reward = 0
        success = False
        
        for step in range(max_steps):
            obs, reward, terminated, truncated, info = self.step_with_hybrid_control()
            total_reward += reward
            
            if info['success']:
                success = True
                break
                
            if terminated or truncated:
                break
        
        episode_info = {
            'total_reward': total_reward,
            'success': success,
            'steps': step + 1,
            'strategy': self.current_strategy,
            'final_insertion_depth': info.get('insertion_depth', 0)
        }
        
        return episode_info
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def get_data_log(self) -> Dict:
        """获取记录的数据"""
        return self.data_log if self.record_data else None


def compare_strategies(strategies: List[str], episodes_per_strategy: int = 5, 
                      render: bool = False) -> Dict:
    """
    比较不同策略的性能
    
    Args:
        strategies: 要比较的策略列表
        episodes_per_strategy: 每个策略运行的episode数
        render: 是否渲染
        
    Returns:
        comparison_results: 比较结果
    """
    print("开始策略比较实验...")
    
    env = PegInsertSideHybridEnv(
        render_mode="human" if render else None,
        record_data=True
    )
    
    results = {}
    
    for strategy in strategies:
        print(f"\n测试策略: {strategy}")
        env.set_selection_strategy(strategy)
        
        strategy_results = {
            'episodes': [],
            'success_rate': 0,
            'avg_reward': 0,
            'avg_steps': 0,
            'avg_insertion_depth': 0
        }
        
        for episode in range(episodes_per_strategy):
            print(f"  Episode {episode + 1}/{episodes_per_strategy}")
            episode_info = env.run_episode()
            strategy_results['episodes'].append(episode_info)
            
            print(f"    Success: {episode_info['success']}, "
                  f"Reward: {episode_info['total_reward']:.2f}, "
                  f"Steps: {episode_info['steps']}")
        
        # 计算统计信息
        episodes = strategy_results['episodes']
        strategy_results['success_rate'] = sum(ep['success'] for ep in episodes) / len(episodes)
        strategy_results['avg_reward'] = np.mean([ep['total_reward'] for ep in episodes])
        strategy_results['avg_steps'] = np.mean([ep['steps'] for ep in episodes])
        strategy_results['avg_insertion_depth'] = np.mean([ep['final_insertion_depth'] for ep in episodes])
        
        results[strategy] = strategy_results
        
        print(f"  策略 {strategy} 结果:")
        print(f"    成功率: {strategy_results['success_rate']:.2%}")
        print(f"    平均奖励: {strategy_results['avg_reward']:.2f}")
        print(f"    平均步数: {strategy_results['avg_steps']:.1f}")
        print(f"    平均插入深度: {strategy_results['avg_insertion_depth']:.4f}m")
    
    env.close()
    return results


def visualize_results(results: Dict, save_plot: bool = True):
    """可视化比较结果"""
    strategies = list(results.keys())
    success_rates = [results[s]['success_rate'] for s in strategies]
    avg_rewards = [results[s]['avg_reward'] for s in strategies]
    avg_steps = [results[s]['avg_steps'] for s in strategies]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 成功率对比
    axes[0,0].bar(strategies, success_rates)
    axes[0,0].set_title('Success Rate by Strategy')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 平均奖励对比
    axes[0,1].bar(strategies, avg_rewards)
    axes[0,1].set_title('Average Reward by Strategy')
    axes[0,1].set_ylabel('Average Reward')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 平均步数对比
    axes[1,0].bar(strategies, avg_steps)
    axes[1,0].set_title('Average Steps by Strategy')
    axes[1,0].set_ylabel('Average Steps')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 成功率vs平均奖励散点图
    axes[1,1].scatter(success_rates, avg_rewards)
    for i, strategy in enumerate(strategies):
        axes[1,1].annotate(strategy, (success_rates[i], avg_rewards[i]))
    axes[1,1].set_xlabel('Success Rate')
    axes[1,1].set_ylabel('Average Reward')
    axes[1,1].set_title('Success Rate vs Average Reward')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
        print("结果图已保存为 'strategy_comparison.png'")
    else:
        plt.show()


def demo_single_strategy(strategy_name: str = 'force_x', render: bool = True, 
                        max_steps: int = 200):
    """
    演示单个策略的运行效果
    
    Args:
        strategy_name: 策略名称
        render: 是否渲染环境
        max_steps: 最大步数
    """
    print(f"演示策略: {strategy_name}")
    print("=" * 50)
    
    env = PegInsertSideHybridEnv(
        render_mode="human" if render else None,
        record_data=True
    )
    
    try:
        env.set_selection_strategy(strategy_name)
        
        obs, info = env.reset()
        print(f"环境已重置")
        print(f"观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        total_reward = 0
        success = False
        
        for step in range(max_steps):
            # 渲染环境
            if render:
                env.render()
                time.sleep(0.05)  # 控制渲染速度
            
            # 执行一步
            obs, reward, terminated, truncated, info = env.step_with_hybrid_control()
            total_reward += reward
            
            # 打印信息
            if step % 20 == 0:
                print(f"Step {step:3d}: Reward={reward:6.2f}, "
                      f"Phase={info['insertion_phase']:12s}, "
                      f"Contact={info['contact_detected']}, "
                      f"Depth={info['insertion_depth']:6.3f}m")
            
            # 检查结束条件
            if info['success']:
                success = True
                print(f"\n✓ 成功插入! 用时 {step+1} 步")
                break
                
            if terminated or truncated:
                print(f"\n× 任务结束 (terminated={terminated}, truncated={truncated})")
                break
        
        # 输出结果
        print(f"\n策略 '{strategy_name}' 结果:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  成功: {'是' if success else '否'}")
        print(f"  步数: {step+1}")
        print(f"  最终插入深度: {info['insertion_depth']:.4f}m")
        
        # 获取数据日志
        data_log = env.get_data_log()
        if data_log and len(data_log['time']) > 0:
            print(f"  记录了 {len(data_log['time'])} 个数据点")
            
            # 保存一些关键数据
            final_pos = data_log['eef_pos'][-1] if data_log['eef_pos'] else [0,0,0]
            final_hole_pos = data_log['hole_pos'][-1] if data_log['hole_pos'] else [0,0,0]
            print(f"  最终末端位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}]")
            print(f"  目标孔洞位置: [{final_hole_pos[0]:.3f}, {final_hole_pos[1]:.3f}, {final_hole_pos[2]:.3f}]")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("环境已关闭")


def main():
    """主函数 - 可以选择演示或比较实验"""
    
    print("Meta-World 力位混合控制实验")
    print("=" * 50)
    
    # 选择运行模式
    mode = input("选择运行模式:\n1. 演示单个策略 (输入1)\n2. 策略比较实验 (输入2)\n请选择 (默认1): ").strip() or "1"
    
    if mode == "1":
        # 演示模式
        available_strategies = [
            'position_only', 'force_x', 'force_y', 'force_z', 
            'force_xy', 'force_xz', 'force_yz', 'force_xyz', 'adaptive'
        ]
        
        print(f"\n可用策略: {', '.join(available_strategies)}")
        strategy = input("选择策略 (默认 force_x): ").strip() or "force_x"
        
        if strategy not in available_strategies:
            print(f"未知策略 '{strategy}', 使用默认策略 'force_x'")
            strategy = "force_x"
        
        # 运行演示
        demo_single_strategy(strategy, render=True, max_steps=300)
        
    elif mode == "2":
        # 比较实验模式
        print("\n运行策略比较实验...")
        
        # 定义要比较的策略
        strategies_to_compare = [
            'position_only',    # 基线
            'force_x',         # X方向力控制
            'force_xy',        # X,Y方向力控制
            'adaptive'         # 自适应策略
        ]
        
        # 运行比较实验
        results = compare_strategies(
            strategies=strategies_to_compare,
            episodes_per_strategy=2,  # 每个策略2个episode
            render=False  # 比较实验不渲染以提高速度
        )
        
        # 打印总结
        print("\n" + "="*60)
        print("实验总结:")
        print("="*60)
        
        for strategy, result in results.items():
            print(f"{strategy:15} | 成功率: {result['success_rate']:6.1%} | "
                  f"平均奖励: {result['avg_reward']:8.2f} | "
                  f"平均步数: {result['avg_steps']:6.1f}")
        
        # 可视化结果
        try:
            visualize_results(results, save_plot=True)
        except Exception as e:
            print(f"可视化失败: {e}")
        
        print("\n实验完成!")
    
    else:
        print("无效选择，退出程序")


if __name__ == "__main__":
    main()