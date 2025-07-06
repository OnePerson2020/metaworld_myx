import numpy as np
import mujoco
import gymnasium as gym
import metaworld
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

@dataclass
class ControlParams:
    """力位混合控制参数"""
    # 位置控制参数
    kp_pos: float = 1000.0  # 位置刚度
    kd_pos: float = 100.0   # 位置阻尼
    
    # 力控制参数
    kp_force: float = 0.001  # 力增益
    ki_force: float = 0.0001 # 力积分增益
    force_deadzone: float = 0.5  # 力死区(N)
    max_force: float = 10.0     # 最大允许力(N)
    
    # 控制频率
    position_control_freq: float = 500.0  # Hz
    force_control_freq: float = 1000.0    # Hz
    
    # 切换阈值
    switch_distance: float = 0.05  # 距离目标多远时切换到力控制(m)

class TaskCoordinateSystem:
    """任务坐标系：以hole为基准建立坐标系"""
    
    def __init__(self, hole_pos: np.ndarray, hole_orientation: np.ndarray):
        self.hole_pos = hole_pos.copy()
        self.hole_orientation = hole_orientation.copy()
        
        # 构建任务坐标系：Z轴为插入方向（指向hole内部）
        self.z_axis = self.hole_orientation / np.linalg.norm(self.hole_orientation)
        
        # 构建正交的X、Y轴
        if abs(self.z_axis[2]) < 0.9:
            self.x_axis = np.cross(self.z_axis, [0, 0, 1])
        else:
            self.x_axis = np.cross(self.z_axis, [1, 0, 0])
        if np.linalg.norm(self.x_axis) < 1e-6:  # 避免零向量
            self.x_axis = np.array([1, 0, 0])
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
        self.y_axis = np.cross(self.z_axis, self.x_axis)
        
        # 旋转矩阵：世界坐标系 -> 任务坐标系
        self.rotation_matrix = np.column_stack([self.x_axis, self.y_axis, self.z_axis])
        
    def world_to_task(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标转任务坐标"""
        relative_pos = world_pos - self.hole_pos
        return self.rotation_matrix.T @ relative_pos
    
    def task_to_world(self, task_pos: np.ndarray) -> np.ndarray:
        """任务坐标转世界坐标"""
        world_relative = self.rotation_matrix @ task_pos
        return world_relative + self.hole_pos
    
    def world_force_to_task(self, world_force: np.ndarray) -> np.ndarray:
        """世界坐标系下的力转换到任务坐标系"""
        return self.rotation_matrix.T @ world_force
    
    def task_force_to_world(self, task_force: np.ndarray) -> np.ndarray:
        """任务坐标系下的力转换到世界坐标系"""
        return self.rotation_matrix @ task_force

class ForcePositionController:
    """力位混合控制器"""
    
    def __init__(self, params: ControlParams):
        self.params = params
        self.force_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"  # "position" 或 "hybrid"
        
    def reset(self):
        """重置控制器状态"""
        self.force_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_force: np.ndarray,
                       target_force: np.ndarray,
                       selection_matrix: np.ndarray,
                       dt: float) -> np.ndarray:
        """
        计算力位混合控制输出
        
        Args:
            current_pos: 当前位置 (任务坐标系)
            target_pos: 目标位置 (任务坐标系)
            current_force: 当前接触力 (任务坐标系)
            target_force: 目标接触力 (任务坐标系)
            selection_matrix: 方向选择矩阵 (3x3, 对角线元素: 1=位置控制, 0=力控制)
            dt: 时间步长
            
        Returns:
            控制输出 (任务坐标系下的位置增量)
        """
        
        # 位置误差
        pos_error = target_pos - current_pos
        
        # 力误差
        force_error = target_force - current_force
        
        # 力积分（仅在力控制方向）
        force_mask = 1 - np.diag(selection_matrix)  # 力控制方向的掩码
        self.force_integral += force_error * force_mask * dt
        
        # 限制积分器
        self.force_integral = np.clip(self.force_integral, -1.0, 1.0)
        
        # 位置控制输出
        pos_control = self.params.kp_pos * pos_error
        
        # 力控制输出（转换为位置增量）
        force_control = (self.params.kp_force * force_error + 
                        self.params.ki_force * self.force_integral)
        
        # 应用死区
        force_control = np.where(np.abs(current_force) > self.params.force_deadzone,
                                force_control, 0)
        
        # 混合控制输出
        output = (selection_matrix @ pos_control + 
                 (np.eye(3) - selection_matrix) @ force_control)
        
        return output

class ForceExtractor:
    """力信息提取器"""
    
    def __init__(self, env):
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        # 查找peg相关的几何体ID
        self.peg_geom_ids = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'peg' in geom_name.lower():
                self.peg_geom_ids.append(i)
        
        print(f"找到peg几何体ID: {self.peg_geom_ids}")
    
    def get_contact_forces(self) -> np.ndarray:
        """获取peg上的接触力"""
        total_force = np.zeros(3)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 检查是否涉及peg
            if contact.geom1 in self.peg_geom_ids or contact.geom2 in self.peg_geom_ids:
                # 获取接触力
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                
                # 接触力的前3个分量是法向力和切向力
                contact_force = c_array[:3]
                
                # 转换到世界坐标系
                contact_frame = contact.frame.reshape(3, 3)
                world_force = contact_frame @ contact_force
                
                total_force += world_force
        
        return total_force

class HybridControlWrapper(gym.Wrapper):
    """力位混合控制包装器"""
    
    def __init__(self, env, control_params: ControlParams = None):
        super().__init__(env)
        self.params = control_params or ControlParams()
        self.force_extractor = ForceExtractor(env)
        self.controller = ForcePositionController(self.params)
        
        # 任务坐标系（将在reset时初始化）
        self.task_coord_system = None
        
        # 控制状态
        self.peg_grasped = False
        self.control_phase = "approach"  # "approach", "contact", "insert"
        
        # 记录数据
        self.episode_data = {
            'forces': [],
            'positions': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'box_positions': []  # 用于检测box移动
        }
        
        # 初始box位置
        self.initial_box_pos = None
        
    def reset(self, **kwargs):
        """重置环境并初始化任务坐标系"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置控制器
        self.controller.reset()
        self.peg_grasped = False
        self.control_phase = "approach"
        
        # 获取hole位置和方向
        hole_pos, hole_orientation = self._get_hole_info()
        self.task_coord_system = TaskCoordinateSystem(hole_pos, hole_orientation)
        
        # 记录初始box位置
        self.initial_box_pos = self._get_box_position()
        
        # 重置记录数据
        self.episode_data = {
            'forces': [],
            'positions': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'box_positions': []
        }
        
        print(f"任务坐标系初始化完成")
        print(f"Hole位置: {hole_pos}")
        print(f"插入方向: {hole_orientation}")
        
        return obs, info
    
    def _get_hole_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取hole的位置和方向"""
        # 获取box的位置和方向
        box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                       mujoco.mjtObj.mjOBJ_BODY, 'box')
        box_pos = self.env.unwrapped.data.xpos[box_body_id].copy()
        box_quat = self.env.unwrapped.data.xquat[box_body_id].copy()
        
        # hole在box上的相对位置（根据具体模型调整）
        hole_relative_pos = np.array([0.03, 0.0, 0.13])  # 相对于box center
        
        # 转换到世界坐标系
        rotation = Rotation.from_quat(box_quat)
        hole_world_pos = box_pos + rotation.apply(hole_relative_pos)
        
        # hole的方向（插入方向）
        hole_direction = rotation.apply(np.array([-1, 0, 0]))  # X负方向为插入方向
        
        return hole_world_pos, hole_direction
    
    def _get_box_position(self) -> np.ndarray:
        """获取box的当前位置"""
        box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                       mujoco.mjtObj.mjOBJ_BODY, 'box')
        return self.env.unwrapped.data.xpos[box_body_id].copy()
    
    def _get_peg_position(self) -> np.ndarray:
        """获取peg的当前位置"""
        return self.env.unwrapped._get_pos_objects()
    
    def _detect_grasp(self) -> bool:
        """检测是否抓取了peg"""
        # 简单的抓取检测：检查夹爪是否关闭且接近peg
        gripper_distance = self.env.unwrapped._get_gripper_distance()
        peg_pos = self._get_peg_position()
        hand_pos = self.env.unwrapped._get_endeff_pos()
        
        distance_to_peg = np.linalg.norm(hand_pos - peg_pos)
        
        return gripper_distance < 0.3 and distance_to_peg < 0.05
    
    def _update_control_phase(self):
        """更新控制阶段"""
        peg_pos = self._get_peg_position()
        hole_pos, _ = self._get_hole_info()
        distance_to_hole = np.linalg.norm(peg_pos - hole_pos)
        
        if not self.peg_grasped:
            self.control_phase = "approach"
        elif distance_to_hole > self.params.switch_distance:
            self.control_phase = "contact"
        else:
            self.control_phase = "insert"
    
    def step(self, action):
        """执行一步，应用力位混合控制"""
        # 检测抓取状态
        self.peg_grasped = self._detect_grasp()
        
        # 更新控制阶段
        self._update_control_phase()
        
        # 获取当前状态
        peg_pos = self._get_peg_position()
        hole_pos, hole_orientation = self._get_hole_info()
        contact_force = self.force_extractor.get_contact_forces()
        
        # 转换到任务坐标系
        peg_pos_task = self.task_coord_system.world_to_task(peg_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        
        # 根据控制阶段选择控制策略
        if self.control_phase == "approach":
            # 纯位置控制阶段
            modified_action = action.copy()
        elif self.control_phase in ["contact", "insert"]:
            # 力位混合控制阶段
            modified_action = self._apply_hybrid_control(
                action, peg_pos_task, hole_pos_task, contact_force_task)
        else:
            modified_action = action.copy()
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 记录数据
        self._record_data(peg_pos, contact_force, modified_action)
        
        # 添加额外的评估指标
        info.update(self._compute_evaluation_metrics())
        
        return obs, reward, terminated, truncated, info
    
    def _apply_hybrid_control(self, action, peg_pos_task, hole_pos_task, contact_force_task):
        """应用力位混合控制"""
        # 定义目标位置（在hole内部一定深度）
        target_depth = 0.08  # 插入深度8cm
        target_pos_task = hole_pos_task + np.array([0, 0, -target_depth])
        
        # 定义目标力（在非插入方向保持小的接触力）
        target_force_task = np.array([0, 0, 5.0])  # Z方向(插入方向)允许一定的力
        
        # 定义方向选择矩阵
        selection_matrix = self._get_selection_matrix()
        
        # 计算控制输出
        dt = 1.0 / self.params.force_control_freq
        control_output_task = self.controller.compute_control(
            peg_pos_task, target_pos_task, contact_force_task, 
            target_force_task, selection_matrix, dt)
        
        # 转换回世界坐标系
        control_output_world = self.task_coord_system.task_force_to_world(control_output_task)
        
        # 修改原始action
        modified_action = action.copy()
        
        # 将控制输出应用到位置控制
        # 这里需要根据具体的action格式调整
        scale_factor = 0.01  # 根据需要调整
        modified_action[:3] = control_output_world * scale_factor
        
        return modified_action
    
    def _get_selection_matrix(self) -> np.ndarray:
        """获取方向选择矩阵"""
        # 这里可以实现不同的选择策略
        # 示例：Z方向（插入方向）做位置控制，XY方向做力控制
        selection_matrix = np.diag([0, 0, 1])  # Z方向位置控制，XY方向力控制
        return selection_matrix
    
    def _record_data(self, peg_pos, contact_force, control_output):
        """记录实验数据"""
        self.episode_data['positions'].append(peg_pos.copy())
        self.episode_data['contact_forces'].append(contact_force.copy())
        self.episode_data['control_outputs'].append(control_output.copy())
        self.episode_data['phases'].append(self.control_phase)
        self.episode_data['box_positions'].append(self._get_box_position())
    
    def _compute_evaluation_metrics(self) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 计算最大接触力
        if self.episode_data['contact_forces']:
            forces = np.array(self.episode_data['contact_forces'])
            metrics['max_contact_force'] = np.max(np.linalg.norm(forces, axis=1))
            metrics['avg_contact_force'] = np.mean(np.linalg.norm(forces, axis=1))
        
        # 计算box移动距离（环境破坏程度）
        if self.initial_box_pos is not None:
            current_box_pos = self._get_box_position()
            box_displacement = np.linalg.norm(current_box_pos - self.initial_box_pos)
            metrics['box_displacement'] = box_displacement
            metrics['environment_damage'] = box_displacement > 0.01  # 阈值可调
        
        # 计算插入深度
        peg_pos = self._get_peg_position()
        hole_pos, hole_orientation = self._get_hole_info()
        peg_pos_task = self.task_coord_system.world_to_task(peg_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        
        insertion_depth = max(0, hole_pos_task[2] - peg_pos_task[2])
        metrics['insertion_depth'] = insertion_depth
        metrics['insertion_success'] = insertion_depth > 0.05  # 插入深度阈值
        
        return metrics

class SelectionMatrixExperiment:
    """方向选择矩阵实验类"""
    
    def __init__(self, env_name: str = 'peg-insert-side-v3'):
        self.env_name = env_name
        self.base_env = None
        self.results = []
    
    def create_env(self, selection_matrix_func=None):
        """创建带有指定选择矩阵的环境"""
        # 创建基础环境
        base_env = metaworld.make_mt_envs(self.env_name, render_mode='rgb_array')
        
        # 包装为混合控制环境
        hybrid_env = HybridControlWrapper(base_env)
        
        # 如果提供了选择矩阵函数，替换默认的
        if selection_matrix_func:
            hybrid_env._get_selection_matrix = selection_matrix_func
        
        return hybrid_env
    
    def define_selection_matrices(self):
        """定义不同的方向选择矩阵配置"""
        matrices = {
            "纯位置控制": lambda: np.eye(3),  # 所有方向位置控制
            "纯力控制": lambda: np.zeros((3, 3)),  # 所有方向力控制
            "Z位置XY力": lambda: np.diag([0, 0, 1]),  # 插入方向位置，其他方向力
            "X位置YZ力": lambda: np.diag([1, 0, 0]),  # X方向位置，其他方向力
            "XZ位置Y力": lambda: np.diag([1, 0, 1]),  # XZ方向位置，Y方向力
            "带偏差Z位置": lambda: self._create_biased_matrix([0, 0, 1], 0.1),
            "带偏差X位置": lambda: self._create_biased_matrix([1, 0, 0], 0.1),
        }
        return matrices
    
    def _create_biased_matrix(self, main_direction, bias_strength):
        """创建带有偏差的选择矩阵"""
        matrix = np.diag(main_direction)
        # 添加小的非对角元素作为偏差
        if bias_strength > 0:
            noise = np.random.uniform(-bias_strength, bias_strength, (3, 3))
            noise = (noise + noise.T) / 2  # 确保对称性
            matrix = matrix + noise
            # 确保对角线元素在合理范围内
            np.fill_diagonal(matrix, np.clip(np.diag(matrix), 0, 1))
        return matrix
    
    def run_experiment(self, num_episodes: int = 5, max_steps: int = 500):
        """运行选择矩阵对比实验"""
        matrices = self.define_selection_matrices()
        
        print(f"开始方向选择矩阵实验，测试{len(matrices)}种配置...")
        
        for matrix_name, matrix_func in matrices.items():
            print(f"\n测试配置: {matrix_name}")
            
            # 创建环境
            env = self.create_env(matrix_func)
            
            episode_results = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_data = {
                    'matrix_name': matrix_name,
                    'episode': episode,
                    'total_reward': 0,
                    'success': False,
                    'metrics': {}
                }
                
                for step in range(max_steps):
                    # 使用专家策略获取基础动作
                    from metaworld.policies import SawyerPegInsertionSideV3Policy
                    policy = SawyerPegInsertionSideV3Policy()
                    action = policy.get_action(obs)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_data['total_reward'] += reward
                    
                    if terminated or truncated:
                        break
                
                # 记录最终指标
                episode_data['success'] = info.get('insertion_success', False)
                episode_data['metrics'] = {
                    'max_contact_force': info.get('max_contact_force', 0),
                    'box_displacement': info.get('box_displacement', 0),
                    'insertion_depth': info.get('insertion_depth', 0),
                    'environment_damage': info.get('environment_damage', False)
                }
                
                episode_results.append(episode_data)
                print(f"  Episode {episode}: Success={episode_data['success']}, "
                      f"Reward={episode_data['total_reward']:.2f}")
            
            # 计算平均结果
            avg_results = self._compute_average_results(episode_results)
            self.results.append(avg_results)
            
            env.close()
    
    def _compute_average_results(self, episode_results):
        """计算平均结果"""
        if not episode_results:
            return {}
        
        avg_result = {
            'matrix_name': episode_results[0]['matrix_name'],
            'num_episodes': len(episode_results),
            'success_rate': np.mean([r['success'] for r in episode_results]),
            'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
            'avg_max_contact_force': np.mean([r['metrics']['max_contact_force'] for r in episode_results]),
            'avg_box_displacement': np.mean([r['metrics']['box_displacement'] for r in episode_results]),
            'avg_insertion_depth': np.mean([r['metrics']['insertion_depth'] for r in episode_results]),
            'damage_rate': np.mean([r['metrics']['environment_damage'] for r in episode_results])
        }
        
        return avg_result
    
    def print_results(self):
        """打印实验结果"""
        print("\n" + "="*80)
        print("方向选择矩阵实验结果汇总")
        print("="*80)
        
        # 表头
        header = f"{'配置名称':<20} {'成功率':<8} {'平均奖励':<10} {'最大接触力':<12} {'Box位移':<10} {'插入深度':<10} {'破坏率':<8}"
        print(header)
        print("-"*80)
        
        # 数据行
        for result in self.results:
            row = (f"{result['matrix_name']:<20} "
                   f"{result['success_rate']:<8.2f} "
                   f"{result['avg_reward']:<10.1f} "
                   f"{result['avg_max_contact_force']:<12.1f} "
                   f"{result['avg_box_displacement']:<10.4f} "
                   f"{result['avg_insertion_depth']:<10.3f} "
                   f"{result['damage_rate']:<8.2f}")
            print(row)
        
        print("-"*80)
        
        # 找出最优配置
        best_success = max(self.results, key=lambda x: x['success_rate'])
        best_damage = min(self.results, key=lambda x: x['damage_rate'])
        
        print(f"\n最高成功率: {best_success['matrix_name']} ({best_success['success_rate']:.2f})")
        print(f"最低破坏率: {best_damage['matrix_name']} ({best_damage['damage_rate']:.2f})")
    
    def plot_results(self):
        """绘制结果图表"""
        if not self.results:
            print("没有结果数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        matrix_names = [r['matrix_name'] for r in self.results]
        
        # 成功率
        success_rates = [r['success_rate'] for r in self.results]
        axes[0, 0].bar(matrix_names, success_rates)
        axes[0, 0].set_title('成功率对比')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 接触力
        contact_forces = [r['avg_max_contact_force'] for r in self.results]
        axes[0, 1].bar(matrix_names, contact_forces)
        axes[0, 1].set_title('最大接触力对比')
        axes[0, 1].set_ylabel('接触力 (N)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Box位移
        box_displacements = [r['avg_box_displacement'] for r in self.results]
        axes[1, 0].bar(matrix_names, box_displacements)
        axes[1, 0].set_title('Box位移对比')
        axes[1, 0].set_ylabel('位移 (m)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 插入深度
        insertion_depths = [r['avg_insertion_depth'] for r in self.results]
        axes[1, 1].bar(matrix_names, insertion_depths)
        axes[1, 1].set_title('插入深度对比')
        axes[1, 1].set_ylabel('深度 (m)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def demo_hybrid_control():
    """演示力位混合控制"""
    print("开始力位混合控制演示...")
    
    # 创建实验环境
    experiment = SelectionMatrixExperiment()
    
    # 运行实验
    experiment.run_experiment(num_episodes=3, max_steps=200)
    
    # 显示结果
    experiment.print_results()
    experiment.plot_results()

if __name__ == "__main__":
    demo_hybrid_control()