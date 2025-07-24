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
    
    # 姿态控制参数
    kp_rot: float = 500.0   # 姿态刚度
    kd_rot: float = 50.0    # 姿态阻尼
    
    # 力控制参数
    kp_force: float = 0.001  # 力增益
    ki_force: float = 0.0001 # 力积分增益
    kp_torque: float = 0.0005 # 力矩增益
    ki_torque: float = 0.00005 # 力矩积分增益
    force_deadzone: float = 0.5  # 力死区(N)
    torque_deadzone: float = 0.1 # 力矩死区(Nm)
    max_force: float = 10.0     # 最大允许力(N)
    max_torque: float = 2.0     # 最大允许力矩(Nm)
    
    # 几何参数（基于XML分析）
    peg_radius: float = 0.015    # peg半径
    hole_radius: float = 0.025   # hole半径（估计）
    insertion_tolerance: float = 0.008  # 径向容差
    min_insertion_depth: float = 0.06   # 最小插入深度
    
    # 插入策略参数
    approach_distance: float = 0.08  # 接近距离
    alignment_distance: float = 0.03 # 对齐距离
    max_orientation_error: float = 0.2 # 最大姿态误差(弧度)
    
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
        
        # 构建任务坐标系：Y轴为插入方向（指向hole内部，即Y负方向）
        self.y_axis = -self.hole_orientation / np.linalg.norm(self.hole_orientation)
        
        # 构建正交的X、Z轴（XZ平面为径向平面）
        if abs(self.y_axis[2]) < 0.9:
            self.z_axis = np.cross(self.y_axis, [0, 0, 1])
        else:
            self.z_axis = np.cross(self.y_axis, [1, 0, 0])
        if np.linalg.norm(self.z_axis) < 1e-6:
            self.z_axis = np.array([0, 0, 1])
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.x_axis = np.cross(self.y_axis, self.z_axis)
        
        # 旋转矩阵：世界坐标系 -> 任务坐标系
        self.rotation_matrix = np.column_stack([self.x_axis, self.y_axis, self.z_axis])
        
        # 任务坐标系的目标姿态（peg应该对齐的方向）
        self.target_rotation = Rotation.from_matrix(self.rotation_matrix.T)
        
        print(f"任务坐标系建立：")
        print(f"  X轴（径向1）: {self.x_axis}")
        print(f"  Y轴（插入方向）: {self.y_axis}")
        print(f"  Z轴（径向2）: {self.z_axis}")
        
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
    
    def get_orientation_error(self, current_rotation: Rotation) -> np.ndarray:
        """计算当前姿态与目标姿态的误差"""
        # 计算相对旋转
        relative_rotation = self.target_rotation * current_rotation.inv()
        
        # 转换为轴角表示
        axis_angle = relative_rotation.as_rotvec()
        
        # 转换到任务坐标系
        return self.rotation_matrix.T @ axis_angle

class InsertionSuccessChecker:
    """插入成功检测器"""
    
    def __init__(self, params: ControlParams, task_coord_system: TaskCoordinateSystem):
        self.params = params
        self.task_coord_system = task_coord_system
        
    def check_insertion_success(self, peg_head_pos: np.ndarray, peg_pos: np.ndarray, 
                              peg_rotation: Rotation) -> Dict:
        """检查插入成功状态"""
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        peg_center_task = self.task_coord_system.world_to_task(peg_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 计算径向距离（在XZ平面上的距离）
        radial_distance = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        
        # 计算插入深度（Y方向，负值表示插入）
        insertion_depth = max(0, -peg_head_task[1])
        
        # 检查是否在hole内（径向约束）
        is_inside_hole = radial_distance <= self.params.insertion_tolerance
        
        # 检查是否达到最小插入深度
        sufficient_depth = insertion_depth >= self.params.min_insertion_depth
        
        # 检查peg是否大致对齐
        center_radial_distance = np.sqrt(peg_center_task[0]**2 + peg_center_task[2]**2)
        is_aligned = center_radial_distance <= self.params.insertion_tolerance * 2
        
        # 检查姿态是否对齐
        orientation_magnitude = np.linalg.norm(orientation_error)
        is_orientation_aligned = orientation_magnitude <= self.params.max_orientation_error
        
        success = is_inside_hole and sufficient_depth and is_aligned and is_orientation_aligned
        
        return {
            'success': success,
            'insertion_depth': insertion_depth,
            'radial_distance': radial_distance,
            'orientation_error': orientation_magnitude,
            'is_inside_hole': is_inside_hole,
            'sufficient_depth': sufficient_depth,
            'is_aligned': is_aligned,
            'is_orientation_aligned': is_orientation_aligned,
            'peg_head_task_pos': peg_head_task,
            'peg_center_task_pos': peg_center_task,
            'orientation_error_vec': orientation_error
        }

class Enhanced6DOFController:
    """增强的6DOF力位混合控制器"""
    
    def __init__(self, params: ControlParams):
        self.params = params
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"
        
    def reset(self):
        """重置控制器状态"""
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_orientation: np.ndarray,  # 姿态误差向量
                       target_orientation: np.ndarray,   # 目标姿态误差（通常为0）
                       current_force: np.ndarray,
                       target_force: np.ndarray,
                       current_torque: np.ndarray,
                       target_torque: np.ndarray,
                       selection_matrix_pos: np.ndarray,  # 位置选择矩阵
                       selection_matrix_rot: np.ndarray,  # 姿态选择矩阵
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算6DOF力位混合控制输出
        
        Returns:
            Tuple[位置控制输出, 姿态控制输出]
        """
        
        # 位置误差
        pos_error = target_pos - current_pos
        
        # 姿态误差
        rot_error = target_orientation - current_orientation
        
        # 力误差
        force_error = target_force - current_force
        torque_error = target_torque - current_torque
        
        # 力积分（仅在力控制方向）
        force_mask = 1 - np.diag(selection_matrix_pos)
        torque_mask = 1 - np.diag(selection_matrix_rot)
        
        self.force_integral += force_error * force_mask * dt
        self.torque_integral += torque_error * torque_mask * dt
        
        # 限制积分器防止积分饱和
        self.force_integral = np.clip(self.force_integral, -1.0, 1.0)
        self.torque_integral = np.clip(self.torque_integral, -0.5, 0.5)
        
        # 位置控制输出
        pos_control = self.params.kp_pos * pos_error
        rot_control = self.params.kp_rot * rot_error
        
        # 力控制输出（转换为位置/姿态增量）
        force_control = (self.params.kp_force * force_error + 
                        self.params.ki_force * self.force_integral)
        torque_control = (self.params.kp_torque * torque_error + 
                         self.params.ki_torque * self.torque_integral)
        
        # 应用死区
        force_control = np.where(np.abs(current_force) > self.params.force_deadzone,
                                force_control, 0)
        torque_control = np.where(np.abs(current_torque) > self.params.torque_deadzone,
                                 torque_control, 0)
        
        # 混合控制输出
        pos_output = (selection_matrix_pos @ pos_control + 
                     (np.eye(3) - selection_matrix_pos) @ force_control)
        rot_output = (selection_matrix_rot @ rot_control + 
                     (np.eye(3) - selection_matrix_rot) @ torque_control)
        
        return pos_output, rot_output

class ForceExtractor:
    """改进的力信息提取器"""
    
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
    
    def get_contact_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取peg上的接触力和力矩"""
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        contact_count = 0
        
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
                
                # 计算力矩（简化计算，假设力作用在接触点）
                contact_pos = contact.pos
                peg_pos = self.env.unwrapped._get_pos_objects()
                r = contact_pos - peg_pos
                contact_torque = np.cross(r, world_force)
                
                total_force += world_force
                total_torque += contact_torque
                contact_count += 1
        
        return total_force, total_torque

class HybridControlWrapper(gym.Wrapper):
    """改进的力位混合控制包装器"""
    
    def __init__(self, env, control_params: ControlParams = None):
        super().__init__(env)
        self.params = control_params or ControlParams()
        self.force_extractor = ForceExtractor(env)
        self.controller = Enhanced6DOFController(self.params)
        
        # 任务坐标系和成功检测器（将在reset时初始化）
        self.task_coord_system = None
        self.success_checker = None
        
        # 控制状态
        self.peg_grasped = False
        self.control_phase = "approach"  # "approach", "align", "insert"
        self.insertion_stage = "move_to_front"  # "move_to_front", "align_orientation", "insert"
        
        # 记录数据
        self.episode_data = {
            'forces': [],
            'torques': [],
            'positions': [],
            'orientations': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'insertion_status': [],
            'box_positions': []
        }
        
        # 初始状态
        self.initial_box_pos = None
        
    def reset(self, **kwargs):
        """重置环境并初始化坐标系"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置控制器
        self.controller.reset()
        self.peg_grasped = False
        self.control_phase = "approach"
        self.insertion_stage = "move_to_front"
        
        # 获取hole位置和方向
        hole_pos, hole_orientation = self._get_hole_info()
        self.task_coord_system = TaskCoordinateSystem(hole_pos, hole_orientation)
        self.success_checker = InsertionSuccessChecker(self.params, self.task_coord_system)
        
        # 记录初始状态
        self.initial_box_pos = self._get_box_position()
        
        # 重置记录数据
        self.episode_data = {
            'forces': [],
            'torques': [],
            'positions': [],
            'orientations': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'insertion_status': [],
            'box_positions': []
        }
        
        print(f"环境重置完成")
        print(f"Hole位置: {hole_pos}")
        print(f"插入方向: {hole_orientation}")
        
        return obs, info
    
    def _get_hole_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取hole的位置和方向"""
        try:
            # 直接使用hole site的位置
            hole_site_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_SITE, 'hole')
            hole_pos = self.env.unwrapped.data.site_xpos[hole_site_id].copy()
            
            # 获取box的方向来确定hole的开口方向
            box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_BODY, 'box')
            box_quat = self.env.unwrapped.data.xquat[box_body_id].copy()
            
            # 根据XML，hole的开口朝向Y轴正方向（在box坐标系中）
            rotation = Rotation.from_quat(box_quat)
            hole_direction = rotation.apply(np.array([0, 1, 0]))  # Y正方向为hole开口方向
            
        except:
            # 备用方案：使用目标位置估算
            print("Warning: 无法直接获取hole信息，使用估算值")
            hole_pos = self.env.unwrapped._target_pos.copy() if hasattr(self.env.unwrapped, '_target_pos') and self.env.unwrapped._target_pos is not None else np.array([-0.3, 0.504, 0.13])
            hole_direction = np.array([0, 1, 0])
        
        return hole_pos, hole_direction
    
    def _get_box_position(self) -> np.ndarray:
        """获取box的当前位置"""
        try:
            box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_BODY, 'box')
            return self.env.unwrapped.data.xpos[box_body_id].copy()
        except:
            return np.zeros(3)
    
    def _get_peg_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取peg的中心位置和头部位置"""
        # peg中心位置
        peg_center = self.env.unwrapped._get_pos_objects()
        
        # peg头部位置（通过pegHead site获取）
        try:
            peg_head_site_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                               mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
            peg_head = self.env.unwrapped.data.site_xpos[peg_head_site_id].copy()
        except:
            # 备用方案：基于peg的方向估算头部位置
            peg_quat = self.env.unwrapped._get_quat_objects()
            rotation = Rotation.from_quat(peg_quat)
            # 根据XML，pegHead在peg的-X方向0.1m处
            peg_head = peg_center + rotation.apply(np.array([-0.1, 0, 0]))
        
        return peg_center, peg_head
    
    def _get_peg_rotation(self) -> Rotation:
        """获取peg的当前姿态"""
        peg_quat = self.env.unwrapped._get_quat_objects()
        return Rotation.from_quat(peg_quat)
    
    def _get_gripper_distance(self) -> float:
        """从观测中获取夹爪距离，用于判断是否成功抓取"""
        obs = self.env.unwrapped._get_obs()
        return obs[3]
    
    def _detect_grasp(self) -> bool:
        """检测是否抓取了peg"""
        gripper_distance = self._get_gripper_distance()
        peg_center, _ = self._get_peg_positions()
        hand_pos = self.env.unwrapped.get_endeff_pos()
        
        distance_to_peg = np.linalg.norm(hand_pos - peg_center)
        
        return gripper_distance < 0.3 and distance_to_peg < 0.05
    
    def _update_control_phase(self):
        """更新控制阶段"""
        peg_center, peg_head = self._get_peg_positions()
        hole_pos, _ = self._get_hole_info()
        
        if not self.peg_grasped:
            self.control_phase = "approach"
            self.insertion_stage = "move_to_front"
        else:
            distance_to_hole = np.linalg.norm(peg_head - hole_pos)
            
            if distance_to_hole > self.params.approach_distance:
                self.control_phase = "approach"
                self.insertion_stage = "move_to_front"
            elif distance_to_hole > self.params.alignment_distance:
                self.control_phase = "align"
                self.insertion_stage = "align_orientation"
            else:
                self.control_phase = "insert"
                self.insertion_stage = "insert"
    
    def step(self, action):
        """执行一步，应用力位混合控制"""
        # 检测抓取状态
        self.peg_grasped = self._detect_grasp()
        
        # 更新控制阶段
        self._update_control_phase()
        
        # 获取当前状态
        peg_center, peg_head = self._get_peg_positions()
        peg_rotation = self._get_peg_rotation()
        hole_pos, hole_orientation = self._get_hole_info()
        contact_force, contact_torque = self.force_extractor.get_contact_forces_and_torques()
        
        # 根据控制阶段选择控制策略
        if self.control_phase == "approach":
            # 纯位置控制阶段：移动到hole前方
            modified_action = self._apply_approach_control(action, peg_head, hole_pos)
        elif self.control_phase == "align":
            # 对齐阶段：调整姿态
            modified_action = self._apply_alignment_control(
                action, peg_head, peg_rotation, hole_pos, contact_force, contact_torque)
        elif self.control_phase == "insert":
            # 插入阶段：力位混合控制
            modified_action = self._apply_insertion_control(
                action, peg_head, peg_rotation, hole_pos, contact_force, contact_torque)
        else:
            modified_action = action.copy()
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 检查插入状态
        insertion_status = self.success_checker.check_insertion_success(
            peg_head, peg_center, peg_rotation)
        
        # 记录数据
        self._record_data(peg_center, peg_head, peg_rotation, contact_force, 
                         contact_torque, modified_action, insertion_status)
        
        # 添加额外的评估指标
        info.update(self._compute_evaluation_metrics(insertion_status))
        
        return obs, reward, terminated, truncated, info
    
    def _apply_approach_control(self, action, peg_head_pos, hole_pos):
        """接近阶段控制：移动到hole前方"""
        # 计算hole前方位置
        hole_front_pos = hole_pos + self.task_coord_system.y_axis * self.params.approach_distance
        
        # 简单位置控制
        direction = hole_front_pos - peg_head_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:
            direction = direction / distance
            modified_action = action.copy()
            scale = min(0.1, distance * 2.0)  # 距离越近速度越慢
            modified_action[:3] += direction * scale
            return modified_action
        
        return action.copy()
    
    def _apply_alignment_control(self, action, peg_head_pos, peg_rotation, hole_pos, 
                               contact_force, contact_torque):
        """对齐阶段控制：调整位置和姿态"""
        if self.task_coord_system is None:
            return action.copy()
        
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)
        
        # 获取姿态误差
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 目标位置：hole前方一点
        target_pos_task = hole_pos_task + np.array([0, self.params.alignment_distance, 0])
        target_orientation = np.zeros(3)  # 目标姿态误差为0
        
        # 目标力和力矩
        target_force_task = np.array([0, 0, 0])  # 对齐阶段不需要大的力
        target_torque_task = np.array([0, 0, 0])
        
        # 获取选择矩阵
        selection_matrix_pos, selection_matrix_rot = self._get_selection_matrices()
        
        # 计算控制输出
        dt = 1.0 / self.params.force_control_freq
        pos_output, rot_output = self.controller.compute_control(
            peg_head_task, target_pos_task, orientation_error, target_orientation,
            contact_force_task, target_force_task, contact_torque_task, target_torque_task,
            selection_matrix_pos, selection_matrix_rot, dt)
        
        # 转换回世界坐标系并应用
        pos_output_world = self.task_coord_system.task_force_to_world(pos_output)
        
        modified_action = action.copy()
        pos_scale = 0.02
        modified_action[:3] += pos_output_world * pos_scale
        
        return modified_action
    
    def _apply_insertion_control(self, action, peg_head_pos, peg_rotation, hole_pos, 
                               contact_force, contact_torque):
        """插入阶段控制：力位混合控制"""
        if self.task_coord_system is None:
            return action.copy()
        
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)
        
        # 获取姿态误差
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 目标位置：深入hole内部
        target_pos_task = hole_pos_task + np.array([0, -self.params.min_insertion_depth, 0])
        target_orientation = np.zeros(3)  # 目标姿态误差为0
        
        # 目标力和力矩
        target_force_task = np.array([0, 1.0, 0])  # Y方向允许推力
        target_torque_task = np.array([0, 0, 0])
        
        # 获取选择矩阵
        selection_matrix_pos, selection_matrix_rot = self._get_selection_matrices()
        
        # 计算控制输出
        dt = 1.0 / self.params.force_control_freq
        pos_output, rot_output = self.controller.compute_control(
            peg_head_task, target_pos_task, orientation_error, target_orientation,
            contact_force_task, target_force_task, contact_torque_task, target_torque_task,
            selection_matrix_pos, selection_matrix_rot, dt)
        
        # 转换回世界坐标系并应用
        pos_output_world = self.task_coord_system.task_force_to_world(pos_output)
        
        modified_action = action.copy()
        pos_scale = 0.01
        modified_action[:3] += pos_output_world * pos_scale
        
        return modified_action
    
    def _get_selection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取位置和姿态的选择矩阵"""
        # 默认配置：径向方向（XZ）做力控制，插入方向（Y）做位置控制
        # 姿态控制：径向旋转用力控制，轴向旋转用位置控制
        selection_matrix_pos = np.diag([0, 1, 0])  # Y方向位置控制，XZ方向力控制
        selection_matrix_rot = np.diag([0, 1, 1])  # YZ方向姿态控制，X方向力矩控制
        return selection_matrix_pos, selection_matrix_rot
    
    def _record_data(self, peg_center, peg_head, peg_rotation, contact_force, 
                    contact_torque, control_output, insertion_status):
        """记录实验数据"""
        self.episode_data['positions'].append(peg_center.copy())
        self.episode_data['orientations'].append(peg_rotation.as_quat().copy())
        self.episode_data['contact_forces'].append(contact_force.copy())
        self.episode_data['torques'].append(contact_torque.copy())
        self.episode_data['control_outputs'].append(control_output.copy())
        self.episode_data['phases'].append(f"{self.control_phase}_{self.insertion_stage}")
        self.episode_data['insertion_status'].append(insertion_status.copy())
        self.episode_data['box_positions'].append(self._get_box_position())
    
    def _compute_evaluation_metrics(self, insertion_status: Dict) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 计算最大接触力和力矩
        if self.episode_data['contact_forces']:
            forces = np.array(self.episode_data['contact_forces'])
            torques = np.array(self.episode_data['torques'])
            metrics['max_contact_force'] = np.max(np.linalg.norm(forces, axis=1))
            metrics['avg_contact_force'] = np.mean(np.linalg.norm(forces, axis=1))
            metrics['max_contact_torque'] = np.max(np.linalg.norm(torques, axis=1))
            metrics['avg_contact_torque'] = np.mean(np.linalg.norm(torques, axis=1))
        else:
            metrics['max_contact_force'] = 0.0
            metrics['avg_contact_force'] = 0.0
            metrics['max_contact_torque'] = 0.0
            metrics['avg_contact_torque'] = 0.0
        
        # 计算box移动距离
        if self.initial_box_pos is not None:
            current_box_pos = self._get_box_position()
            box_displacement = np.linalg.norm(current_box_pos - self.initial_box_pos)
            metrics['box_displacement'] = box_displacement
            metrics['environment_damage'] = box_displacement > 0.01
        else:
            metrics['box_displacement'] = 0.0
            metrics['environment_damage'] = False
        
        # 插入相关指标
        metrics['insertion_depth'] = insertion_status['insertion_depth']
        metrics['radial_distance'] = insertion_status['radial_distance']
        metrics['orientation_error'] = insertion_status['orientation_error']
        metrics['insertion_success'] = insertion_status['success']
        metrics['is_inside_hole'] = insertion_status['is_inside_hole']
        metrics['sufficient_depth'] = insertion_status['sufficient_depth']
        metrics['is_aligned'] = insertion_status['is_aligned']
        metrics['is_orientation_aligned'] = insertion_status['is_orientation_aligned']
        
        return metrics

class SimplePolicy:
    """改进的简单策略"""
    
    def __init__(self):
        self.phase = "reach"
        self.grasp_threshold = 0.05
        
    def get_action(self, obs):
        """生成动作"""
        hand_pos = obs[:3]
        gripper_distance = obs[3]
        obj_pos = obs[4:7]
        goal_pos = obs[-3:]
        
        action = np.zeros(4)
        
        # 计算到目标的距离
        hand_to_obj = np.linalg.norm(hand_pos - obj_pos)
        obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        
        if self.phase == "reach":
            # 移动到物体
            if hand_to_obj > self.grasp_threshold:
                direction = (obj_pos - hand_pos) / max(hand_to_obj, 1e-6)
                action[:3] = direction * 0.1
                action[3] = -1  # 打开夹爪
            else:
                self.phase = "grasp"
                
        elif self.phase == "grasp":
            # 抓取物体
            action[:3] = (obj_pos - hand_pos) * 0.5
            action[3] = 1  # 关闭夹爪
            
            if gripper_distance < 0.3 and hand_to_obj < 0.03:
                self.phase = "transport"
                
        elif self.phase == "transport":
            # 运输到目标位置附近（不直接插入）
            target_offset = 0.08  # 停在hole前方8cm
            adjusted_goal = goal_pos + np.array([0, target_offset, 0])  # 假设Y是插入方向
            
            if np.linalg.norm(obj_pos - adjusted_goal) > 0.02:
                direction = (adjusted_goal - obj_pos) / max(np.linalg.norm(adjusted_goal - obj_pos), 1e-6)
                action[:3] = direction * 0.05  # 更慢的速度
                action[3] = 1  # 保持夹爪关闭
            else:
                # 到达目标附近，进行细微调整
                action[:3] = (adjusted_goal - obj_pos) * 1.0
                action[3] = 1
        
        # 限制动作范围
        action[:3] = np.clip(action[:3], -1, 1)
        action[3] = np.clip(action[3], -1, 1)
        
        return action

class SelectionMatrixExperiment:
    """改进的方向选择矩阵实验类"""
    
    def __init__(self, env_name: str = 'peg-insert-side-v3'):
        self.env_name = env_name
        self.results = []
    
    def create_env(self, selection_matrix_func=None):
        """创建带有指定选择矩阵的环境"""
        ml1 = metaworld.ML1(self.env_name, seed=42)
        # 不使用render以提高速度
        env = ml1.train_classes[self.env_name](render_mode='human')
        task = ml1.train_tasks[0]
        env.set_task(task)
        
        # 包装为混合控制环境
        hybrid_env = HybridControlWrapper(env)
        
        # 替换选择矩阵函数
        if selection_matrix_func:
            hybrid_env._get_selection_matrices = selection_matrix_func
        
        return hybrid_env
    
    def define_selection_matrices(self):
        """定义不同的方向选择矩阵配置"""
        matrices = {
            "纯位置控制": lambda: (np.eye(3), np.eye(3)),
            "Y位置XZ力": lambda: (np.diag([0, 1, 0]), np.diag([0, 1, 1])),  # 推荐配置
            "XZ位置Y力": lambda: (np.diag([1, 0, 1]), np.diag([1, 0, 0])),
            "纯力控制": lambda: (np.zeros((3, 3)), np.zeros((3, 3))),
            "位置控制_力矩控制": lambda: (np.eye(3), np.zeros((3, 3))),
            "力控制_姿态控制": lambda: (np.zeros((3, 3)), np.eye(3)),
            "混合控制1": lambda: (np.diag([0, 1, 0]), np.diag([0, 0, 1])),
            "混合控制2": lambda: (np.diag([1, 1, 0]), np.diag([0, 1, 0])),
        }
        return matrices
    
    def run_experiment(self, num_episodes: int = 2, max_steps: int = 800):
        """运行选择矩阵对比实验"""
        matrices = self.define_selection_matrices()
        
        print(f"开始6DOF方向选择矩阵实验，测试{len(matrices)}种配置...")
        
        for matrix_name, matrix_func in matrices.items():
            print(f"\n测试配置: {matrix_name}")
            
            env = self.create_env(matrix_func)
            episode_results = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                policy = SimplePolicy()
                
                episode_data = {
                    'matrix_name': matrix_name,
                    'episode': episode,
                    'total_reward': 0,
                    'success': False,
                    'metrics': {}
                }
                
                for step in range(max_steps):
                    env.render()
                    action = policy.get_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_data['total_reward'] += reward
                    
                    if terminated or truncated:
                        break
                
                # 记录最终指标
                episode_data['success'] = info.get('insertion_success', False)
                episode_data['metrics'] = {
                    'max_contact_force': info.get('max_contact_force', 0),
                    'max_contact_torque': info.get('max_contact_torque', 0),
                    'box_displacement': info.get('box_displacement', 0),
                    'insertion_depth': info.get('insertion_depth', 0),
                    'radial_distance': info.get('radial_distance', 0),
                    'orientation_error': info.get('orientation_error', 0),
                    'environment_damage': info.get('environment_damage', False),
                    'is_inside_hole': info.get('is_inside_hole', False),
                    'is_orientation_aligned': info.get('is_orientation_aligned', False)
                }
                
                episode_results.append(episode_data)
                print(f"  Episode {episode}: Success={episode_data['success']}, "
                      f"Depth={episode_data['metrics']['insertion_depth']:.3f}, "
                      f"Radial={episode_data['metrics']['radial_distance']:.3f}, "
                      f"Orient={episode_data['metrics']['orientation_error']:.3f}")
            
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
            'avg_max_contact_torque': np.mean([r['metrics']['max_contact_torque'] for r in episode_results]),
            'avg_box_displacement': np.mean([r['metrics']['box_displacement'] for r in episode_results]),
            'avg_insertion_depth': np.mean([r['metrics']['insertion_depth'] for r in episode_results]),
            'avg_radial_distance': np.mean([r['metrics']['radial_distance'] for r in episode_results]),
            'avg_orientation_error': np.mean([r['metrics']['orientation_error'] for r in episode_results]),
            'damage_rate': np.mean([r['metrics']['environment_damage'] for r in episode_results]),
            'inside_hole_rate': np.mean([r['metrics']['is_inside_hole'] for r in episode_results]),
            'orientation_aligned_rate': np.mean([r['metrics']['is_orientation_aligned'] for r in episode_results])
        }
        
        return avg_result
    
    def print_results(self):
        """打印实验结果"""
        print("\n" + "="*120)
        print("6DOF方向选择矩阵实验结果汇总")
        print("="*120)
        
        # 表头
        header = f"{'配置名称':<20} {'成功率':<8} {'插入深度':<10} {'径向距离':<10} {'姿态误差':<10} {'孔内率':<8} {'姿态对齐率':<12} {'接触力':<10}"
        print(header)
        print("-"*120)
        
        # 数据行
        for result in self.results:
            row = (f"{result['matrix_name']:<20} "
                   f"{result['success_rate']:<8.2f} "
                   f"{result['avg_insertion_depth']:<10.3f} "
                   f"{result['avg_radial_distance']:<10.3f} "
                   f"{result['avg_orientation_error']:<10.3f} "
                   f"{result['inside_hole_rate']:<8.2f} "
                   f"{result['orientation_aligned_rate']:<12.2f} "
                   f"{result['avg_max_contact_force']:<10.1f}")
            print(row)
        
        print("-"*120)
        
        # 找出最优配置
        if self.results:
            best_success = max(self.results, key=lambda x: x['success_rate'])
            best_precision = min(self.results, key=lambda x: (x['avg_radial_distance'] + x['avg_orientation_error']) if x['success_rate'] > 0 else float('inf'))
            
            print(f"\n最高成功率: {best_success['matrix_name']} ({best_success['success_rate']:.2f})")
            if best_precision['success_rate'] > 0:
                print(f"最高精度: {best_precision['matrix_name']} (综合误差: {best_precision['avg_radial_distance'] + best_precision['avg_orientation_error']:.3f})")

def demo_enhanced_hybrid_control():
    """演示改进的6DOF力位混合控制"""
    print("开始改进的6DOF力位混合控制演示...")
    
    # 创建实验环境
    experiment = SelectionMatrixExperiment()
    
    # 运行实验
    experiment.run_experiment(num_episodes=3, max_steps=400)
    
    # 显示结果
    experiment.print_results()

if __name__ == "__main__":
    demo_enhanced_hybrid_control()