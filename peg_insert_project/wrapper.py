# wrapper.py
import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Callable

# 从本地模块导入
from params import ControlParams
from force_extractor import ForceExtractor
from controllers import Enhanced6DOFController
from coordinate_systems import TaskCoordinateSystem
from success_checker import InsertionSuccessChecker

class HybridControlWrapper(gym.Wrapper):
    """力位混合控制包装器"""
    
    def __init__(self, env, control_params: ControlParams = None):
        super().__init__(env)
        self.params = control_params or ControlParams()
        self.force_extractor = ForceExtractor(env)
        self.controller = Enhanced6DOFController(self.params)
        
        self.task_coord_system = None
        self.success_checker = None
        
        # 插入控制的阶段状态
        self.insertion_phase = "approach"  # approach -> align -> insert
        self.phase_start_time = 0
        self._get_selection_matrices_func = None

    def set_selection_matrices_func(self, func: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        self._get_selection_matrices_func = func

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.controller.reset()
        self.insertion_phase = "approach"
        self.phase_start_time = 0
        
        hole_pos, hole_orientation = self._get_hole_info()
        self.task_coord_system = TaskCoordinateSystem(hole_pos, hole_orientation)
        self.success_checker = InsertionSuccessChecker(self.params, self.task_coord_system)
        
        return obs, info

    def _get_hole_info(self) -> Tuple[np.ndarray, np.ndarray]:
        hole_site_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'hole')
        hole_pos = self.env.unwrapped.data.site_xpos[hole_site_id].copy()
        box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'box')
        box_quat = self.env.unwrapped.data.xquat[box_body_id].copy()
        rotation = Rotation.from_quat(box_quat)
        hole_direction = rotation.apply(np.array([0, 1, 0]))
        return hole_pos, hole_direction

    def _get_peg_state(self) -> Tuple[np.ndarray, np.ndarray, Rotation]:
        peg_center = self.env.unwrapped._get_pos_objects()
        peg_head_site_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
        peg_head = self.env.unwrapped.data.site_xpos[peg_head_site_id].copy()
        peg_quat = self.env.unwrapped._get_quat_objects()
        peg_rotation = Rotation.from_quat(peg_quat)
        return peg_center, peg_head, peg_rotation

    def _determine_insertion_phase(self, peg_head_pos: np.ndarray, peg_rotation: Rotation, 
                                 contact_force: np.ndarray) -> str:
        """根据当前状态确定插入阶段"""
        hole_pos, _ = self._get_hole_info()
        distance_to_hole = np.linalg.norm(peg_head_pos - hole_pos)
        
        # 转换到任务坐标系检查对齐情况
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 检查径向偏差和姿态误差
        radial_error = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        orientation_magnitude = np.linalg.norm(orientation_error)
        
        # 检查是否有接触力
        has_contact = np.linalg.norm(contact_force) > 0.5
        
        if self.insertion_phase == "approach":
            # 接近阶段：距离hole较远时
            if distance_to_hole < self.params.approach_distance:
                return "align"
            return "approach"
            
        elif self.insertion_phase == "align":
            # 对齐阶段：调整位置和姿态
            if (radial_error < self.params.insertion_tolerance and 
                orientation_magnitude < self.params.max_orientation_error and
                distance_to_hole < self.params.alignment_distance):
                return "insert"
            return "align"
            
        elif self.insertion_phase == "insert":
            # 插入阶段：保持插入状态
            if has_contact or -peg_head_task[1] > 0.01:  # 已经有接触或已经插入
                return "insert"
            else:
                return "align"  # 如果失去接触，回到对齐阶段
                
        return self.insertion_phase

    def _get_target_position(self, peg_head_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        """根据插入阶段确定目标位置"""
        hole_pos, hole_orientation = self._get_hole_info()
        
        if self.insertion_phase == "approach":
            # 接近阶段：跟随上层策略，但不要太接近hole
            target_world = hole_pos + np.array([0,10,0])
            # 融合上层策略的意图
            target_world += action[:3] * 0.02
            
        elif self.insertion_phase == "align":
            # 对齐阶段：目标位置在hole前方一小段距离
            target_world = hole_pos + hole_orientation * self.params.alignment_distance
            
        elif self.insertion_phase == "insert":
            # 插入阶段：目标位置在hole内部
            target_world = hole_pos + hole_orientation * self.params.min_insertion_depth
            
        return target_world

    def step(self, action):
        if self.task_coord_system is None:
            raise RuntimeError("环境未重置，请先调用reset()")

        peg_center, peg_head, peg_rotation = self._get_peg_state()
        contact_force, contact_torque = self.force_extractor.get_contact_forces_and_torques()
        
        # 更新插入阶段
        old_phase = self.insertion_phase
        self.insertion_phase = self._determine_insertion_phase(peg_head, peg_rotation, contact_force)
        
        if old_phase != self.insertion_phase:
            print(f"插入阶段切换: {old_phase} -> {self.insertion_phase}")
        
        # 将接触力/力矩转换到任务坐标系
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)

        # 获取当前状态在任务坐标系下的表示
        peg_head_task = self.task_coord_system.world_to_task(peg_head)
        orientation_error_task = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 根据阶段确定目标位置
        target_pos_world = self._get_target_position(peg_head, action)
        target_pos_task = self.task_coord_system.world_to_task(target_pos_world)
        
        # 目标姿态始终是对齐到hole
        target_orientation_error_task = np.zeros(3)
        
        # 根据阶段设置目标力/力矩
        if self.insertion_phase == "insert":
            # 插入阶段：Y方向施加插入力，其他方向保持0力
            target_force_task = np.array([0, 5.0, 0])  # Y方向插入力
            target_torque_task = np.zeros(3)
        else:
            # 其他阶段：所有力/力矩都为0
            target_force_task = np.zeros(3)
            target_torque_task = np.zeros(3)

        # 根据阶段调整选择矩阵
        if self.insertion_phase == "insert":
            # 插入阶段：Y方向改为力控制
            selection_pos = np.diag([1, 0, 1])  # X,Z位置控制，Y力控制
            selection_rot = np.eye(3)  # 姿态仍然位置控制
        else:
            # 其他阶段：纯位置控制
            selection_pos, selection_rot = self._get_selection_matrices_func()

        # 计算控制器输出
        dt = self.env.unwrapped.dt
        pos_out, rot_out = self.controller.compute_control(
            peg_head_task, target_pos_task,
            orientation_error_task, target_orientation_error_task,
            contact_force_task, target_force_task,
            contact_torque_task, target_torque_task,
            selection_pos, selection_rot, dt
        )

        # 将控制器输出转换回世界坐标系
        pos_correction_world = self.task_coord_system.task_force_to_world(pos_out)
        rot_correction_world = self.task_coord_system.task_force_to_world(rot_out)

        # 根据阶段调整控制器权重
        if self.insertion_phase == "approach":
            # 接近阶段：主要依赖上层策略
            weight_upper = 0.8
            weight_lower = 0.2
        elif self.insertion_phase == "align":
            # 对齐阶段：增加底层控制器权重
            weight_upper = 0.3
            weight_lower = 0.7
        elif self.insertion_phase == "insert":
            # 插入阶段：主要依赖底层控制器
            weight_upper = 0.1
            weight_lower = 0.9
        
        # 限制控制器输出幅度
        pos_correction_world = np.clip(pos_correction_world, -0.2, 0.2)
        rot_correction_world = np.clip(rot_correction_world, -0.2, 0.2)
        
        # 融合上层策略和底层控制器
        modified_action = action.copy()
        modified_action[:3] = (weight_upper * action[:3] + 
                              weight_lower * pos_correction_world)
        
        # 限制最终动作幅度
        modified_action[:3] = np.clip(modified_action[:3], -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 更新info
        final_peg_center, final_peg_head, final_peg_rotation = self._get_peg_state()
        insertion_status = self.success_checker.check_insertion_success(
            final_peg_head, final_peg_center, final_peg_rotation)
        info.update(insertion_status)
        
        # 添加阶段信息到info
        info['insertion_phase'] = self.insertion_phase
        info['target_pos_world'] = target_pos_world
        info['distance_to_target'] = np.linalg.norm(final_peg_head - target_pos_world)
        
        # 调试信息
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            
        if self._debug_step_count % 100 == 0:
            hole_pos, _ = self._get_hole_info()
            print(f"Debug - Phase: {self.insertion_phase}, "
                  f"Distance to hole: {np.linalg.norm(final_peg_head - hole_pos):.3f}, "
                  f"Contact force: {np.linalg.norm(contact_force):.3f}")
        
        return obs, reward, terminated, truncated, info