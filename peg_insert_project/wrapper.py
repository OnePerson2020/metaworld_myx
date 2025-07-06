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
        
        self.control_phase = "approach"
        self._get_selection_matrices_func = None

    def set_selection_matrices_func(self, func: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        self._get_selection_matrices_func = func

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.controller.reset()
        self.control_phase = "approach"
        
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

    def step(self, action):
        if self.task_coord_system is None:
            raise RuntimeError("环境未重置，请先调用reset()")

        peg_center, peg_head, peg_rotation = self._get_peg_state()
        contact_force, contact_torque = self.force_extractor.get_contact_forces_and_torques()
        
        # 将接触力/力矩转换到任务坐标系
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)

        # 获取当前状态在任务坐标系下的表示
        peg_head_task = self.task_coord_system.world_to_task(peg_head)
        orientation_error_task = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 设定目标（在任务坐标系下）
        # 目标位置是原点，目标姿态误差是0
        target_pos_task = np.zeros(3)
        target_orientation_error_task = np.zeros(3)
        
        # 目标力/力矩在纯位置控制下为0
        target_force_task = np.zeros(3)
        target_torque_task = np.zeros(3)

        # 获取选择矩阵
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
        # pos_out和rot_out是期望的校正量
        pos_correction_world = self.task_coord_system.task_force_to_world(pos_out)
        
        # 简单的姿态校正，直接使用世界坐标系下的误差
        # rot_out是任务坐标系下的期望力矩/角速度，直接施加到动作上
        rot_correction_world = self.task_coord_system.task_force_to_world(rot_out)


        # 将底层的PD控制器输出与上层策略给的动作结合
        # 上层策略提供一个大的方向，底层控制器进行精细微调
        # 这里使用一个简单的加权融合
        modified_action = action.copy()
        modified_action[:3] = action[:3] + pos_correction_world * 0.1 # 底层控制器权重较小
        # 姿态部分暂时不混合，依赖上层策略
        
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 更新info
        final_peg_center, final_peg_head, final_peg_rotation = self._get_peg_state()
        insertion_status = self.success_checker.check_insertion_success(
            final_peg_head, final_peg_center, final_peg_rotation)
        info.update(insertion_status)
        
        return obs, reward, terminated, truncated, info