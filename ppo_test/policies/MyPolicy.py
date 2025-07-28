from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

class MyPolicy(Policy):

    def __init__(self, force_feedback_gain=1, force_threshold=15):

        super().__init__()
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "hand_quat": obs[3:7],
            "gripper_distance_apart": obs[7],
            "pegHead_force": obs[8:11],
            "peg_pos": obs[11:14],
            "peg_rot": obs[14:18],
            "unused_info_curr_obs": obs[18:25],
            "_prev_obs": obs[25:50],
            "goal_pos": obs[-3:],
        }
    
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action(8)

        desired_pos, desired_r = self._desired_pose(o_d)
        
        force_vector = o_d["pegHead_force"]
        force_magnitude = np.linalg.norm(force_vector)
    
        if self.current_stage == 4 and force_magnitude > 1:
            # peg_head - peg
            lever_arm = np.array([-0.15, 0, 0])
            torque_vector = np.cross(lever_arm, force_vector) * self.force_feedback_gain
            # 应用旋转修正
            r_correction = Rotation.from_rotvec(torque_vector)
            desired_r = r_correction * desired_r
            
            print(f"Rot Correction Axis: {r_correction.as_euler('xyz', degrees=True)}")

        # desired_pos = np.array([0.2,0.4,0.3])
        desired_pos = o_d["hand_pos"]
        delta_pos = move(o_d["hand_pos"], to_xyz=desired_pos)
        delta_rot_quat = self._calculate_rotation_action(o_d["hand_quat"], desired_r)

        # delta_rot = np.zeros(3)

        gripper_effort = self._grab_effort(o_d)

        full_action = np.hstack((delta_pos, delta_rot_quat, gripper_effort))
        # full_action = np.zeros(7)
        
        action.set_action(full_action)
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = o_d["goal_pos"]
        gripper_distance = o_d["gripper_distance_apart"]
        ini_r = Rotation.from_quat([0,1,0,1])
        # desired_r = ini_r * Rotation.from_euler('xyz', [0,0,45], degrees=True)
        
        # 阶段1: 移动到peg正上方
        if self.current_stage == 1:
            # print("Stage 1: Moving to peg top")
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04:
                self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), ini_r

        # 阶段2: 下降抓取peg
        if self.current_stage == 2:
            # print(f"Stage 2: Descending to peg.")
            if pos_curr[2] - pos_peg[2] < -0.001 and gripper_distance < 0.35:
                # print(">>> Peg lifted! Transitioning to Stage 3.")
                self.current_stage = 3
            return pos_peg - np.array([0.0, 0.0, 0.02]), ini_r
            
        # 阶段3: 移动到洞口预备位置并旋转
        if self.current_stage == 3:
            # print("Stage 3: Moving to hole side")
            if np.linalg.norm(pos_curr[1:] - pos_hole[1:]) < 0.03:
                self.current_stage = 4
            return pos_hole + np.array([0.4, 0.0, 0.0]), ini_r
        
        # 阶段4: 执行插入
        if self.current_stage == 4:
            # print("Stage 4: Inserting peg")
            return pos_hole + np.array([0.1, 0.0, 0.0]), ini_r
            
        return None

    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
            """
            根据当前和目标姿态，计算出平滑的旋转增量（欧拉角格式）。
            如果角度差大于1度，则以恒定的1度角速度旋转；否则，旋转剩余的角度。
            """
            kp = 0.3
            kd = 0.3
            speed = np.deg2rad(0.5)
            
            r_curr = Rotation.from_quat(current_quat_mujoco[[1, 2, 3, 0]])
            r_error = target_Rotation * r_curr.inv()
        
            # 步骤 2: 将差异旋转转换为“旋转向量”(Axis-Angle)
            error_rotvec = r_error.as_rotvec()
            rotation_axis = error_rotvec / np.linalg.norm(error_rotvec) + 1e-16
            # print(f"Current Rotation Error: {np.rad2deg(angle_in_radians):.2f} degrees")
            unconstrained_increment_rotvec = kp * error_rotvec + kd * (error_rotvec - self.prev_r_error_rotvec)
            self.prev_r_error_rotvec = error_rotvec
            
            speed_of_increment = np.linalg.norm(unconstrained_increment_rotvec)
            
            if speed_of_increment > speed:
                increment_rotvec = rotation_axis * speed
            else:
                increment_rotvec = unconstrained_increment_rotvec

            angle_of_increment = np.linalg.norm(increment_rotvec)
            if angle_of_increment < 1e-6:
                return np.array([0., 0., 0., 1.])
            
            r_increment = Rotation.from_rotvec(increment_rotvec)
            delta_rot_quat = r_increment.as_quat()
                     
            r_increment = Rotation.from_rotvec(increment_rotvec)

            delta_rot_quat = r_increment.as_quat()

            return delta_rot_quat


    def _grab_effort(self, o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]

        if not self.gasp:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04 and (pos_curr[2] - pos_peg[2]) < -0.001:
                self.gasp = True
                return 0.4
            return -1.0
        else:
            return 0.6