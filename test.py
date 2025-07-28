import re
import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random
import pandas as pd

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

class CorrectedPolicyV2(Policy):

    def __init__(self, force_feedback_gain=1, force_threshold=15):

        super().__init__()
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.ini_r = Rotation.from_quat([0,1,0,1])

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.ini_r = Rotation.from_quat([0,1,0,1])

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
        
        desired_pos, desired_r = self._desired_pose(o_d)
        # desired_pos = o_d["hand_pos"]
        delta_pos = move(o_d["hand_pos"], to_xyz=desired_pos)
        
        delta_rot_quat = self._calculate_rotation_action(o_d["hand_quat"], desired_r)
        # delta_rot = np.zeros(3)
        gripper_effort = self._grab_effort(o_d)

        force_vector = o_d["pegHead_force"]
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude > 10:
            delta_pos = np.zeros(3)
        
        full_action = np.hstack((delta_pos, delta_rot_quat, gripper_effort))
        
        action = Action(8)
        action.set_action(full_action)
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = o_d["goal_pos"]
        gripper_distance = o_d["gripper_distance_apart"]

        force_vector = o_d["pegHead_force"]
        force_magnitude = np.linalg.norm(force_vector)

        # 仅在阶段4 (插入阶段) 且力大于阈值时激活
        if self.current_stage == 4 and force_magnitude > 10:
            
            # 1. 获取peg当前的姿态（从mujoco四元数转换为scipy Rotation对象）
            peg_current_rotation = Rotation.from_quat(o_d["peg_rot"])
            
            # 2. 定义在peg局部坐标系下的力臂向量
            lever_arm_local = np.array([-1, 0.0, -0.1])
            
            # 3. 将力臂向量从局部坐标系转换到世界坐标系
            lever_arm_world = peg_current_rotation.apply(lever_arm_local)
            
            corrective_torque_vector = np.cross(lever_arm_world, force_vector)* 1
            
            # 5. 限制修正速度，保证稳定性
            speed = np.deg2rad(2) # 最大修正角速度
            torque_magnitude = np.linalg.norm(corrective_torque_vector)
            
            if torque_magnitude > 1e-6: # 避免除以零
                unit_torque_axis = corrective_torque_vector / torque_magnitude
                
                # 如果计算出的旋转速度超过上限，则使用上限速度
                if torque_magnitude > speed:
                    increment_rotvec = unit_torque_axis * speed
                else:
                    increment_rotvec = corrective_torque_vector
                    
                # 6. 生成修正旋转并更新累积的目标姿态
                #    这是关键修正点之二：对 self.ini_r 进行累积更新
                r_correction = Rotation.from_rotvec(increment_rotvec)
                self.ini_r = r_correction * self.ini_r
                
                # desir_pos = pos_curr
                
                print(f"Force Detected: {force_magnitude:.2f} N. Applying rotational correction.")
                print(f"Corrected Target Euler: {r_correction.as_euler('xyz', degrees=True)}")
                   
        # 阶段1: 移动到peg正上方
        if self.current_stage == 1:
            # print("Stage 1: Moving to peg top")
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04:
                self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), self.ini_r
            # return pos_curr, self.ini_r

        # 阶段2: 下降抓取peg
        if self.current_stage == 2:
            # print(f"Stage 2: Descending to peg.")
            if pos_curr[2] - pos_peg[2] < -0.001 and gripper_distance < 0.35:
                # print(">>> Peg lifted! Transitioning to Stage 3.")
                self.current_stage = 3
            return pos_peg - np.array([0.0, 0.0, 0.02]), self.ini_r
            
        # 阶段3: 移动到洞口预备位置并旋转
        if self.current_stage == 3:
            # print("Stage 3: Moving to hole side")
            if np.linalg.norm(pos_curr[1:] - pos_hole[1:]) < 0.03:
                self.current_stage = 4
            return pos_hole + np.array([0.4, 0.0, 0.0]), self.ini_r
        
        # 阶段4: 执行插入
        if self.current_stage == 4:
            # print("Stage 4: Inserting peg")
            desir_pos = pos_hole + np.array([0.1, 0.0, 0.0])
            return desir_pos, self.ini_r
            
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

# --- Main Execution Block ---
if __name__ == "__main__":
    env_name = 'peg-insert-side-v3'
    env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
    env = env_class(render_mode='human', width=1280, height=720) # human or rgb_array

    benchmark = ppo_test.MT1(env_name)

    policy = CorrectedPolicyV2()

    all_force_data = []
    num_episodes = 5
    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        # task = benchmark.train_tasks[0] 
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)
        
        obs, info = env.reset()
        policy.reset()

        episode_forces = []
        done = False
        count = 0
        while count < 5000 and not done:
            env.render()
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 从info字典中记录力的信息
            force_magnitude = info.get('pegHead_force_magnitude', 0.0)
            force_direction = info.get('pegHead_force_direction', np.zeros(3))
            episode_forces.append({
                'step': count,
                'magnitude': force_magnitude,
                'direction_x': force_direction[0],
                'direction_y': force_direction[1],
                'direction_z': force_direction[2],
            })

            done = terminated or truncated

            if info.get('success', 0.0) > 0.5:
                print("任务成功！")
                # time.sleep(1)
                break

            # time.sleep(0.01)
            count += 1
        
        for data_point in episode_forces:
            data_point['episode'] = i + 1
        all_force_data.extend(episode_forces)
        
        print(f"Episode finished. Final Info: {info}")
    env.close()
    
    # --- 数据保存 ---
    df = pd.DataFrame(all_force_data)
    df.to_csv("force_analysis.csv", index=False)
    print("\nForce analysis data saved to force_analysis.csv")
    from visualize_forces import visualize_force_data
    visualize_force_data("force_analysis.csv")
    