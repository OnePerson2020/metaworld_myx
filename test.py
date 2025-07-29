import re

from matplotlib.pyplot import plot
import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random
import pandas as pd

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed

plot_forces = []
plot_torque = []


class CorrectedPolicyV2(Policy):

    def __init__(self, force_feedback_gain=1, force_threshold=15):

        super().__init__()
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_quat([0,1,0,1])

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_quat([0,1,0,1])

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "hand_quat": obs[3:7],
            "gripper_distance_apart": obs[7],
            "pegHead_force": obs[8:11],
            "pegHead_torque": obs[11:14],       
            "peg_pos": obs[14:17],            
            "peg_rot": obs[17:21],            
            "unused_info_curr_obs": obs[21:28],
            "_prev_obs": obs[28:56],          
            "goal_pos": obs[-3:],             
        }
    
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        
        o_d = self._parse_obs(obs)
        
        desired_pos, desired_r = self._desired_pose(o_d)
        # desired_pos = o_d["hand_pos"]
        delta_pos = self._calculate_pos_action(o_d["hand_pos"], to_xyz=desired_pos)
        
        delta_rot_euler = self._calculate_rotation_action(o_d["hand_quat"], desired_r)
        # delta_rot = np.zeros(3)
        gripper_effort = self._grab_effort(o_d)
        
        # if self.current_stage == 4 and np.linalg.norm(o_d["pegHead_force"]) > 5:
        #     delta_pos, delta_rot_euler = self._pos_im(o_d["pegHead_force"], o_d["pegHead_torque"], delta_pos, delta_rot_euler)

        delta_rot = Rotation.from_euler('xyz', delta_rot_euler, degrees=True)
        delta_rot_quat = delta_rot.as_quat()
        action = Action(8)
        action.set_action(np.hstack((delta_pos, delta_rot_quat, gripper_effort)))
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = o_d["goal_pos"]
        gripper_distance = o_d["gripper_distance_apart"]
                   
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
        if self.current_stage == 4 :
            # print("Stage 4: Inserting peg")
            desir_pos = pos_hole + np.array([0.3, 0.0, 0.0])
                    
            return pos_curr, self.ini_r
        return None

    def _calculate_pos_action(self,
        from_xyz: npt.NDArray[any], 
        to_xyz: npt.NDArray[any], 
        speed: float = 0.2
    ) -> npt.NDArray[any]:
        """
        根据一个恒定的速度预算，计算从一点到另一点的移动向量。

        Args:
            from_xyz: 起始坐标。
            to_xyz: 目标坐标。
            max_dist_per_step: 在这一个时间步内允许移动的最大距离。

        Returns:
            一个代表本次位移的XYZ向量。
        """
        error_vec = to_xyz - from_xyz
        Kp = 0.1
        Kd = 0.8
        
        distance = np.linalg.norm(error_vec)
        max_dist_per_step = speed * 0.0125
        # 如果距离非常小，则不移动
        if distance < 1e-6:
            return np.zeros(3)

        # 如果剩余距离小于单步最大距离，则直接移动到终点以避免过冲
        if distance < max_dist_per_step:
            return Kp * error_vec + Kd * self.prev_pos_error

        # 否则，沿着指向目标的方向，移动一个步长的距离
        direction = error_vec / distance
        delta_pos = direction * max_dist_per_step
        
        self.prev_pos_error = error_vec
        return delta_pos

    def _pos_im(self, force, torque, delta_pos, delta_rot_euler):
        dt = 0.0125
        M_d_inv = np.diag([
            0/100.0, 0/100.0, 0/200.0,  # Mass_inv for x, y, z
            0/0.01, 0/0.01, 0     # Inertia_inv for rot x, y, z
        ])
        
        # 在所有需要柔顺的轴上都加上阻尼，防止震荡
        # 关键是Z轴线性和XY轴旋转
        D_d = np.diag([
            5.0, 5.0, 8.0,      # Damping for x, y, z
            0.5, 0.5, 0.5       # Damping for rot x, y, z
        ])
        
        # 刚度可以先设为0，避免控制器抵抗有用的接触力
        K_d = np.diag([
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        
        F_ext = np.concatenate([force, torque])
        F_ext = np.clip(F_ext, -50, 50)
        E_ddot = M_d_inv @ (F_ext - D_d @ self.e_dot_im - K_d @ self.e_im)
        self.e_dot_im += E_ddot * dt
        self.e_im +=  self.e_dot_im * dt

        self.e_im[:3] = np.clip(self.e_im[:3], -0.01, 0.01) # 线性位移限制
        self.e_im[3:] = np.clip(self.e_im[3:], -np.deg2rad(5), np.deg2rad(5)) # 旋转限制

        # print(force)
        plot_forces.append(force)
        # print(torque)
        plot_torque.append(torque)
        # print(E_ddot)
        print('-----------------')
        delta_pos = self.e_im[0:3]
        delta_rot_euler = self.e_im[3:6]
        print(delta_rot_euler)
        return delta_pos, delta_rot_euler
    
    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
            """
            根据当前和目标姿态，计算出平滑的旋转增量（欧拉角格式）。
            如果角度差大于1度，则以恒定的1度角速度旋转；否则，旋转剩余的角度。
            """
            kp = 0.1
            kd = 0.8
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
                return np.array([0., 0., 0.])
            
            r_increment = Rotation.from_rotvec(increment_rotvec)
            delta_rot_euler = r_increment.as_euler('xyz', degrees=True)

            return delta_rot_euler

    def _grab_effort(self, o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]

        if not self.gasp:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04 and (pos_curr[2] - pos_peg[2]) < -0.001:
                self.gasp = True
                return 0.4
            return -1.0
        else:
            return 0.4

# --- Main Execution Block ---
if __name__ == "__main__":
    env_name = 'peg-insert-side-v3'
    env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
    env = env_class(render_mode='human', width=1280, height=720) # human or rgb_array
    
    benchmark = ppo_test.MT1(env_name)

    policy = CorrectedPolicyV2()

    all_force_data = []
    num_episodes = 3
    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        # task = benchmark.train_tasks[0] 
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)
        
        obs, info = env.reset()
        policy.reset()
        env.mujoco_renderer.viewer.cam.azimuth = 245
        env.mujoco_renderer.viewer.cam.elevation = -20
        
        episode_forces = []
        done = False
        count = 0
        while count < 5000 and not done:
            env.render()
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 从info字典中记录力的信息
            force = info.get('pegHead_force', np.zeros(3))
            force_magnitude = np.linalg.norm(force)
            episode_forces.append({
                'step': count,
                'magnitude': force_magnitude,
                'direction_x': force[0],
                'direction_y': force[1],
                'direction_z': force[2],
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
