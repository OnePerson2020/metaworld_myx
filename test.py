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

    def __init__(self, position_gain=0.025, rotation_gain=0.04, force_feedback_gain=10, force_threshold=15):

        super().__init__()
        self.position_gain = position_gain
        self.rotation_gain = rotation_gain
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False

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
        action = Action(7)

        desired_pos, desired_r = self._desired_pose(o_d)
        
        force_vector = o_d["pegHead_force"]
        force_magnitude = np.linalg.norm(force_vector)
    
        # # 仅在插入阶段（例如阶段4）且力超过阈值时应用调整
        # if self.current_stage == 4 and force_magnitude > self.force_threshold:
        #     # peg_head - peg
        #     lever_arm = np.array([-0.15, 0, 0])
        #     torque_vector = np.cross(lever_arm, force_vector) * self.force_feedback_gain
        #     # 应用旋转修正
        #     r_correction = Rotation.from_rotvec(torque_vector)
        #     desired_r = r_correction * desired_r
            
        #     print(f"Rot Correction Axis: {r_correction.as_euler('xyz', degrees=True)}")

        # desired_pos = np.array([0.2,0.6,0.3])
        delta_pos = move(o_d["hand_pos"], to_xyz=desired_pos, p=self.position_gain)
        delta_rot = self._calculate_rotation_action(o_d["hand_quat"], desired_r)

        gripper_effort = self._grab_effort(o_d)

        full_action = np.hstack((delta_pos, delta_rot, gripper_effort))
        # full_action = np.zeros(7)
        
        action.set_action(full_action)
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = o_d["goal_pos"]
        ini_r = Rotation.from_euler('xyz', [0,90,0], degrees=True)
        # desired_r = ini_r * Rotation.from_euler('xyz', [5,0,0], degrees=True)
        
        # 阶段1: 移动到peg正上方
        if self.current_stage == 1:
            # print("Stage 1: Moving to peg top")
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04:
                self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), ini_r

        # 阶段2: 下降抓取peg
        if self.current_stage == 2:
            # print(f"Stage 2: Descending to peg.")
            if pos_curr[2] - pos_peg[2] < -0.005:
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
            return pos_hole + np.array([0.0, 0.0, 0.01]), ini_r
            
        return None

    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
        """
        根据当前和目标姿态，计算出平滑的旋转增量（欧拉角格式）。
        """
        # 步骤 0: 将输入从MuJoCo格式([w,x,y,z])转换为SciPy的Rotation对象
        r_curr = Rotation.from_quat(current_quat_mujoco[[1, 2, 3, 0]])
        
        
        # 步骤 1: 计算旋转误差 (从当前到目标的差异)
        r_error = target_Rotation * r_curr.inv()
        print(f"Current Rotation: {r_error.as_euler('xyz', degrees=True)}")
    
        # 步骤 2: 将差异旋转转换为“旋转向量”(Axis-Angle)
        error_rotvec = r_error.as_rotvec()

        # 步骤 3: 应用增益，计算本次步进的增量
        increment_rotvec = error_rotvec * self.rotation_gain
        
        # 步骤 4: 将这个小的增量向量转换回Rotation对象
        r_increment = Rotation.from_rotvec(increment_rotvec)
        
        # 步骤 5: 将增量转换为环境期望的“欧拉角”格式
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
        while count < 500 and not done:
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
    