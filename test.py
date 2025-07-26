import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

class CorrectedPolicyV2(Policy):

    def __init__(self, position_gain = 0.25, rotation_gain = 1.5):
        super().__init__()
        self.position_gain = position_gain
        self.rotation_gain = rotation_gain
        self.current_stage = 1
        self.peg_init_z = 0.0 # 用于存储peg的初始高度

    def reset(self, obj_init_pos: npt.NDArray[Any]):
        """在每轮开始时重置策略状态，并记录peg的初始Z坐标。"""
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.peg_init_z = obj_init_pos[2]
        print(f"Peg initial Z-height recorded: {self.peg_init_z:.4f}")

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

    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
        """
        根据当前和目标姿态，计算出平滑的旋转增量（欧拉角格式）。
        """
        # 步骤 0: 将输入从MuJoCo格式([w,x,y,z])转换为SciPy的Rotation对象
        r_curr = Rotation.from_quat(current_quat_mujoco[[1, 2, 3, 0]])
        r_target = target_Rotation

        # 步骤 1: 计算旋转误差 (从当前到目标的差异)
        r_error = r_target * r_curr.inv()

        # 步骤 2: 将差异旋转转换为“旋转向量”(Axis-Angle)
        error_rotvec = r_error.as_rotvec()

        # 步骤 3: 应用增益，计算本次步进的增量
        # self.rotation_gain 就是您提到的“速度”控制器
        increment_rotvec = error_rotvec * self.rotation_gain
        
        # 步骤 4: 将这个小的增量向量转换回Rotation对象
        r_increment = Rotation.from_rotvec(increment_rotvec)
        
        # 步骤 5: 将增量转换为环境期望的“欧拉角”格式
        delta_rot_euler = r_increment.as_euler('xyz', degrees=False)

        return delta_rot_euler
    
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action(7)

        desired_pos, desired_r = self._desired_pose(o_d)
        
        delta_pos = move(o_d["hand_pos"], to_xyz=desired_pos, p=self.position_gain)
        delta_rot = self._calculate_rotation_action(o_d["hand_quat"], desired_r)

        gripper_effort = self._grab_effort(o_d)

        full_action = np.hstack((delta_pos, delta_rot, gripper_effort))
        action.set_action(full_action)
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = np.array([-0.3, o_d["goal_pos"][1], 0.13])
        ini_r = Rotation.from_euler('xyz', [0,90,0], degrees=True)
        desired_r = ini_r * Rotation.from_euler('xyz', [15,0,0], degrees=True)
        
        # 阶段1: 移动到peg正上方
        if self.current_stage == 1:
            print("Stage 1: Moving to peg top")
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04:
                self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), ini_r

        # 阶段2: 下降抓取peg
        if self.current_stage == 2:
            print(f"Stage 2: Descending to peg. Current peg Z: {pos_peg[2]:.4f}, Target: > {self.peg_init_z + 0.02:.4f}")
            if np.linalg.norm(pos_curr[2] - pos_peg[2]) < 0.04 :
                print(">>> Peg lifted! Transitioning to Stage 3.")
                self.current_stage = 3
            return pos_peg, ini_r

        # 阶段3: 移动到洞口预备位置并旋转
        if self.current_stage == 3:
            print("Stage 3: Moving to hole side")
            if np.linalg.norm(pos_curr[1:] - pos_hole[1:]) < 0.03:
                self.current_stage = 4
            return pos_hole + np.array([0.4, 0.0, 0.0]), desired_r
        
        # 阶段4: 执行插入
        if self.current_stage == 4:
            print("Stage 4: Inserting peg")
            return pos_hole - np.array([0.8, 0.0, 0.0]), desired_r
            
        return pos_curr, o_d["hand_quat"]
        
    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        # 放宽Z轴的抓取条件，避免在下降过程中过早判断为远离
        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04 or abs(pos_curr[2] - pos_peg[2]) > 0.1:
            return -1.0
        else:
            return 0.6

# --- Main Execution Block ---
if __name__ == "__main__":
    env_name = 'peg-insert-side-v3'
    env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
    env = env_class(render_mode='human', width=1280, height=720)

    benchmark = ppo_test.MT1(env_name)
    policy = CorrectedPolicyV2()

    num_episodes = 5
    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)
        
        obs, info = env.reset()

        # 在重置策略时，传入peg的初始位置
        policy.reset(obj_init_pos=env.obj_init_pos)

        done = False
        count = 0
        while count < 500 and not done:
            env.render()
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            if info.get('success', 0.0) > 0.5:
                print("任务成功！")
                time.sleep(1)
                break

            time.sleep(0.01)
            count += 1
        
        print(f"Episode finished. Final Info: {info}")

    env.close()