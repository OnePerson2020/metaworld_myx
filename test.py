import matplotlib.pyplot as plt
import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random
import pandas as pd
from collections import deque
import mujoco

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed

# --- 全局变量用于暂停功能 ---
is_paused = False
d_raw = 17

def key_callback(key):
    """用于处理键盘输入的函数"""
    global is_paused
    if key == mujoco.mjKEY_SPACE:
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "RUNNING"
        print(f"\n--- Simulation {status} ---")

class ActionVisualizer:
    """一个用于动态可视化action变量的类"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        self.fig.suptitle('Real-time Action Visualization')
        self.reset()
        self.axs[0].set_ylabel('Delta Position (m)')
        self.pos_lines = [self.axs[0].plot([], [], label=label)[0] for label in ['dx', 'dy', 'dz']]
        self.axs[0].legend(loc='upper left')
        self.axs[0].grid(True)
        self.axs[1].set_ylabel('Delta Rotation (deg)')
        self.rot_lines = [self.axs[1].plot([], [], label=label)[0] for label in ['roll', 'pitch', 'yaw']]
        self.axs[1].legend(loc='upper left')
        self.axs[1].grid(True)
        self.axs[2].set_ylabel('Gripper Effort')
        self.gripper_line = self.axs[2].plot([], [], label='effort', color='purple')[0]
        self.axs[2].legend(loc='upper left')
        self.axs[2].grid(True)
        self.axs[2].set_xlabel('Time Step')
        plt.show(block=False)
        self.fig.canvas.draw()

    def reset(self):
        """重置图表数据，用于开始新的episode"""
        self.timesteps = deque(maxlen=self.window_size)
        self.pos_data = [deque(maxlen=self.window_size) for _ in range(3)]
        self.rot_data = [deque(maxlen=self.window_size) for _ in range(3)]
        self.gripper_data = deque(maxlen=self.window_size)
        self.current_step = 0

    def update(self, delta_pos, delta_rot_euler, gripper_effort):
        """使用新的动作数据更新图表"""
        self.timesteps.append(self.current_step)
        for i in range(3):
            self.pos_data[i].append(delta_pos[i])
            self.rot_data[i].append(delta_rot_euler[i])
        self.gripper_data.append(gripper_effort)

        for i in range(3):
            self.pos_lines[i].set_data(self.timesteps, self.pos_data[i])
            self.rot_lines[i].set_data(self.timesteps, self.rot_data[i])
        self.gripper_line.set_data(self.timesteps, self.gripper_data)

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def step(self):
        """推进step计数器"""
        self.current_step += 1

    def close(self):
        """关闭图表窗口"""
        plt.ioff()
        plt.close(self.fig)


class CorrectedPolicyV2(Policy):

    def __init__(self):

        super().__init__()
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)

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
        delta_pos = self._calculate_pos_action(o_d["hand_pos"], to_xyz=desired_pos)
        
        delta_rot_euler = self._calculate_rotation_action(o_d["hand_quat"], desired_r)
        gripper_effort = self._grab_effort(o_d)
        
        # if self.current_stage == 4 and np.linalg.norm(o_d["pegHead_force"]) > 5:
        #     delta_pos, delta_rot_euler = self._pos_im(o_d["pegHead_force"], o_d["pegHead_torque"], delta_pos, delta_rot_euler)
        
        # 将数据传递给可视化器
        if 'visualizer' in globals() and visualizer is not None:
             visualizer.update(delta_pos, delta_rot_euler, gripper_effort)

        delta_rot = Rotation.from_euler('xyz', delta_rot_euler, degrees=True)
        delta_rot_quat = delta_rot.as_quat()
        action = Action(8)
        action.set_action(np.hstack((delta_pos, delta_rot_quat, gripper_effort)))
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        pos_curr, pos_peg, pos_hole, gripper_distance = o_d["hand_pos"], o_d["peg_pos"], o_d["goal_pos"], o_d["gripper_distance_apart"]
        
        if self.current_stage == 1:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04: self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), self.ini_r
        if self.current_stage == 2:
            if pos_curr[2] - pos_peg[2] < 0.01 and gripper_distance < 0.45: print(">>> Peg lifted! Transitioning to Stage 3."); self.current_stage = 3
            return pos_peg - np.array([0.0, 0.0, 0.02]), self.ini_r
        if self.current_stage == 3:
            desir_3_pos = self.compute_peg_tip_position(pos_hole, [0,0,d_raw], L=0.32)
            if np.linalg.norm(pos_curr[1:] - desir_3_pos[1:]) < 0.03: self.current_stage = 4
            return desir_3_pos, self.ini_r
        if self.current_stage == 4:
            return self.compute_peg_tip_position(pos_hole, [0,0,d_raw], L=0.08), Rotation.from_euler('xyz', [-d_raw, 90, 0], degrees=True)
        return None, None
    
    def _calculate_pos_action(self, from_xyz: npt.NDArray[any], to_xyz: npt.NDArray[any], speed: float = 0.2) -> npt.NDArray[any]:
        error_vec = to_xyz - from_xyz; Kp = 0.3; Kd = 0.5
        error_vec_pd = Kp * error_vec + Kd * self.prev_pos_error
        distance = np.linalg.norm(error_vec_pd); max_dist_per_step = speed * 0.0125
        if distance < 1e-6: return np.zeros(3)
        if distance < max_dist_per_step: return error_vec_pd
        direction = error_vec / distance; delta_pos = direction * max_dist_per_step
        self.prev_pos_error = error_vec
        return delta_pos
    def _pos_im(self, force, torque, delta_pos, delta_rot_euler):
        dt = 0.0125; M_d_inv = np.diag([0, 0.01, 0.01, 0, 0.1, 0.1])
        D_d = np.diag([0, 0.3, 0.3, 0, 0.5, 0.5]); K_d = np.diag([0, 0.4, 0.4, 0., 0.8, 0.8])
        F_ext = np.clip(np.concatenate([force, torque]), -50, 50)
        E_ddot = M_d_inv @ (F_ext - D_d @ self.e_dot_im - K_d @ self.e_im)
        self.e_dot_im += E_ddot * dt; self.e_im += self.e_dot_im * dt
        limit_e = 0.2 * 0.0125
        self.e_im[:3] = np.clip(self.e_im[:3], -limit_e, limit_e)
        self.e_im[3:] = np.clip(self.e_im[3:], -np.deg2rad(5), np.deg2rad(5))
        delta_pos = self.e_im[0:3]; delta_rot_euler = np.rad2deg(self.e_im[3:6])
        return delta_pos, delta_rot_euler
    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
        if target_Rotation is None: return np.zeros(3)
        kp=0.1; kd=0.8; speed=np.deg2rad(0.8)
        r_curr = Rotation.from_quat(current_quat_mujoco[[1, 2, 3, 0]])
        r_error = target_Rotation * r_curr.inv()
        error_rotvec = r_error.as_rotvec()
        if np.linalg.norm(error_rotvec) < 1e-8: self.prev_r_error_rotvec = np.zeros(3); return np.zeros(3)
        rotation_axis = error_rotvec / np.linalg.norm(error_rotvec)
        unconstrained_increment_rotvec = kp * error_rotvec + kd * (error_rotvec - self.prev_r_error_rotvec)
        self.prev_r_error_rotvec = error_rotvec
        speed_of_increment = np.linalg.norm(unconstrained_increment_rotvec)
        increment_rotvec = rotation_axis * speed if speed_of_increment > speed else unconstrained_increment_rotvec
        if np.linalg.norm(increment_rotvec) < 1e-6: return np.zeros(3)
        r_increment = Rotation.from_rotvec(increment_rotvec)
        return r_increment.as_euler('xyz', degrees=True)
    def _grab_effort(self, o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr, pos_peg = o_d["hand_pos"], o_d["peg_pos"]
        if not self.gasp:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04 and (pos_curr[2] - pos_peg[2]) < 0.01: self.gasp = True; return 0.4
            return -1.0
        return 0.8

    def compute_peg_tip_position(self,peg_head_pos, peg_euler, L):
        """
        根据钉头位置、钉子旋转四元数和长度，计算钉子末端（插入端）的位置。

        Args:
            peg_head_pos (np.ndarray): 钉头位置, shape (3,)
            peg_quat (np.ndarray): 钉子的旋转四元数 [w, x, y, z]
            L (float): 钉子长度

        Returns:
            np.ndarray: 钉子末端（插入端）在世界坐标系中的位置, shape (3,)
        """
        # 将四元数转换为 Rotation 对象
        # 注意：scipy 的 Rotation 使用 [x, y, z, w] 顺序
        r = Rotation.from_euler('xyz', peg_euler, degrees=True)

        # 定义在 peg 本地坐标系中，从头部指向末端的单位方向向量
        # 假设 peg 初始沿 -z 方向延伸
        local_tip_direction = np.array([1, 0, 0])  # 从头部指向末端

        # 将该方向向量旋转到世界坐标系
        world_tip_direction = r.apply(local_tip_direction)

        # 计算末端位置 = 钉头位置 + 方向向量 * 长度
        peg_tip_pos = peg_head_pos + L * world_tip_direction

        return peg_tip_pos

if __name__ == "__main__":
    env_name = 'peg-insert-side-v3'
    env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
    env = env_class(render_mode='human', width=1000, height=720)
    
    
    benchmark = ppo_test.MT1(env_name)
    policy = CorrectedPolicyV2()
    
    visualizer = ActionVisualizer(window_size=100)
    
    UPDATE_VISUALIZER_EVERY_N_STEPS = 5 # 设置为1则每步都画，数字越大仿真越快
    
    all_force_data = []
    num_episodes = 3
    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        task = random.choice(benchmark.train_tasks)
        env.set_task(task)
        
        obs, info = env.reset()
        policy.reset()
        visualizer.reset() 
        env.mujoco_renderer.viewer.key_callback = key_callback
        env.mujoco_renderer.viewer.cam.azimuth = 245
        env.mujoco_renderer.viewer.cam.elevation = -20
        
        episode_forces = []
        done = False
        count = 0
        while count < 500 and not done:
            # --- 关键改动：检查暂停状态 ---
            while is_paused:
                env.render() # 暂停时也要持续渲染，否则窗口会无响应
                time.sleep(0.1) # 避免CPU空转

            env.render()
            
            # 从策略中获取action（不再在policy内部更新图表）
            o_d = policy._parse_obs(obs)
            desired_pos, desired_r = policy._desired_pose(o_d)
            if policy.current_stage == 4:
                delta_pos = policy._calculate_pos_action(o_d["hand_pos"], to_xyz=desired_pos, speed= 0.1)
                delta_rot_euler = policy._calculate_rotation_action(o_d["hand_quat"], desired_r)
                # if np.linalg.norm(o_d["pegHead_force"]) > 1:
                #     delta_pos, delta_rot_euler = policy._pos_im(o_d["pegHead_force"], o_d["pegHead_torque"], delta_pos, delta_rot_euler)
            else:
                delta_pos = policy._calculate_pos_action(o_d["hand_pos"], to_xyz=desired_pos)
                delta_rot_euler = policy._calculate_rotation_action(o_d["hand_quat"], desired_r)
                
            gripper_effort = policy._grab_effort(o_d)
            
            visualizer.step()
            if count % UPDATE_VISUALIZER_EVERY_N_STEPS == 0:
                visualizer.update(delta_rot_euler, o_d["pegHead_torque"], gripper_effort)

            delta_rot = Rotation.from_euler('xyz', delta_rot_euler, degrees=True)
            delta_rot_quat = delta_rot.as_quat()
            action = Action(8)
            action.set_action(np.hstack((delta_pos, delta_rot_quat, gripper_effort)))
            obs, reward, terminated, truncated, info = env.step(action.array.astype(np.float32))
            
            force = info.get('pegHead_force', np.zeros(3)); force_magnitude = np.linalg.norm(force)
            episode_forces.append({'step': count, 'magnitude': force_magnitude, 'direction_x': force[0], 'direction_y': force[1], 'direction_z': force[2]})
            done = terminated or truncated
            if info.get('success', 0.0) > 0.5: print("任务成功！"); break
            count += 1
        
        for data_point in episode_forces: data_point['episode'] = i + 1
        all_force_data.extend(episode_forces)
        print(f"Episode finished. Final Info: {info}")

    print("All episodes finished. Closing environment.")
    env.close()
    visualizer.close()
    
    df = pd.DataFrame(all_force_data)
    df.to_csv("force_analysis.csv", index=False)
    print("\nForce analysis data saved to force_analysis.csv")

    try:
        from visualize_forces import visualize_force_data
        visualize_force_data("force_analysis.csv")
    except ImportError:
        print("Skipping force visualization: 'visualize_forces.py' not found.")
    except Exception as e:
        print(f"An error occurred during force visualization: {e}")

    print("\nSimulation finished. Close the plot window to exit.")
    plt.ioff()
    plt.show()