# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any, Tuple

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from ppo_test.asset_path_utils import full_V3_path_for
from ppo_test.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from ppo_test.utils import reward_utils

box_raw = 0
quat_box = Rotation.from_euler('xyz', [0, 0, 90+box_raw], degrees=True).as_quat()[[3,0, 1, 2]]

class SawyerPegInsertionSideEnvV3(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.07

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
    ) -> None:

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.5, 0.02)
        obj_high = (0.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            height=height,
            width=width,
        )

        self.goal = np.array([-0.3, 0.6, 0.0])

        self.hand_init_pos = np.array([0, 0.6, 0.2])
        self.hand_init_quat = Rotation.from_euler('xyz', [0,90,0], degrees=True).as_quat()[[3, 0, 1, 2]]
        
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13])),
            np.array(goal_high) + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13])),
            dtype=np.float64,
        )

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")
    
    def get_peghead_force_and_torque(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        计算并返回 pegHead_geom 在世界坐标系下受到的总接触力和相对于 pegGrasp 点的总力矩。

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
            一个元组，包含：
            - total_world_force (3,): 世界坐标系下的总受力向量。
            - total_world_torque (3,): 世界坐标系下，相对于 pegGrasp 点的总力矩向量。
        """
        # --- 初始化 ---
        peg_head_geom_id = self.data.geom("pegHead_geom").id
        total_world_force = np.zeros(3)
        total_world_torque = np.zeros(3)

        # try:
        #     grasp_point_world = self.data.site("pegGrasp").xpos
        # except KeyError:
        #     grasp_point_world = self.data.body("peg").xpos
        grasp_point_world = self.data.body("peg").xpos
        
        # --- 遍历所有接触点 ---
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 检查当前接触是否涉及 pegHead_geom
            if contact.geom1 == peg_head_geom_id or contact.geom2 == peg_head_geom_id:
                
                # 步骤 1: 获取在“接触坐标系”下的6D力/力矩向量
                force_contact_frame = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)
                
                # 步骤 2: 将接触力从“接触坐标系”旋转到“世界坐标系”
                contact_frame_rot = contact.frame.reshape(3, 3)
                force_world_frame = contact_frame_rot @ force_contact_frame[:3]
                
                # 步骤 3: 根据牛顿第三定律确定力的正确方向
                if contact.geom1 == peg_head_geom_id:
                    # 如果 geom1 是我们的传感器，力是由它施加的，我们需要反向的力
                    force_on_peghead = -force_world_frame
                else: # contact.geom2 == peg_head_geom_id
                    # 如果 geom2 是我们的传感器，力是施加于它的，方向正确
                    force_on_peghead = force_world_frame
                
                # --- 力矩计算 ---
                # 步骤 4: 获取接触点在世界坐标系下的位置
                contact_position_world = contact.pos
                
                # 步骤 5: 计算从抓取点到接触点的矢量（力臂）
                lever_arm = contact_position_world - grasp_point_world
                
                # 步骤 6: 计算该接触力产生的力矩 (tau = r x F)
                torque_i = np.cross(lever_arm, force_on_peghead)
                
                # --- 累加总力和总力矩 ---
                total_world_force += force_on_peghead
                total_world_torque += torque_i
        
        return total_world_force, total_world_torque

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self._get_site_pos("pegGrasp")

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        geom_xmat = self.data.site("pegGrasp").xmat.reshape(3, 3)
        return Rotation.from_matrix(geom_xmat).as_quat()

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        self.obj_init_pos = pos_peg
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        self._set_obj_xyz(self.obj_init_pos)
        self.model.body("box").pos = pos_box
        self.model.body("box").quat = quat_box
        self._goal_pos = pos_box + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13]))
        
        self.model.site("goal").pos = self._goal_pos

        self.objHeight = self.get_body_com("peg").copy()[2]
                
        return self._get_obs()

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[14:17]
        assert self._goal_pos is not None and self.obj_init_pos is not None        
        tcp_open: float = obs[7] 
        tcp = self.tcp_center
                
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        
        # 使用优化后的奖励计算
        reward, stage_rewards = self.compute_reward_test(action, obj)
        
        # 获取插入信息
        insertion_info = self.get_insertion_info()
        
        assert self.obj_init_pos is not None
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
        
        # 成功条件：插入深度达到目标
        success = float(insertion_info["insertion_depth"] >= 0.1)  # 5cm插入深度
        # success =  float(stage_rewards["approach"] == 1)
                    
        info = {
            "success": success,
            "grasp_success": grasp_success,
            "stage_rewards": stage_rewards,
            "insertion_depth": insertion_info["insertion_depth"],
            "unscaled_reward": reward,
        }

        return reward, info

    def compute_reward_test(
        self, action: npt.NDArray[Any], obj: npt.NDArray[Any]
    ) -> tuple[float, dict[str, float]]:
        """
        优化的分阶段奖励函数
        阶段1: 接近peg
        阶段2: 抓取peg
        阶段3: 对准hole
        阶段4: 插入
        """
        
        tcp = self.tcp_center
        obs = self._get_obs()
        tcp_opened = obs[7]
        
        # 获取关键位置
        obj_head = self._get_site_pos("pegHead")
        insertion_info = self.get_insertion_info()
        hole_pos = insertion_info["hole_pos"]
        hole_orientation = insertion_info["hole_orientation"]
        insertion_depth = insertion_info["insertion_depth"]

        # 初始化任务阶段状态（如果不存在）
        if not hasattr(self, 'task_phase'):
            self.task_phase = 'approach'
            self.max_alignment_achieved = 0.0
            self.insertion_started = False
        # ========== 阶段1: 接近Peg ==========
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        approach_margin = float(np.linalg.norm(self.obj_init_pos - self.hand_init_pos))
        
        approach_reward = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.02),  # 目标是接近到2cm以内
            margin=approach_margin,
            sigmoid="long_tail"
        )
        
        # ========== 阶段2: 抓取Peg ==========
        # 使用现有的gripper caging reward函数
        grasp_reward = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.0075,
            pad_success_thresh=0.03,
            xz_thresh=0.005,
            high_density=True,
        )
        
        # 检查是否已经抓取成功
        obj_lifted = obj[2] - self.obj_init_pos[2] > 0.01  # 物体抬起超过1cm
        if tcp_to_obj < 0.08 and tcp_opened > 0 and obj_lifted:
            grasp_reward = 1.0
            if self.task_phase == 'approach':
                self.task_phase = 'grasp'

        # ========== 阶段3: 对准Hole ==========
        # 计算peg head到hole入口的距离
        head_to_hole = obj_head - hole_pos
        
        # 横向对准（垂直于hole方向的距离）
        lateral_offset = head_to_hole - np.dot(head_to_hole, hole_orientation) * hole_orientation
        lateral_distance = np.linalg.norm(lateral_offset)
        
        # 纵向对准（沿hole方向，但在hole前方）
        longitudinal_distance = np.dot(head_to_hole, hole_orientation)
        
        # 计算当前对准质量
        current_alignment = 0.0
        if grasp_reward > 0.8:  # 只有抓取成功后才计算对准
            # 横向对准奖励
            lateral_alignment = reward_utils.tolerance(
                lateral_distance,
                bounds=(0, 0.005),  # 横向误差小于5mm
                margin=0.1,
                sigmoid="long_tail"
            )
            
            # 纵向位置奖励（在hole前方0-3cm的位置最佳）
            longitudinal_alignment = reward_utils.tolerance(
                abs(longitudinal_distance),
                bounds=(0, 0.03),  
                margin=0.1,
                sigmoid="long_tail"
            )
            
            current_alignment = reward_utils.hamacher_product(
                lateral_alignment, 
                longitudinal_alignment
            )
            
            # 更新最大对准值
            self.max_alignment_achieved = max(self.max_alignment_achieved, current_alignment)
            
            # 状态转换：进入对准阶段
            if self.task_phase == 'grasp' and current_alignment > 0.5:
                self.task_phase = 'alignment'
        
        # 对准奖励（根据阶段决定）
        if not self.insertion_started:
            # 未开始插入时，使用当前对准值
            alignment_reward = current_alignment
        else:
            # 已开始插入，使用历史最大值，避免插入时的下降影响
            alignment_reward = self.max_alignment_achieved
        
        # ========== 阶段4: 插入 ==========
        insertion_reward = 0.0
        
        # 检查是否可以开始插入
        if self.max_alignment_achieved > 0.7 or self.insertion_started:
            # 一旦开始插入，就保持插入状态
            if insertion_depth > 0.001:  # 检测到开始插入（1mm以上）
                self.insertion_started = True
                self.task_phase = 'insertion'
            
            if self.insertion_started:
                # 插入深度奖励
                insertion_reward = reward_utils.tolerance(
                    insertion_depth,
                    bounds=(0.10, 0.10),  # 目标插入深度5-10cm
                    margin=0.1,
                    sigmoid="gaussian"
                )
                
                # 插入过程中的对准保持奖励（独立计算，不影响alignment_reward）
                insertion_alignment_bonus = 0.0
                if lateral_distance < 0.01:  # 插入时的对准容差可以更宽松
                    insertion_alignment_bonus = 0.2 * (1.0 - lateral_distance / 0.01)
                insertion_reward = min(1.0, insertion_reward + insertion_alignment_bonus)
        
        stage_weights = {"approach": 1, "grasp": 0, "alignment": 0, "insertion": 0}
            
        
        # 计算加权总奖励
        total_reward = (
            stage_weights["approach"] * approach_reward +
            stage_weights["grasp"] * grasp_reward +
            stage_weights["alignment"] * alignment_reward +
            stage_weights["insertion"] * insertion_reward
        ) / sum(stage_weights.values())
        
        # 成功奖励
        if insertion_depth >= 0.05:
            total_reward = 10.0
        
        # 阶段奖励字典（用于调试和监控）
        stage_rewards = {
            "approach": approach_reward,
            "grasp": grasp_reward,
            "alignment": alignment_reward,
            "insertion": insertion_reward,
            "lateral_distance": lateral_distance,
            "insertion_depth": insertion_depth,
            "task_phase": self.task_phase,
            "max_alignment": self.max_alignment_achieved,
            "insertion_started": self.insertion_started
        }

        
        labels = ["App", "Grasp", "Align", "Insert", "Lat", "Long", "Depth"]
        values = [approach_reward, grasp_reward, alignment_reward, insertion_reward, lateral_distance, longitudinal_distance, insertion_depth]
        print(" ".join(f"{v:6.3f}" for v in values))
        print(" ".join(f"{l:^6}" for l in labels))  # 可选：打印标签行

        return total_reward, stage_rewards

    def get_hole_info(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """获取hole的位置和朝向信息。
        朝向由从 'hole' 站点指向 'goal' 站点的单位向量表示。
        """
        hole_pos = self._get_site_pos("hole")
        goal_pos = self._get_site_pos("goal")

        # 计算从 hole 指向 goal 的向量
        orientation_vec = goal_pos - hole_pos
        
        hole_orientation = orientation_vec / np.linalg.norm(orientation_vec)
            
        return hole_pos, hole_orientation

    def get_insertion_info(self) -> dict[str, Any]:
        """获取插入相关信息"""
        hole_pos, hole_orientation = self.get_hole_info() 
        peg_head_pos = self._get_site_pos("pegHead")
        
        # 计算插入深度：将 (peg头 - hole口) 的向量，投影到 hole 的朝向向量上
        insertion_depth = np.dot(peg_head_pos - hole_pos, hole_orientation)
        
        return {
            "hole_pos": hole_pos,
            "hole_orientation": hole_orientation,
            "peg_head_pos": peg_head_pos,
            "insertion_depth": max(0, insertion_depth),
        }