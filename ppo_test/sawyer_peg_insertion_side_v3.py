# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from ppo_test.asset_path_utils import full_V3_path_for
from ppo_test.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from ppo_test.utils import reward_utils

quat_box = Rotation.from_euler('xyz', [0, 0, 90+15], degrees=True).as_quat()[[3,0, 1, 2]]

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

        self.obj_init_pos = np.array([0, 0.6, 0.02])
        self.hand_init_pos = np.array([0, 0.6, 0.2])
        self.hand_init_quat = Rotation.from_euler('xyz', [0,45,0], degrees=True).as_quat()[[1, 2, 3, 0]]
        
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0.03, 0.0, 0.13]),
            np.array(goal_high) + np.array([0.03, 0.0, 0.13]),
            dtype=np.float64,
        )

        self.liftThresh = 0.11

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")

    def get_peg_contact_wrench(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        使用 data.cfrc_ext 获取施加在 peg 物体上的纯净外部接触力与力矩。
        这些值已经补偿了惯性力和重力。
        
        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
            一个元组，包含：
            - contact_force (3,): 在世界坐标系下，作用于peg质心的总接触力。
            - contact_torque (3,): 在世界坐标系下，作用于peg质心的总接触力矩。
        """
        # 获取 peg 物体的 ID
        peg_body_id = self.data.body("peg").id
        
        # data.cfrc_ext 是一个 (nbody, 6) 的数组
        # 每一行包含一个物体的 [力(3), 力矩(3)]
        # 我们通过 body ID 来索引
        wrench = self.data.cfrc_ext[peg_body_id]
        
        contact_force = wrench[:3]
        contact_torque = wrench[3:]
        
        return contact_force.copy(), contact_torque.copy()

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

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        # obs structure: hand_pos(3) + quat_hand(4) + gripper_distance_apart(1) + force(3) + torque(3) + peg_pos(3) + peg_rot(4) + unused_info(7) + _prev_obs(28) + goal_pos(3) = 59
        obj = obs[14:17]

        assert self._target_pos is not None and self.obj_init_pos is not None
        
        obj_head = self._get_site_pos("pegHead")
        
        tcp_opened: float = obs[7] 
        tcp = self.tcp_center
        
        target = self._target_pos
        
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        
        reward, tcp_open, obj_to_target, grasp_reward, in_place_reward = self.compute_reward(action, obj, tcp_opened)
        
        assert self.obj_init_pos is not None
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
        
        success = float(
            obj_to_target <= 0.03
        )
            
        near_object = float(tcp_to_obj <= 0.03)
        
        peg_force, peg_torque = self.get_peg_contact_wrench()
        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "pegHead_force": peg_force,
            "pegHead_torque": peg_torque
        }

        return reward, info

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
        self._target_pos = pos_box + np.array([0.03, 0.0, 0.13])
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.get_body_com("peg").copy()[2]
                
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obj: npt.NDArray[Any], tcp_opened: npt.NDArray[Any]
    ) -> tuple[float, float, float, float, float, float, float, float]:

        target = self._target_pos
        tcp = self.tcp_center
        obj_head = self._get_site_pos("pegHead")
        scale = np.array([1.0, 2.0, 2.0])
        
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        obj_to_target = float(np.linalg.norm((obj_head - target) * scale))
        
        in_place_margin = float(
            np.linalg.norm((self.peg_head_pos_init - target) * scale)
        )
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        brc_col_box_1 = self._get_site_pos("bottom_right_corner_collision_box_1")
        tlc_col_box_1 = self._get_site_pos("top_left_corner_collision_box_1")

        brc_col_box_2 = self._get_site_pos("bottom_right_corner_collision_box_2")
        tlc_col_box_2 = self._get_site_pos("top_left_corner_collision_box_2")
        collision_box_bottom_1 = reward_utils.rect_prism_tolerance(
            curr=obj_head, one=tlc_col_box_1, zero=brc_col_box_1
        )
        collision_box_bottom_2 = reward_utils.rect_prism_tolerance(
            curr=obj_head, one=tlc_col_box_2, zero=brc_col_box_2
        )
        collision_boxes = reward_utils.hamacher_product(
            collision_box_bottom_2, collision_box_bottom_1
        )
        in_place = reward_utils.hamacher_product(in_place, collision_boxes)

        pad_success_margin = 0.03
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.0075

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_margin,
            xz_thresh=x_z_margin,
            high_density=True,
        )
        if (
            tcp_to_obj < 0.08
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        ):
            object_grasped = 1.0
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, in_place
        )
        reward = in_place_and_object_grasped

        if (
            tcp_to_obj < 0.08
            and (tcp_opened > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        ):
            reward += 1.0 + 5 * in_place

        if obj_to_target <= 0.07:
            reward = 10.0

        return (
            reward,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        )

    def get_hole_info(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """获取hole的位置和朝向信息"""
        hole_pos = self._get_site_pos("hole")
        # 假设hole的朝向与box一致，这里简化处理
        box_quat = self.data.body("box").xquat
        return hole_pos, box_quat

    def get_insertion_info(self) -> dict[str, Any]:
        """获取插入相关信息"""
        hole_pos, hole_quat = self.get_hole_info()
        peg_head_pos = self._get_site_pos("pegHead")
        peg_quat = self._get_quat_objects()
        
        # 计算插入深度
        insertion_depth = np.dot(peg_head_pos - hole_pos, np.array([-1, 0, 0]))  # 假设hole朝向为-y方向
        
        return {
            "hole_pos": hole_pos,
            "hole_quat": hole_quat,
            "peg_head_pos": peg_head_pos,
            "peg_quat": peg_quat,
            "insertion_depth": max(0, insertion_depth),
        }