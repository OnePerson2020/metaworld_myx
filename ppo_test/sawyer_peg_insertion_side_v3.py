# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from ppo_test.asset_path_utils import full_V3_path_for
from ppo_test.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from ppo_test.types import InitConfigDict
from ppo_test.utils import reward_utils


class SawyerPegInsertionSideEnvV3(SawyerXYZEnv):
    TARGET_RADIUS: float = 0.07

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        reward_function_version: str | None = None,
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
            reward_function_version=reward_function_version,
        )

        self.init_config: InitConfigDict = {
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.3, 0.6, 0.0])

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

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
        self.insertion_phase = "approach"  # approach, align, insert

        self.pegHead_force_id = self.model.sensor("pegHead_force").id

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")

    def _get_pegHead_force(self) -> npt.NDArray[np.float64]:
        """获取pegHead所受的力 (Fx, Fy, Fz)."""
        # MuJoCo force sensors output a 3D vector
        return self.data.sensordata[self.pegHead_force_id : self.pegHead_force_id + 3]

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        # Updated observation parsing for new 53-dimensional observation space
        # obs structure: hand_pos(3) + quat_hand(4) + gripper_distance_apart(1) + peg_pos(3) + peg_rot(4) + force(3)  + unused_info(7) + _prev_obs(25) + goal_pos(3) = 53
        obj = obs[8:11]  # peg_pos, index changed from 4:7 to 8:11

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            collision_box_front,
            ip_orig,
        ) = self.compute_reward(action, obs)
        assert self.obj_init_pos is not None
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        # Get pegHead force, magnitude, and direction
        pegHead_force = self._get_pegHead_force()
        force_magnitude = np.linalg.norm(pegHead_force)
        # Avoid division by zero if force is (0,0,0)
        force_direction = pegHead_force / (force_magnitude + 1e-8) if force_magnitude > 1e-8 else np.zeros_like(pegHead_force)


        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "insertion_phase": self.insertion_phase,
            "pegHead_force": pegHead_force,             # Raw force vector
            "pegHead_force_magnitude": force_magnitude, # Force magnitude
            "pegHead_force_direction": force_direction, # Force direction (unit vector)
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
        self._target_pos = pos_box + np.array([0.03, 0.0, 0.13])
        self.model.site("goal").pos = self._target_pos

        self.objHeight = self.get_body_com("peg").copy()[2]
        self.heightTarget = self.objHeight + self.liftThresh

        # 重置插入阶段
        self.insertion_phase = "approach"
                
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        tcp = self.tcp_center
        # Updated observation parsing for new 53-dimensional observation space
        obj = obs[8:11]  # peg_pos
        obj_head = self._get_site_pos("pegHead")
        tcp_opened: float = obs[7]  # gripper_distance_apart, index changed from 3 to 7
        target = self._target_pos
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        scale = np.array([1.0, 2.0, 2.0])
        #  force agent to pick up object then insert
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
        ip_orig = in_place
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
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
            collision_boxes,
            ip_orig,
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
        insertion_depth = np.dot(peg_head_pos - hole_pos, np.array([0, -1, 0]))  # 假设hole朝向为-y方向
        
        return {
            "hole_pos": hole_pos,
            "hole_quat": hole_quat,
            "peg_head_pos": peg_head_pos,
            "peg_quat": peg_quat,
            "insertion_depth": max(0, insertion_depth),
            "insertion_phase": self.insertion_phase
        }