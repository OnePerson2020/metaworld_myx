# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any, Tuple, Callable, SupportsFloat

import pickle
import mujoco
import numpy as np
import numpy.typing as npt
from pathlib import Path

from scipy.spatial.transform import Rotation

from ppo_test.sawyer_xyz_env import SawyerMocapBase
from ppo_test.types import Task, XYZ, RenderMode
from ppo_test import reward_utils

from gymnasium.spaces import Box
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle


box_raw = 0
quat_box = Rotation.from_euler('xyz', [0, 0, 90+box_raw], degrees=True).as_quat()[[3,0, 1, 2]]

Len_observation: int = 21

_HAND_POS_SPACE = Box(
    
    np.array([-0.525, 0.348, -0.0525]),
    np.array([+0.525, 1.025, 0.7]),
    dtype=np.float64,
)
"""Bounds for hand position."""

_HAND_QUAT_SPACE = Box(
    np.array([-1.0, -1.0, -1.0, -1.0]),
    np.array([1.0, 1.0, 1.0, 1.0]),
    dtype=np.float64,
)
"""Bounds for hand quaternion."""

TARGET_RADIUS: float = 0.05
"""Upper bound for distance from the target when checking for task completion."""

class _Decorators:
    @classmethod
    def assert_task_is_set(cls, func: Callable) -> Callable:
        """Asserts that the task has been set in the environment before proceeding with the function call.
        To be used as a decorator for SawyerPegInsertionSideEnvV4 methods."""

        def inner(*args, **kwargs) -> Any:
            env = args[0]
            if not env._set_task_called:
                raise RuntimeError(
                    "You must call env.set_task before using env." + func.__name__
                )
            return func(*args, **kwargs)

        return inner
        
class SawyerPegInsertionSideEnvV4(SawyerMocapBase, EzPickle):
    
    max_path_length: int = 300
    """The maximum path length for the environment (the task horizon)."""

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        frame_skip: int = 5,
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        print_flag: bool = False,
        pos_action_scale = 0.01,
    ) -> None:

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.5, 0.02)
        obj_high = (0.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

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
    
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, [-1.0, -1.0, -1.0]))
        self.mocap_high = np.hstack((mocap_high, [1.0, 1.0, 1.0]))
        
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        self._partially_observable: bool = False

        self._set_task_called: bool = False
        self.print_flag = print_flag
        self.pos_action_scale = pos_action_scale

        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )

        mujoco.mj_forward(
            self.model, self.data
        )  # *** DO NOT REMOVE: EZPICKLE WON'T WORK *** #

        self.action_space = Box(  # type: ignore
            np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            np.array([+1.0, +1.0, +1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._prev_obs = np.zeros(Len_observation, dtype=np.float64)

        # 任务阶段初始化
        self.task_phase = 'approach'
        
        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            render_mode,
            camera_id,
            camera_name,
            width,
            height,
        )
        
    @property
    def model_name(self) -> str:
        _CURRENT_FILE_DIR = Path(__file__).parent.absolute()
        ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / 'xml'
        file_name = "sawyer_peg_insertion_side.xml"
        return str(ENV_ASSET_DIR_V3 / file_name)

    @property
    def sawyer_observation_space(self) -> Box:
        obj_low = np.full(7, -np.inf, dtype=np.float64)
        obj_high = np.full(7, +np.inf, dtype=np.float64)
        
        if self._partially_observable:
            goal_low = np.zeros(3)
            goal_high = np.zeros(3)
        else:
            assert (
                self.goal_space is not None
            ), "The goal space must be defined to use full observability"
            goal_low = self.goal_space.low
            goal_high = self.goal_space.high
            
        gripper_low = -1.0
        gripper_high = +1.0
        
        force_low = np.full(3, -20, dtype=np.float64)
        force_high = np.full(3, 20, dtype=np.float64)

        torque_low = np.full(3, -20, dtype=np.float64)
        torque_high = np.full(3, 20, dtype=np.float64)

        # return Box(
        #     np.hstack(
        #         (
        #             _HAND_POS_SPACE.low, 
        #             gripper_low,
        #             obj_low,
        #             _HAND_POS_SPACE.low, 
        #             gripper_low,
        #             obj_low,
        #             goal_low,
        #         )
        #     ),
        #     np.hstack(
        #         (
        #             _HAND_POS_SPACE.high,
        #             gripper_high,
        #             obj_high,
        #             _HAND_POS_SPACE.high,
        #             gripper_high,
        #             obj_high,
        #             goal_high,
        #         )
        #     ),
        #     dtype=np.float64,
        # )
            
        # Current observation: pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + pegHead_force (3) + pegHead_torque + obs_obj_padded (7) = 21
        # Goal: 3
        return Box(
            np.hstack(
                (
                    # Current obs (21)
                    _HAND_POS_SPACE.low, 
                    _HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    # Previous obs (21)
                    _HAND_POS_SPACE.low, 
                    _HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    # Goal (3)
                    goal_low,
                )
            ),
            np.hstack(
                (
                    # Current obs (21)
                    _HAND_POS_SPACE.high,
                    _HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    # Previous obs (21)
                    _HAND_POS_SPACE.high,
                    _HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    # Goal (3)
                    goal_high,
                )
            ),
            dtype=np.float64,
        )
        
    def seed(self, seed: int) -> list[int]:
        """Seeds the environment.

        Args:
            seed: The seed to use.

        Returns:
            The seed used inside a 1 element list.
        """
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        assert self.goal_space
        self.goal_space.seed(seed)
        return [seed]

    def set_task(self, task: Task) -> None:
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._freeze_rand_vec = data.get("freeze", False)
        self.seeded_rand_vec = data.get("seeded_rand_vec", True)
        self._last_rand_vec = data.get("rand_vec", None)
        self._partially_observable =  data["partially_observable"]
        del data["partially_observable"]

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Adjusts the position of the mocap body from the given action.
        Moves each body axis in XYZ by the amount described by the action.

        Args:
            action: The action to apply (in offsets between :math:`[-1, 1]` for each axis in XYZ).
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pos_delta = self.pos_action_scale * action[:3]
        
        # 应用位移并裁剪到工作空间
        new_mocap_pos = self.data.mocap_pos.copy()
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :] + pos_delta,
            self.mocap_low[:3],
            self.mocap_high[:3],
        )
        self.data.mocap_pos = new_mocap_pos
        
        # r_increment = Rotation.from_quat(action[3:7])
        
        # current_mocap = self.data.mocap_quat[0]
        # new_mocap_r = r_increment * Rotation.from_quat(current_mocap[[1,2,3,0]])
        # new_mocap_quat = new_mocap_r.as_quat()[[3,0,1,2]]
        # self.data.mocap_quat[0] = new_mocap_quat

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

    @_Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64]
    ) -> tuple[float, dict[str, Any]]:
                                
        reward, stage_rewards = self.compute_reward_test(obs)
        insertion_info = self.get_insertion_info()
        
        success = float(insertion_info["insertion_depth"] >= 0.04)
                    
        info = {
            "success": success,
            "stage_rewards": stage_rewards,
            "unscaled_reward": reward,
        }

        return reward, info

    def compute_reward_test(
        self,
        obs: npt.NDArray[Any],
    ) -> tuple[float, dict[str, float]]:
        """
        改进版分阶段奖励函数
        阶段1: 接近 peg
        阶段2: 抓取 + 抬起
        阶段3: 对准 hole
        阶段4: 插入
        """
        tcp = obs[:3]
        tcp_opened = obs[7]
        obj = obs[14:17]
        obj_z_pos = obj[2] # 获取 peg 的 Z 轴高度

        # 获取插入信息
        insertion_info = self.get_insertion_info()
        head_to_hole = insertion_info["peg_head_pos"] - insertion_info["hole_pos"]
        hole_orientation = insertion_info["hole_orientation"]
        insertion_depth = insertion_info["insertion_depth"]

        # 初始化任务阶段状态
        if not hasattr(self, 'task_phase'):
            self.task_phase = 'approach'
            
        # --- 初始化所有阶段奖励 ---
        approach_reward = 0.0
        grasp_reward = 0.0
        lift_reward = 0.0  # 新增
        alignment_reward = 0.0
        insertion_reward = 0.0

        # ---------------------------------
        # 阶段1: 接近 Peg
        # ---------------------------------
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        if self.task_phase == 'approach':
            approach_margin = float(np.linalg.norm(self.obj_init_pos - self.init_tcp))
            approach_reward = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0.0, 0.0),
                margin=approach_margin,
                sigmoid="long_tail"
            )
            # 转换条件: 足够近就准备抓取
            if tcp_to_obj < 0.025:
                self.task_phase = 'grasp'

        # ---------------------------------
        # 阶段2: 抓取
        # ---------------------------------
        if self.task_phase == 'grasp':
            close_ness = np.clip((1 - tcp_opened) / 0.65, 0.0, 1.0)
            grasp_reward = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0.0, 0.02),
                margin=0.2, # 给一个小的容忍边界
                sigmoid="long_tail"
            )
            grasp_reward = reward_utils.hamacher_product(grasp_reward, close_ness)

            # 转换条件: 接触到物体并且夹爪闭合
            if self.touching_main_object and tcp_opened < 0.4: 
                self.task_phase = 'lift' # *** 关键修改：转换到 lift 阶段，而不是 alignment ***
        
        # ---------------------------------
        # 阶段3: 抬起 (新增)
        # ---------------------------------
        if self.task_phase == 'lift':
            target_lift_height = self.obj_init_pos[2] + 0.07 # 目标抬起高度，比如比初始高7cm
            lift_margin = 0.10 # 10cm 的奖励范围
            
            lift_reward = reward_utils.tolerance(
                obj_z_pos,
                bounds=(target_lift_height, target_lift_height + lift_margin),
                margin=lift_margin,
                sigmoid="long_tail"
            )
            
            # 转换条件: 达到目标高度
            if obj_z_pos >= target_lift_height - 0.01:
                self.task_phase = 'alignment'
        
        # ---------------------------------
        # 阶段4: 对准 Hole
        # ---------------------------------
        if self.task_phase == 'alignment':
            head_to_hole_init = self.peg_head_pos_init - self._goal_pos
            lateral_offset_init = head_to_hole_init - np.dot(head_to_hole_init, hole_orientation) * hole_orientation
            lateral_offset = head_to_hole - np.dot(head_to_hole, hole_orientation) * hole_orientation

            lateral_distance = float(np.linalg.norm(lateral_offset))
            longitudinal_distance = float(np.dot(head_to_hole, hole_orientation))

            lateral_alignment_margin = float(np.linalg.norm(lateral_offset_init))
            lateral_alignment = reward_utils.tolerance(
                lateral_distance,
                bounds=(0.0, 0.02), 
                margin=lateral_alignment_margin,
                sigmoid="long_tail",
            )

            longitudinal_alignment_margin = abs(float(np.dot(head_to_hole_init, hole_orientation)))
            longitudinal_alignment = reward_utils.tolerance(
                longitudinal_distance,
                bounds=(-0.15, 0.10),
                margin=longitudinal_alignment_margin,
                sigmoid="long_tail",
            )
            alignment_reward = reward_utils.hamacher_product(lateral_alignment, longitudinal_alignment)

            if alignment_reward > 0.9:
                self.task_phase = 'insertion'
                
        # 确保成功对准后，奖励保持为1
        if self.task_phase == 'insertion':
            alignment_reward = 1.0

        # ---------------------------------
        # 阶段5: 插入
        # ---------------------------------
        if self.task_phase == 'insertion':
            # (这个阶段的逻辑可以保持不变)
            insertion_reward = min(1.0, insertion_depth / 0.04) # 使用线性奖励
            if lateral_distance < 0.01:
                insertion_reward = min(1.0, insertion_reward + 0.2 * (1.0 - lateral_distance / 0.01))
        
        # ---------------------------------
        # 总奖励计算
        # ---------------------------------
        # 让每个阶段的奖励更加独立，避免后续阶段的0奖励影响当前阶段
        reward_by_phase = {
            "approach": approach_reward,
            "grasp": grasp_reward,
            "lift": lift_reward,
            "alignment": alignment_reward,
            "insertion": insertion_reward,
        }
        
        # 简单的权重相加
        total_reward = (
            reward_by_phase.get("approach", 0) +
            reward_by_phase.get("grasp", 0) +
            reward_by_phase.get("lift", 0) +
            reward_by_phase.get("alignment", 0) +
            2 * reward_by_phase.get("insertion", 0) # 插入阶段给予更高权重
        )
        
        # ---------------------------------
        # 调试信息
        # ---------------------------------
        stage_rewards = {
            "approach": float(approach_reward),
            "grasp": float(grasp_reward),
            "lift": float(lift_reward),
            "alignment": float(alignment_reward),
            "insertion": float(insertion_reward),
            "insertion_depth": float(insertion_depth),
            "obj_z_pos": float(obj_z_pos),
            "task_phase": self.task_phase,
        }

        if self.print_flag:
            values = [approach_reward,
                    grasp_reward,
                    alignment_reward,
                    insertion_reward,
                    insertion_depth]
            print(" ".join(f"{v:6.3f}" for v in values))
            print(self.task_phase)
            
        return float(total_reward), stage_rewards

    def get_hole_info(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """获取hole的位置和朝向信息。
        朝向由从 'hole' 站点指向 'goal' 站点的单位向量表示。
        """
        hole_pos = self._get_site_pos("hole")
        goal_pos = self._get_site_pos("goal")

        # 计算从 hole 指向 goal 的向量
        orientation_vec = goal_pos - hole_pos
        
        hole_orientation = orientation_vec / np.linalg.norm(orientation_vec)
            
        return goal_pos, hole_orientation

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

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy() # 前 9 个分别对应 7 个关节角度和 2 个夹爪的控制量
        qvel[9:15] = 0  # 一个刚体在空间中有 6 个自由度的速度：3 个线速度（dx, dy, dz）和 3 个角速度（ωx, ωy, ωz）。
        self.set_state(qpos, qvel)

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float64]:
        """Gets the position of a given site.

        Args:
            site_name: The name of the site to get the position of.

        Returns:
            Flat, 3 element array indicating site's location.
        """
        return self.data.site(site_name).xpos.copy()

    @property
    def touching_main_object(self) -> bool:
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            Whether the gripper is touching the object
        """
        return self.touching_object(self.data.geom("peg").id)

    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id: the ID of the object in question

        Returns:
            Whether the gripper is touching the object
        """

        leftpad_geom_id = self.data.geom("leftpad_geom").id
        rightpad_geom_id = self.data.geom("rightpad_geom").id

        leftpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in rightpad_object_contacts
        )

        return 1 < leftpad_object_contact_force and 1 < rightpad_object_contact_force

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:

        pos_hand = self.tcp_center
        quat_hand = self.get_endeff_quat()

        finger_right, finger_left = (
            self.data.body("rightclaw"),
            self.data.body("leftclaw"),
        )
        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obj_pos = self._get_pos_objects()
        obj_quat = self._get_quat_objects()
        pegHead_force, pegHead_torque = self.get_peghead_force_and_torque()
        pegHead_force = np.tanh(pegHead_force / 10.0)
        pegHead_torque = np.tanh(pegHead_torque / 1.0)
        
        return np.hstack((pos_hand, quat_hand, gripper_distance_apart, pegHead_force, pegHead_torque, obj_pos, obj_quat))

    def _get_obs(self) -> npt.NDArray[np.float64]:
        pos_goal = self._goal_pos
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs
        
    @_Decorators.assert_task_is_set
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment."""

        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")

        # 位姿控制（位置 3 维 + 夹爪 1 维）
        self.set_xyz_action(action[:-1])
        
        u = float(np.clip(action[-1], -1, 1))
        r_ctrl = 0.02 + 0.02 * u        # 映射到 [0, 0.04]
        l_ctrl = -0.015 - 0.015 * u     # 映射到 [-0.03, 0]
        self.do_simulation([r_ctrl, l_ctrl], n_frames=self.frame_skip)

        self.curr_path_length += 1

        # 观测裁剪
        self._last_stable_obs = self._get_obs()
        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.sawyer_observation_space.high,
            a_min=self.sawyer_observation_space.low
        ).astype(np.float64)

        # 奖励与信息
        reward, info = self.evaluate_state(self._last_stable_obs)

        # 成功判定
        terminated = False
        if info.get("stage_rewards", {}).get("insertion_depth", 0) >= 0.04:
            terminated = True
            info["success"] = 1.0
        else:
            info["success"] = 0.0

        truncated = (self.curr_path_length == self.max_path_length)

        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
        self.obj_init_pos = pos_peg
        self._set_obj_xyz(self.obj_init_pos)
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        self.model.body("box").pos = pos_box
        self.model.body("box").quat = quat_box
        self._goal_pos = pos_box + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13]))
        self.model.site("goal").pos = self._goal_pos

        return self._get_obs()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use `seed()` instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The `(obs, info)` tuple.
        """
        self.curr_path_length = 0
        _, info = super().reset(seed=seed, options=options)
        initial_obs_curr = self._get_curr_obs_combined_no_goal()
        self._prev_obs = initial_obs_curr.copy() 
        pos_goal = self._goal_pos
        obs = np.hstack((initial_obs_curr, self._prev_obs, pos_goal))
        return obs, info

    def _reset_hand(self, steps: int = 50) -> None:
        """Resets the hand position.

        Args:
            steps: The number of steps to take to reset the hand.
        """
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = self.hand_init_quat
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self) -> npt.NDArray[np.float64]:
        """Gets or generates a random vector for the hand position at reset."""
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        elif self.seeded_rand_vec:
            assert self._random_reset_space is not None
            rand_vec = self.np_random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            )
            self._last_rand_vec = rand_vec
            return rand_vec
        else:
            assert self._random_reset_space is not None
            rand_vec: npt.NDArray[np.float64] = np.random.uniform(  # type: ignore
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            ).astype(np.float64)
            self._last_rand_vec = rand_vec
            return rand_vec