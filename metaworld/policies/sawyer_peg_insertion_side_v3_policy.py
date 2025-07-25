from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegInsertionSideV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "quat_hand": obs[3:7], # New: hand quaternion
            "gripper_distance_apart": obs[7], # Index changed
            "peg_pos": obs[8:11], # Index changed
            "peg_rot": obs[11:15], # Index changed
            "unused_info_curr_obs": obs[15:22], # Index changed, size changed
            "_prev_obs": obs[22:44], # Index changed, size changed
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        # Action is now 7 dimensions: [delta_pos_x, delta_pos_y, delta_pos_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper_effort]
        action = Action(7) # Initialize Action with 7 dimensions

        # Calculate delta_pos
        delta_pos = move(o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0)
        
        # For now, set delta_rot to zero. This can be expanded later if rotation control is needed.
        delta_rot = np.zeros(3) 

        # Calculate gripper_effort
        gripper_effort = self._grab_effort(o_d)

        # Combine all parts into a 7-dimensional action array
        full_action = np.hstack((delta_pos, delta_rot, gripper_effort))
        action.set_action(full_action)

        return action.array.astype(np.float32) # Ensure the return type is float32

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        # lowest X is -.35, doesn't matter if we overshoot
        # Y is given by hole_vec
        # Z is constant at .16
        pos_hole = np.array([-0.35, o_d["goal_pos"][1], 0.16])

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04:
            return pos_peg + np.array([0.0, 0.0, 0.3])
        elif abs(pos_curr[2] - pos_peg[2]) > 0.025:
            return pos_peg
        elif np.linalg.norm(pos_peg[1:] - pos_hole[1:]) > 0.03:
            return pos_hole + np.array([0.4, 0.0, 0.0])
        else:
            return pos_hole - np.array([0.8, 0.0, 0.0])

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]

        if (
            np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04
            or abs(pos_curr[2] - pos_peg[2]) > 0.15
        ):
            return -1.0
        else:
            return 0.6
