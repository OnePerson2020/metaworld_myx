# metaworld/policies/sawyer_peg_insertion_side_v3_policy.py

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegInsertionSideV3Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        # Current observation breakdown:
        # 0-2: hand_pos (3)
        # 3-6: quat_hand (4)
        # 7: gripper_distance_apart (1)
        # 8-10: pegHead_force (3)
        # 11-13: peg_pos (3)        
        # 14-17: peg_rot (4)        
        # 18-24: unused_info_curr_obs (7)
        # Total curr_obs = 3 + 4 + 1 + 3 + 3 + 4 + 7 = 25

        # 25-49: _prev_obs (25)
        # 50-52: goal_pos (3)
        # Total observation length = 53

        return {
            "hand_pos": obs[:3],
            "quat_hand": obs[3:7],
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

        # Action is now 7 dimensions: [delta_pos_x, delta_pos_y, delta_pos_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper_effort]
        action = Action(7)

        # Calculate delta_pos
        delta_pos = move(o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0)
        
        # For now, set delta_rot to zero. This can be expanded later if rotation control is needed.
        delta_rot = np.ones(3) 

        # Calculate gripper_effort
        gripper_effort = self._grab_effort(o_d)

        # Combine all parts into a 7-dimensional action array
        full_action = np.hstack((delta_pos, delta_rot, gripper_effort))
        action.set_action(full_action)

        return action.array.astype(np.float32)

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