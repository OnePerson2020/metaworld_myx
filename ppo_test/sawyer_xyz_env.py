# metaworld/sawyer_xyz_env.py

from __future__ import annotations

import copy
from functools import cached_property
from typing import Any

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv as mjenv_gym
from gymnasium.spaces import Space

from ppo_test.types import EnvironmentStateDict



class SawyerMocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Sawyer Mujoco envs that use mocap for XYZ control."""

    @cached_property
    def sawyer_observation_space(self) -> Space:
        raise NotImplementedError

    def __init__(
        self,
        model_name: str,
        frame_skip: int = 5,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        mjenv_gym.__init__(
            self,
            model_name,
            frame_skip=frame_skip,
            observation_space=self.sawyer_observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
            width=width,
            height=height,
        )
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

    def get_endeff_pos(self) -> npt.NDArray[Any]:
        """Returns the position of the end effector."""
        return self.data.body("hand").xpos

    def get_endeff_quat(self) -> npt.NDArray[Any]:
        """Returns the quaternion of the end effector."""
        return self.data.body("hand").xquat
    

    @property
    def tcp_center(self) -> npt.NDArray[Any]:
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """
        right_finger_pos = self.data.site("rightEndEffector")
        left_finger_pos = self.data.site("leftEndEffector")
        tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        return tcp_center

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def get_env_state(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get the environment state.

        Returns:
            A tuple of (qpos, qvel).
        """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return copy.deepcopy((qpos, qvel))

    def set_env_state(
        self, state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> None:
        """
        Set the environment state.

        Args:
            state: A tuple of (qpos, qvel).
        """
        qpos, qvel = state
        self.set_state(qpos, qvel)

    def __getstate__(self) -> EnvironmentStateDict:
        """Returns the full state of the environment as a dict.

        Returns:
            A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        state = self.__dict__.copy()
        return {"state": state, "mjb": self.model_name, "mocap": self.get_env_state()}

    def __setstate__(self, state: EnvironmentStateDict) -> None:
        """Sets the state of the environment from a dict exported through `__getstate__()`.

        Args:
            state: A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        self.__dict__ = state["state"]
        mjenv_gym.__init__(
            self,
            state["mjb"],
            frame_skip=self.frame_skip,
            observation_space=self.sawyer_observation_space,
            render_mode=self.render_mode,
            camera_name=self.camera_name,
            camera_id=self.camera_id,
            width=self.width,
            height=self.height,
        )
        self.set_env_state(state["mocap"])

    def reset_mocap_welds(self) -> None:
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
                    )