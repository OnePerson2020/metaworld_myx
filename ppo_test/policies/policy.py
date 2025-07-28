from __future__ import annotations

import abc
import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

def assert_fully_parsed(
    func: Callable[[npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]
) -> Callable[[npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
    """Decorator function to ensure observations are fully parsed.

    Args:
        func: The function to check

    Returns:
        The input function, decorated to assert full parsing
    """

    def inner(obs) -> dict[str, Any]:
        obs_dict = func(obs)
        assert len(obs) == sum(
            [len(i) if isinstance(i, np.ndarray) else 1 for i in obs_dict.values()]
        ), "Observation not fully parsed"
        return obs_dict

    return inner

def move(
    from_xyz: npt.NDArray[any], 
    to_xyz: npt.NDArray[any], 
    speed: float = 0.2
) -> npt.NDArray[any]:
    """
    根据一个恒定的速度预算，计算从一点到另一点的移动向量。

    Args:
        from_xyz: 起始坐标。
        to_xyz: 目标坐标。
        max_dist_per_step: 在这一个时间步内允许移动的最大距离。

    Returns:
        一个代表本次位移的XYZ向量。
    """
    error_vec = to_xyz - from_xyz
    distance = np.linalg.norm(error_vec)
    max_dist_per_step = speed * 0.0125
    # 如果距离非常小，则不移动
    if distance < 1e-6:
        return np.zeros(3)

    # 如果剩余距离小于单步最大距离，则直接移动到终点以避免过冲
    if distance < max_dist_per_step:
        return error_vec

    # 否则，沿着指向目标的方向，移动一个步长的距离
    direction = error_vec / distance
    delta_pos = direction * max_dist_per_step
    
    return delta_pos

class Policy(abc.ABC):
    """Abstract base class for policies."""

    @staticmethod
    @abc.abstractmethod
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        """Pulls pertinent information out of observation and places in a dict.

        Args:
            obs: Observation which conforms to env.observation_space

        Returns:
            dict: Dictionary which contains information from the observation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        """Gets an action in response to an observation.

        Args:
            obs: Observation which conforms to env.observation_space

        Returns:
            Array (usually 4 elements) representing the action to take
        """
        raise NotImplementedError
