# 项目导出

**文件数量**: 21  
**总大小**: 142.6 KB  
**Token 数量**: 39.1K  
**生成时间**: 2025/7/30 14:43:40

## 文件结构

```
📁 .
  📁 ppo_test
    📁 policies
      📄 __init__.py
      📄 action.py
      📄 MyPolicy.py
      📄 policy.py
      📄 sawyer_peg_insertion_side_v3_policy.py
    📁 utils
      📄 reward_utils.py
    📄 __init__.py
    📄 asset_path_utils.py
    📄 env_dict.py
    📄 evaluation.py
    📄 sawyer_peg_insertion_side_v3.py
    📄 sawyer_xyz_env.py
    📄 types.py
    📄 wrappers.py
  📁 src_test
    📄 0_init.py
    📄 1-fix_cam.py
  📄 force_v4.py
  📄 plant.py
  📄 show.py
  📄 test.py
  📄 visualize_forces.py
```

## 源文件

### ppo_test/policies/__init__.py

*大小: 281 B | Token: 78*

```python
from ppo_test.policies.sawyer_peg_insertion_side_v3_policy import (
    SawyerPegInsertionSideV3Policy,
)

ENV_POLICY_MAP = dict(
    {
        "peg-insert-side-v3": SawyerPegInsertionSideV3Policy,
    }
)

__all__ = [
    "SawyerPegInsertionSideV3Policy",
    "ENV_POLICY_MAP",
]
```

### ppo_test/policies/action.py

*大小: 854 B | Token: 235*

```python
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Action:
    """Represents an action to be taken in an environment."""

    def __init__(self, action_dim: int) -> None:
        """Action.

        Args:
            action_dim: The dimension of the action space.
        """
        self.action_dim = action_dim
        self.array = np.zeros(action_dim, dtype=np.float32)

    def sample(self) -> np.ndarray:
        """Samples a random action from the action space."""
        return np.random.uniform(-1, 1, self.action_dim).astype(np.float32)

    def set_action(self, action: np.ndarray) -> None:
        """Sets the action array.

        Args:
            action: The action array to set.
        """
        assert action.shape[0] == self.action_dim, "Action dimension mismatch"
        self.array = action
```

### ppo_test/policies/MyPolicy.py

*大小: 6.2 KB | Token: 1.7K*

```python
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

class MyPolicy(Policy):

    def __init__(self, force_feedback_gain=1, force_threshold=15):

        super().__init__()
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)

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
    
    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)
        action = Action(8)

        desired_pos, desired_r = self._desired_pose(o_d)
        
        force_vector = o_d["pegHead_force"]
        force_magnitude = np.linalg.norm(force_vector)
    
        if self.current_stage == 4 and force_magnitude > 1:
            # peg_head - peg
            lever_arm = np.array([-0.15, 0, 0])
            torque_vector = np.cross(lever_arm, force_vector) * self.force_feedback_gain
            # 应用旋转修正
            r_correction = Rotation.from_rotvec(torque_vector)
            desired_r = r_correction * desired_r
            
            print(f"Rot Correction Axis: {r_correction.as_euler('xyz', degrees=True)}")

        # desired_pos = np.array([0.2,0.4,0.3])
        desired_pos = o_d["hand_pos"]
        delta_pos = move(o_d["hand_pos"], to_xyz=desired_pos)
        delta_rot_quat = self._calculate_rotation_action(o_d["hand_quat"], desired_r)

        # delta_rot = np.zeros(3)

        gripper_effort = self._grab_effort(o_d)

        full_action = np.hstack((delta_pos, delta_rot_quat, gripper_effort))
        # full_action = np.zeros(7)
        
        action.set_action(full_action)
        return action.array.astype(np.float32)

    def _desired_pose(self, o_d: dict[str, npt.NDArray[np.float64]]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:

        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]
        pos_hole = o_d["goal_pos"]
        gripper_distance = o_d["gripper_distance_apart"]
        ini_r = Rotation.from_quat([0,1,0,1])
        # desired_r = ini_r * Rotation.from_euler('xyz', [0,0,45], degrees=True)
        
        # 阶段1: 移动到peg正上方
        if self.current_stage == 1:
            # print("Stage 1: Moving to peg top")
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04:
                self.current_stage = 2
            return pos_peg + np.array([0.0, 0.0, 0.3]), ini_r

        # 阶段2: 下降抓取peg
        if self.current_stage == 2:
            # print(f"Stage 2: Descending to peg.")
            if pos_curr[2] - pos_peg[2] < -0.001 and gripper_distance < 0.35:
                # print(">>> Peg lifted! Transitioning to Stage 3.")
                self.current_stage = 3
            return pos_peg - np.array([0.0, 0.0, 0.02]), ini_r
            
        # 阶段3: 移动到洞口预备位置并旋转
        if self.current_stage == 3:
            # print("Stage 3: Moving to hole side")
            if np.linalg.norm(pos_curr[1:] - pos_hole[1:]) < 0.03:
                self.current_stage = 4
            return pos_hole + np.array([0.4, 0.0, 0.0]), ini_r
        
        # 阶段4: 执行插入
        if self.current_stage == 4:
            # print("Stage 4: Inserting peg")
            return pos_hole + np.array([0.1, 0.0, 0.0]), ini_r
            
        return None

    def _calculate_rotation_action(self, current_quat_mujoco, target_Rotation):
            """
            根据当前和目标姿态，计算出平滑的旋转增量（欧拉角格式）。
            如果角度差大于1度，则以恒定的1度角速度旋转；否则，旋转剩余的角度。
            """
            kp = 0.3
            kd = 0.3
            speed = np.deg2rad(0.5)
            
            r_curr = Rotation.from_quat(current_quat_mujoco[[1, 2, 3, 0]])
            r_error = target_Rotation * r_curr.inv()
        
            # 步骤 2: 将差异旋转转换为“旋转向量”(Axis-Angle)
            error_rotvec = r_error.as_rotvec()
            rotation_axis = error_rotvec / np.linalg.norm(error_rotvec) + 1e-16
            # print(f"Current Rotation Error: {np.rad2deg(angle_in_radians):.2f} degrees")
            unconstrained_increment_rotvec = kp * error_rotvec + kd * (error_rotvec - self.prev_r_error_rotvec)
            self.prev_r_error_rotvec = error_rotvec
            
            speed_of_increment = np.linalg.norm(unconstrained_increment_rotvec)
            
            if speed_of_increment > speed:
                increment_rotvec = rotation_axis * speed
            else:
                increment_rotvec = unconstrained_increment_rotvec

            angle_of_increment = np.linalg.norm(increment_rotvec)
            if angle_of_increment < 1e-6:
                return np.array([0., 0., 0., 1.])
            
            r_increment = Rotation.from_rotvec(increment_rotvec)
            delta_rot_quat = r_increment.as_quat()
                     
            r_increment = Rotation.from_rotvec(increment_rotvec)

            delta_rot_quat = r_increment.as_quat()

            return delta_rot_quat


    def _grab_effort(self, o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        pos_curr = o_d["hand_pos"]
        pos_peg = o_d["peg_pos"]

        if not self.gasp:
            if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) < 0.04 and (pos_curr[2] - pos_peg[2]) < -0.001:
                self.gasp = True
                return 0.4
            return -1.0
        else:
            return 0.6
```

### ppo_test/policies/policy.py

*大小: 2.6 KB | Token: 664*

```python
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
```

### ppo_test/policies/sawyer_peg_insertion_side_v3_policy.py

*大小: 3.3 KB | Token: 933*

```python
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
        # 11-13
        # 14-17: peg_pos (3)        
        # 17-20: peg_rot (4)        
        # 21-27: unused_info_curr_obs (7)
        # Total curr_obs = 3 + 4 + 1 + 3 + 3+ 3 + 4 + 7 = 28

        # 28-55: _prev_obs (28)
        # 56-58: goal_pos (3)
        # Total observation length = 59
        return {
            "hand_pos": obs[:3],
            "quat_hand": obs[3:7],
            "gripper_distance_apart": obs[7],
            "pegHead_force": obs[8:11],
            "pegHead_force": obs[11:14],       
            "peg_pos": obs[14:17],            
            "peg_rot": obs[17:21],            
            "unused_info_curr_obs": obs[21:28],
            "_prev_obs": obs[28:56],          
            "goal_pos": obs[-3:],             
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        # Action is now 7 dimensions: [delta_pos_x, delta_pos_y, delta_pos_z, delta_rot_x, delta_rot_y, delta_rot_z, gripper_effort]
        action = Action(7)

        # Calculate delta_pos
        delta_pos = move(o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0)
        # delta_pos = np.zeros(3) 
        # For now, set delta_rot to zero. This can be expanded later if rotation control is needed.
        delta_rot = np.zeros(3) 
        # delta_rot[0] = 0.05
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
```

### ppo_test/utils/reward_utils.py

*大小: 8.2 KB | Token: 2.3K*

```python
"""A set of reward utilities written by the authors of dm_control."""
from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


SIGMOID_TYPE = Literal[
    "gaussian",
    "hyperbolic",
    "long_tail",
    "reciprocal",
    "cosine",
    "linear",
    "quadratic",
    "tanh_squared",
]

X = TypeVar("X", float, npt.NDArray, np.floating)


def _sigmoids(x: X, value_at_1: float, sigmoid: SIGMOID_TYPE) -> X:
    """Maps the input to values between 0 and 1 using a specified sigmoid function. Returns 1 when the input is 0, between 0 and 1 otherwise.

    Args:
        x: The input.
        value_at_1: The output value when `x` == 1. Must be between 0 and 1.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.

    Returns:
        The input mapped to values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}."
            )
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(
                f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}."
            )

    if sigmoid == "gaussian":
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == "long_tail":
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "cosine":
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        ret = np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
        return ret.item() if np.isscalar(x) else ret

    elif sigmoid == "tanh_squared":
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float | np.floating[Any] = 0.0,
    sigmoid: SIGMOID_TYPE = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> X:
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: The input.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError(f"`margin` must be non-negative. Current value: {margin}")

    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value.item() if np.isscalar(x) else value


def inverse_tolerance(
    x: X,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: SIGMOID_TYPE = "reciprocal",
) -> X:
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: The input
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: Choice of sigmoid type. Valid values are 'gaussian', 'hyperbolic',
        'long_tail', 'reciprocal', 'cosine', 'linear', 'quadratic', 'tanh_squared'.
        value_at_margin: A value between 0 and 1 specifying the output when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(
        x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0
    )
    return 1 - bound


def rect_prism_tolerance(
    curr: npt.NDArray[np.float_],
    zero: npt.NDArray[np.float_],
    one: npt.NDArray[np.float_],
) -> float:
    """Computes a reward if curr is inside a rectangular prism region.

    All inputs are 3D points with shape (3,).

    Args:
        curr: The point that the prism reward region is being applied for.
        zero: The diagonal opposite corner of the prism with reward 0.
        one: The corner of the prism with reward 1.

    Returns:
        A reward if curr is inside the prism, 1.0 otherwise.
    """

    def in_range(a, b, c):
        return float(b <= a <= c) if c >= b else float(c <= a <= b)

    in_prism = (
        in_range(curr[0], zero[0], one[0])
        and in_range(curr[1], zero[1], one[1])
        and in_range(curr[2], zero[2], one[2])
    )
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
    else:
        return 1.0


def hamacher_product(a: float, b: float) -> float:
    """Returns the hamacher (t-norm) product of a and b.

    Computes (a * b) / ((a + b) - (a * b)).

    Args:
        a: 1st term of the hamacher product.
        b: 2nd term of the hamacher product.

    Returns:
        The hammacher product of a and b

    Raises:
        ValueError: a and b must range between 0 and 1
    """
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError(f"a ({b}) and b ({b}) must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0.0 <= h_prod <= 1.0
    return h_prod
```

### ppo_test/__init__.py

*大小: 19.6 KB | Token: 5.5K*

```python
from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Literal

import gymnasium as gym 
import numpy as np
import numpy.typing as npt

from gymnasium.envs.registration import register

import ppo_test.env_dict as _env_dict
from ppo_test.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from ppo_test.sawyer_xyz_env import SawyerXYZEnv 
from ppo_test.types import Task 
from ppo_test.wrappers import (
    AutoTerminateOnSuccessWrapper,
    CheckpointWrapper,
    NormalizeRewardsExponential,
    OneHotWrapper,
    PseudoRandomTaskSelectWrapper,
    RandomTaskSelectWrapper,
    RNNBasedMetaRLWrapper,
)


class MetaWorldEnv(abc.ABC):
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    @abc.abstractmethod
    def set_task(self, task: Task) -> None:
        """Sets the task.
        Args:
            task: The task to set.
        Raises:
            ValueError: If `task.env_name` is different from the current task.
        """
        raise NotImplementedError


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    _train_classes: _env_dict.EnvDict
    _test_classes: _env_dict.EnvDict
    _train_tasks: list[Task]
    _test_tasks: list[Task]

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> _env_dict.EnvDict:
        """Returns all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> list[Task]:
        """Returns all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> list[Task]:
        """Returns all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
"""The overrides for the Meta-Learning benchmarks. Disables the inclusion of the goal position in the observation."""

_MT_OVERRIDE = dict(partially_observable=False)
"""The overrides for the Multi-Task benchmarks. Enables the inclusion of the goal position in the observation."""

_N_GOALS = 50
"""The number of goals to generate for each environment."""


def _encode_task(env_name, data) -> Task:
    """Instantiates a new `Task` object after pickling the data.

    Args:
        env_name: The name of the environment.
        data: The task data (will be pickled).

    Returns:
        A `Task` object.
    """
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(
    classes: _env_dict.EnvDict,
    args_kwargs: _env_dict.EnvArgsKwargsDict,
    kwargs_override: dict,
    seed: int | None = None,
) -> list[Task]:
    """Initialises goals for a given set of environments.

    Args:
        classes: The environment classes as an `EnvDict`.
        args_kwargs: The environment arguments and keyword arguments.
        kwargs_override: Any kwarg overrides.
        seed: The random seed to use.

    Returns:
        A flat list of `Task` objects, `_N_GOALS` for each environment in `classes`.
    """
    # Cache existing random state
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    tasks = []
    for env_name, args in args_kwargs.items():
        kwargs = args["kwargs"].copy()
        assert isinstance(kwargs, dict)
        assert len(args["args"]) == 0

        # Init env
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs: list[npt.NDArray[Any]] = []

        # Set task
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)

        for _ in range(_N_GOALS):  # Generate random goals
            env.reset()
            assert env._last_rand_vec is not None
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert (
            unique_task_rand_vecs.shape[0] == _N_GOALS
        ), f"Only generated {unique_task_rand_vecs.shape[0]} unique goals, not {_N_GOALS}"
        env.close()

        # Create a task for each random goal
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            assert isinstance(kwargs, dict)
            del kwargs["task_id"]

            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)

            tasks.append(_encode_task(env_name, kwargs))

        del env

    # Restore random state
    if seed is not None:
        np.random.set_state(st0)

    return tasks


# MT Benchmarks


class MT1(Benchmark):
    """
    The MT1 benchmark.
    A goal-conditioned RL environment for a single Metaworld task.
    """

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")
        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed
        )

        self._test_tasks = []

# ML Benchmarks


class ML1(Benchmark):
    """
    The ML1 benchmark.
    A meta-RL environment for a single Metaworld task.
    The train and test set contain different goal positions.
    The goal position is not part of the observation.
    """

    ENV_NAMES = list(_env_dict.ALL_V3_ENVIRONMENTS.keys())

    def __init__(self, env_name, seed=None):
        super().__init__()
        if env_name not in _env_dict.ALL_V3_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V3 environment")

        cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = self._train_classes
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_tasks(
            self._train_classes, {env_name: args_kwargs}, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes,
            {env_name: args_kwargs},
            _ML_OVERRIDE,
            seed=(seed + 1 if seed is not None else seed),
        )

class CustomML(Benchmark):
    """
    A custom meta RL benchmark.
    Provide the desired train and test env names during initialisation.
    """

    def __init__(self, train_envs: list[str], test_envs: list[str], seed=None):
        if len(set(train_envs).intersection(set(test_envs))) != 0:
            raise ValueError("The test tasks cannot contain any of the train tasks.")

        self._train_classes = _env_dict._get_env_dict(train_envs)
        train_kwargs = _env_dict._get_args_kwargs(
            ALL_V3_ENVIRONMENTS, self._train_classes
        )

        self._test_classes = _env_dict._get_env_dict(test_envs)
        test_kwargs = _env_dict._get_args_kwargs(
            ALL_V3_ENVIRONMENTS, self._test_classes
        )

        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


def _init_each_env(
    env_cls: type[SawyerXYZEnv],
    tasks: list[Task],
    seed: int | None = None,
    max_episode_steps: int | None = None,
    terminate_on_success: bool = False,
    use_one_hot: bool = False,
    env_id: int | None = None,
    num_tasks: int | None = None,
    recurrent_info_in_obs: bool = False,
    normalize_reward_in_recurrent_info: bool = True,
    task_select: Literal["random", "pseudorandom"] = "random",
    reward_normalization_method: Literal["gymnasium", "exponential"] | None = None,
    normalize_observations: bool = False,
    reward_alpha: float = 0.001,
    render_mode: Literal["human", "rgb_array", "depth_array"] | None = None,
    camera_name: str | None = None,
    camera_id: int | None = None,
    width: int = 480,
    height: int = 480,
) -> gym.Env:
    env: gym.Env = env_cls(
        render_mode=render_mode,
        camera_name=camera_name,
        camera_id=camera_id,
        width=width,
        height=height,
    )
    if seed is not None:
        env.seed(seed) 
    env = gym.wrappers.TimeLimit(env, max_episode_steps or env.max_path_length) 
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    if use_one_hot:
        assert env_id is not None, "Need to pass env_id through constructor"
        assert num_tasks is not None, "Need to pass num_tasks through constructor"
        env = OneHotWrapper(env, env_id, num_tasks)
    if recurrent_info_in_obs:
        env = RNNBasedMetaRLWrapper(
            env, normalize_reward=normalize_reward_in_recurrent_info
        )
    if reward_normalization_method == "gymnasium":
        env = gym.wrappers.NormalizeReward(env)
    elif reward_normalization_method == "exponential":
        env = NormalizeRewardsExponential(reward_alpha=reward_alpha, env=env)
    if normalize_observations:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if task_select != "random":
        env = PseudoRandomTaskSelectWrapper(env, tasks)
    else:
        env = RandomTaskSelectWrapper(env, tasks)

    env = CheckpointWrapper(env, f"{env_cls}_{env_id}")
    if seed is not None:
        env.action_space.seed(seed)
    return env


def make_mt_envs(
    name: str,
    seed: int | None = None,
    num_tasks: int | None = None,
    vector_strategy: Literal["sync", "async"] = "sync",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
) -> gym.Env | gym.vector.VectorEnv:
    benchmark: Benchmark
    if name in ALL_V3_ENVIRONMENTS.keys():
        benchmark = MT1(name, seed=seed)
        tasks = [task for task in benchmark.train_tasks]
        return _init_each_env(
            env_cls=benchmark.train_classes[name],
            tasks=tasks,
            seed=seed,
            num_tasks=num_tasks or 1,
            **kwargs,
        )
    else:
        raise ValueError(
            "Invalid MT env name. Must either be a valid Metaworld task name (e.g. 'reach-v3'), 'MT10' or 'MT50'."
        )


def _make_ml_envs_inner(
    benchmark: Benchmark,
    meta_batch_size: int,
    seed: int | None = None,
    total_tasks_per_cls: int | None = None,
    split: Literal["train", "test"] = "train",
    vector_strategy: Literal["sync", "async"] = "sync",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
):
    all_classes = (
        benchmark.train_classes if split == "train" else benchmark.test_classes
    )
    all_tasks = benchmark.train_tasks if split == "train" else benchmark.test_tasks
    assert (
        meta_batch_size % len(all_classes) == 0
    ), "meta_batch_size must be divisible by envs_per_task"
    tasks_per_env = meta_batch_size // len(all_classes)

    env_tuples = []
    for env_name, env_cls in all_classes.items():
        tasks = [task for task in all_tasks if task.env_name == env_name]
        if total_tasks_per_cls is not None:
            tasks = tasks[:total_tasks_per_cls]
        subenv_tasks = [tasks[i::tasks_per_env] for i in range(0, tasks_per_env)]
        for tasks_for_subenv in subenv_tasks:
            assert (
                len(tasks_for_subenv) == len(tasks) // tasks_per_env
            ), f"Invalid division of subtasks, expected {len(tasks) // tasks_per_env} got {len(tasks_for_subenv)}"
            env_tuples.append((env_cls, tasks_for_subenv))

    vectorizer: type[gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv] = getattr(
        gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
    )
    return vectorizer(
        [
            partial(
                _init_each_env,
                env_cls=env_cls,
                tasks=tasks,
                seed=seed,
                **kwargs,
            )
            for env_cls, tasks in env_tuples
        ],
        autoreset_mode=autoreset_mode,
    )


def make_ml_envs(
    name: str,
    seed: int | None = None,
    meta_batch_size: int = 20,
    total_tasks_per_cls: int | None = None,
    split: Literal["train", "test"] = "train",
    vector_strategy: Literal["sync", "async"] = "sync",
    autoreset_mode: gym.vector.AutoresetMode | str = gym.vector.AutoresetMode.SAME_STEP,
    **kwargs,
) -> gym.vector.VectorEnv:
    benchmark: Benchmark
    if name in ALL_V3_ENVIRONMENTS.keys():
        benchmark = ML1(name, seed=seed)
    elif name == "ML10" or name == "ML45" or name == "ML25":
        benchmark = globals()[name](seed=seed)
    else:
        raise ValueError(
            "Invalid ML env name. Must either be a valid Metaworld task name (e.g. 'reach-v3'), 'ML10', 'ML25', or 'ML45'."
        )
    return _make_ml_envs_inner(
        benchmark,
        meta_batch_size=meta_batch_size,
        seed=seed,
        total_tasks_per_cls=total_tasks_per_cls,
        split=split,
        vector_strategy=vector_strategy,
        autoreset_mode=autoreset_mode,
        **kwargs,
    )


make_ml_envs_train = partial(
    make_ml_envs,
    terminate_on_success=False,
    task_select="pseudorandom",
    split="train",
)
make_ml_envs_test = partial(
    make_ml_envs, terminate_on_success=True, task_select="pseudorandom", split="test"
)


def register_mw_envs() -> None:
    def _mt_bench_vector_entry_point(
        mt_bench: str,
        vector_strategy: Literal["sync", "async"],
        autoreset_mode: gym.vector.AutoresetMode
        | str = gym.vector.AutoresetMode.SAME_STEP,
        seed=None,
        use_one_hot=False,
        num_envs=None,
        **lamb_kwargs,
    ):
        if "num_goals" in lamb_kwargs:
            global _N_GOALS
            _N_GOALS = lamb_kwargs["num_goals"]
            del lamb_kwargs["num_goals"]
        return make_mt_envs( 
            mt_bench,
            seed=seed,
            use_one_hot=use_one_hot,
            vector_strategy=vector_strategy, 
            autoreset_mode=autoreset_mode,
            **lamb_kwargs,
        )

    def _ml_bench_vector_entry_point(
        ml_bench: str,
        split: Literal["train", "test"],
        vector_strategy: Literal["sync", "async"],
        autoreset_mode: gym.vector.AutoresetMode
        | str = gym.vector.AutoresetMode.SAME_STEP,
        total_tasks_per_cls: int | None = None,
        seed: int | None = None,
        meta_batch_size: int = 20,
        num_envs=None,
        **lamb_kwargs,
    ):
        env_generator = make_ml_envs_train if split == "train" else make_ml_envs_test
        return env_generator(
            ml_bench,
            seed=seed,
            meta_batch_size=meta_batch_size,
            total_tasks_per_cls=total_tasks_per_cls,
            vector_strategy=vector_strategy,
            split=split,
            autoreset_mode=autoreset_mode,
            **lamb_kwargs,
        )

    register(
        id="Meta-World/MT1",
        entry_point=lambda env_name, use_one_hot=False, vector_strategy="sync", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, seed=None, num_envs=None, **kwargs: _mt_bench_vector_entry_point(
            env_name,
            vector_strategy,
            autoreset_mode,
            seed,
            use_one_hot,
            num_envs,
            **kwargs,
        ),
        kwargs={},
    )

    for split in ["train", "test"]:
        register(
            id=f"Meta-World/ML1-{split}",
            vector_entry_point=lambda env_name, vector_strategy="sync", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, total_tasks_per_cls=None, meta_batch_size=20, seed=None, num_envs=None, **kwargs: _ml_bench_vector_entry_point(
                env_name,
                split,  # type: ignore[arg-type]
                vector_strategy,
                autoreset_mode,
                total_tasks_per_cls,
                seed,
                meta_batch_size,
                num_envs,
                **kwargs,
            ),
            kwargs={},
        )
    register(
        id="Meta-World/goal_hidden",
        entry_point=lambda env_name, seed: ALL_V3_ENVIRONMENTS_GOAL_HIDDEN[
            env_name + "-goal-hidden" if "-goal-hidden" not in env_name else env_name
        ]( 
            seed=seed,
        ),
        kwargs={},
    )

    register(
        id="Meta-World/goal_observable",
        entry_point=lambda env_name, seed: ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[
            env_name + "-goal-observable"
            if "-goal-observable" not in env_name
            else env_name
        ]( 
            seed=seed
        ),
        kwargs={},
    )
    def _custom_mt_vector_entry_point(
        vector_strategy: str,
        envs_list: list[str],
        seed=None,
        autoreset_mode: gym.vector.AutoresetMode
        | str = gym.vector.AutoresetMode.SAME_STEP,
        use_one_hot: bool = False,
        num_envs=None,
        **lamb_kwargs,
    ):
        vectorizer: type[gym.vector.VectorEnv] = getattr(
            gym.vector, f"{vector_strategy.capitalize()}VectorEnv"
        )
        return vectorizer( 
            [
                partial( 
                    make_mt_envs,
                    env_name,
                    num_tasks=len(envs_list),
                    env_id=idx,
                    seed=None if not seed else seed + idx,
                    use_one_hot=use_one_hot,
                    **lamb_kwargs,
                )
                for idx, env_name in enumerate(envs_list)
            ],
            autoreset_mode=autoreset_mode,
        )

    register(
        id="Meta-World/custom-mt-envs",
        vector_entry_point=lambda vector_strategy, envs_list, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, seed=None, use_one_hot=False, num_envs=None, **kwargs: _custom_mt_vector_entry_point(
            vector_strategy,
            envs_list,
            seed,
            autoreset_mode,
            use_one_hot,
            num_envs,
            **kwargs,
        ),
        kwargs={},
    )

    def _custom_ml_vector_entry_point(
        vector_strategy: str,
        train_envs: list[str],
        test_envs: list[str],
        autoreset_mode: gym.vector.AutoresetMode
        | str = gym.vector.AutoresetMode.SAME_STEP,
        total_tasks_per_cls: int | None = None,
        meta_batch_size: int = 20,
        seed=None,
        num_envs=None,
        **lamb_kwargs,
    ):
        return _make_ml_envs_inner( 
            CustomML(train_envs, test_envs, seed=seed),
            meta_batch_size=meta_batch_size,
            vector_strategy=vector_strategy, 
            autoreset_mode=autoreset_mode,
            total_tasks_per_cls=total_tasks_per_cls,
            seed=seed,
            **lamb_kwargs,
        )

    register(
        id="Meta-World/custom-ml-envs",
        vector_entry_point=lambda vector_strategy, train_envs, test_envs, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP, total_tasks_per_cls=None, meta_batch_size=20, seed=None, num_envs=None, **kwargs: _custom_ml_vector_entry_point(
            vector_strategy,
            train_envs,
            test_envs,
            autoreset_mode,
            total_tasks_per_cls,
            meta_batch_size,
            seed,
            num_envs,
            **kwargs,
        ),
        kwargs={},
    )


register_mw_envs()
__all__: list[str] = []
```

### ppo_test/asset_path_utils.py

*大小: 532 B | Token: 147*

```python
"""Set of utilities for retrieving asset paths for the environments."""

from __future__ import annotations

from pathlib import Path

_CURRENT_FILE_DIR = Path(__file__).parent.absolute()

ENV_ASSET_DIR_V3 = _CURRENT_FILE_DIR / 'xml'

def full_V3_path_for(file_name: str) -> str:
    """Retrieves the full, absolute path for a given V3 asset

    Args:
        file_name: Name of the asset file. Can include subdirectories.

    Returns:
        The full path to the asset file.
    """
    return str(ENV_ASSET_DIR_V3 / file_name)
```

### ppo_test/env_dict.py

*大小: 5.8 KB | Token: 1.6K*

```python
"""Dictionaries mapping environment name strings to environment classes,
and organising them into various collections and splits for the benchmarks."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Literal
from typing import OrderedDict as Typing_OrderedDict
from typing import Sequence, Union

import numpy as np
from typing_extensions import TypeAlias

# from metaworld import envs
from ppo_test.sawyer_xyz_env import SawyerXYZEnv
from ppo_test.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3

# Utils

EnvDict: TypeAlias = "Typing_OrderedDict[str, type[SawyerXYZEnv]]"
TrainTestEnvDict: TypeAlias = "Typing_OrderedDict[Literal['train', 'test'], EnvDict]"
EnvArgsKwargsDict: TypeAlias = (
    "Dict[str, Dict[Literal['args', 'kwargs'], Union[List, Dict]]]"
)

ENV_CLS_MAP = {
    "peg-insert-side-v3": SawyerPegInsertionSideEnvV3,
}


def _get_env_dict(env_names: Sequence[str]) -> EnvDict:
    """Returns an `OrderedDict` containing `(env_name, env_cls)` tuples for the given env_names.

    Args:
        env_names: The environment names

    Returns:
        The appropriate `OrderedDict.
    """
    return OrderedDict([(env_name, ENV_CLS_MAP[env_name]) for env_name in env_names])


def _get_train_test_env_dict(
    train_env_names: Sequence[str], test_env_names: Sequence[str]
) -> TrainTestEnvDict:
    """Returns an `OrderedDict` containing two sub-keys ("train" and "test" at positions 0 and 1),
    each containing the appropriate `OrderedDict` for the train and test classes of the benchmark.

    Args:
        train_env_names: The train environment names.
        test_env_names: The test environment names

    Returns:
        The appropriate `OrderedDict`.
    """
    return OrderedDict(
        (
            ("train", _get_env_dict(train_env_names)),
            ("test", _get_env_dict(test_env_names)),
        )
    )


def _get_args_kwargs(all_envs: EnvDict, env_subset: EnvDict) -> EnvArgsKwargsDict:
    """Returns containing a `dict` of "args" and "kwargs" for each environment in a given list of environments.
    Specifically, sets an empty "args" array and a "kwargs" dictionary with a "task_id" key for each env.

    Args:
        all_envs: The full list of envs
        env_subset: The subset of envs to get args and kwargs for

    Returns:
        The args and kwargs dictionary.
    """
    return {
        key: dict(args=[], kwargs={"task_id": list(all_envs.keys()).index(key)})
        for key, _ in env_subset.items()
    }


def _create_hidden_goal_envs(all_envs: EnvDict) -> EnvDict:
    """Create versions of the environments with the goal hidden.

    Args:
        all_envs: The full list of envs in the benchmark.

    Returns:
        An `EnvDict` where the classes have been modified to hide the goal.
    """
    hidden_goal_envs = {}
    for env_name, env_cls in all_envs.items():
        d = {}

        def initialize(env, seed=None):
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)
            super(type(env), env).__init__()
            env._partially_observable = True
            del env.sawyer_observation_space
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed=seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        hg_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        hg_env_name = hg_env_name.replace("-", "")
        hg_env_key = f"{env_name}-goal-hidden"
        hg_env_name = f"{hg_env_name}GoalHidden"
        HiddenGoalEnvCls = type(hg_env_name, (env_cls,), d)
        hidden_goal_envs[hg_env_key] = HiddenGoalEnvCls

    return OrderedDict(hidden_goal_envs)


def _create_observable_goal_envs(all_envs: EnvDict) -> EnvDict:
    """Create versions of the environments with the goal observable.

    Args:
        all_envs: The full list of envs in the benchmark.

    Returns:
        An `EnvDict` where the classes have been modified to make the goal observable.
    """
    observable_goal_envs = {}
    for env_name, env_cls in all_envs.items():
        d = {}

        def initialize(env, *args, **kwargs):
            seed = kwargs.pop('seed', None)
            render_mode = kwargs.pop('render_mode', None)
            if seed is not None:
                st0 = np.random.get_state()
                np.random.seed(seed)

            super(type(env), env).__init__(*args, **kwargs)
            env._partially_observable = False
            env._freeze_rand_vec = False
            del env.sawyer_observation_space
            env._set_task_called = True
            env.render_mode = render_mode
            env.reset()
            env._freeze_rand_vec = True
            if seed is not None:
                env.seed(seed)
                np.random.set_state(st0)

        d["__init__"] = initialize
        og_env_name = re.sub(
            r"(^|[-])\s*([a-zA-Z])", lambda p: p.group(0).upper(), env_name
        )
        og_env_name = og_env_name.replace("-", "")

        og_env_key = f"{env_name}-goal-observable"
        og_env_name = f"{og_env_name}GoalObservable"
        ObservableGoalEnvCls = type(og_env_name, (env_cls,), d)
        observable_goal_envs[og_env_key] = ObservableGoalEnvCls

    return OrderedDict(observable_goal_envs)


# V3 DICTS

ALL_V3_ENVIRONMENTS = _get_env_dict(
    [
        "peg-insert-side-v3",
    ]
)


ALL_V3_ENVIRONMENTS_GOAL_HIDDEN = _create_hidden_goal_envs(ALL_V3_ENVIRONMENTS)
ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE = _create_observable_goal_envs(ALL_V3_ENVIRONMENTS)

# ML Dicts

ML1_V3 = _get_train_test_env_dict(
    list(ALL_V3_ENVIRONMENTS.keys()), list(ALL_V3_ENVIRONMENTS.keys())
)
ML1_args_kwargs = _get_args_kwargs(ALL_V3_ENVIRONMENTS, ML1_V3["train"])
```

### ppo_test/evaluation.py

*大小: 5.5 KB | Token: 1.5K*

```python
from __future__ import annotations

from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from env_dict import ALL_V3_ENVIRONMENTS


class Agent(Protocol):
    def eval_action(
        self, observations: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        ...


class MetaLearningAgent(Agent, Protocol):
    def init(self) -> None:
        ...

    def adapt_action(
        self, observations: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray]]:
        ...

    def step(self, timestep: Timestep) -> None:
        ...

    def adapt(self) -> None:
        ...


def _get_task_names(
    envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
) -> list[str]:
    metaworld_cls_to_task_name = {v.__name__: k for k, v in ALL_V3_ENVIRONMENTS.items()}
    return [
        metaworld_cls_to_task_name[task_name]
        for task_name in envs.get_attr("task_name")
    ]


def evaluation(
    agent: Agent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_episodes: int = 50,
) -> tuple[float, float, dict[str, float], dict[str, list[float]]]:
    terminate_on_success = np.all(eval_envs.get_attr("terminate_on_success")).item()
    eval_envs.call("toggle_terminate_on_success", True)

    obs: npt.NDArray[np.float64]
    obs, _ = eval_envs.reset()
    agent.reset(np.ones(eval_envs.num_envs, dtype=np.bool_))

    task_names = _get_task_names(eval_envs)
    successes = {task_name: 0 for task_name in set(task_names)}
    episodic_returns: dict[str, list[float]] = {
        task_name: [] for task_name in set(task_names)
    }

    def eval_done(returns):
        return all(len(r) >= num_episodes for _, r in returns.items())

    while not eval_done(episodic_returns):
        actions = agent.eval_action(obs)
        obs, _, terminations, truncations, infos = eval_envs.step(actions)

        dones = np.logical_or(terminations, truncations)
        agent.reset(dones)

        for i, env_ended in enumerate(dones):
            if env_ended:
                episodic_returns[task_names[i]].append(
                    float(infos["final_info"]["episode"]["r"][i])
                )
                if len(episodic_returns[task_names[i]]) <= num_episodes:
                    successes[task_names[i]] += int(infos["final_info"]["success"][i])

    episodic_returns = {
        task_name: returns[:num_episodes]
        for task_name, returns in episodic_returns.items()
    }

    success_rate_per_task = {
        task_name: task_successes / num_episodes
        for task_name, task_successes in successes.items()
    }
    mean_success_rate = np.mean(list(success_rate_per_task.values()))
    mean_returns = np.mean(list(episodic_returns.values()))

    eval_envs.call("toggle_terminate_on_success", terminate_on_success)

    return (
        float(mean_success_rate),
        float(mean_returns),
        success_rate_per_task,
        episodic_returns,
    )


def metalearning_evaluation(
    agent: MetaLearningAgent,
    eval_envs: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    num_evals: int = 10,  # Assuming 40 goals per test task and meta batch size of 20
    adaptation_steps: int = 1,
    adaptation_episodes: int = 10,
    evaluation_episodes: int = 3,
) -> tuple[float, float, dict[str, float]]:
    eval_envs.call("toggle_sample_tasks_on_reset", False)
    eval_envs.call("toggle_terminate_on_success", False)
    task_names = _get_task_names(eval_envs)

    total_mean_success_rate = 0.0
    total_mean_return = 0.0
    success_rate_per_task = np.zeros((num_evals, len(set(task_names))))

    for i in range(num_evals):
        obs: npt.NDArray[np.float64]

        eval_envs.call("sample_tasks")
        agent.init()

        for _ in range(adaptation_steps):
            obs, _ = eval_envs.reset()
            episodes_elapsed = np.zeros((eval_envs.num_envs,), dtype=np.uint16)

            while not (episodes_elapsed >= adaptation_episodes).all():
                actions, aux_policy_outs = agent.adapt_action(obs)
                next_obs, rewards, terminations, truncations, _ = eval_envs.step(
                    actions
                )
                agent.step(
                    Timestep(
                        obs,
                        actions,
                        rewards,
                        terminations,
                        truncations,
                        aux_policy_outs,
                    )
                )
                episodes_elapsed += np.logical_or(terminations, truncations)
                obs = next_obs

            agent.adapt()

        # Evaluation
        mean_success_rate, mean_return, _success_rate_per_task, _ = evaluation(
            agent, eval_envs, evaluation_episodes
        )
        total_mean_success_rate += mean_success_rate
        total_mean_return += mean_return
        success_rate_per_task[i] = np.array(list(_success_rate_per_task.values()))

    success_rates = (success_rate_per_task).mean(axis=0)
    task_success_rates = {
        task_name: success_rates[i] for i, task_name in enumerate(set(task_names))
    }

    return (
        total_mean_success_rate / num_evals,
        total_mean_return / num_evals,
        task_success_rates,
    )


class Timestep(NamedTuple):
    observation: npt.NDArray
    action: npt.NDArray
    reward: npt.NDArray
    terminated: npt.NDArray
    truncated: npt.NDArray
    aux_policy_outputs: dict[str, npt.NDArray]
```

### ppo_test/sawyer_peg_insertion_side_v3.py

*大小: 11.9 KB | Token: 3.1K*

```python
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

    def get_peg_force_and_torque_from_sensor(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        从内置的MuJoCo传感器中直接读取peg头部的力和力矩。
        """
        # data.sensordata 是一个包含所有传感器读数的一维数组
        # 我们可以通过名称找到每个传感器的读数
        force_reading = self.data.sensor("peg_force_sensor").data
        torque_reading = self.data.sensor("peg_torque_sensor").data
        
        # force_reading 和 torque_reading 都已经是 (3,) 的 numpy 数组
        return force_reading.copy(), torque_reading.copy()
    
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
            obj_to_target <= 0.008
        )
            
        near_object = float(tcp_to_obj <= 0.03)
        
        peg_force, peg_torque = self.get_peghead_force_and_torque()
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
        scale = np.array([1.0, 2.0, 2.0]) / 3.0
        
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
```

### ppo_test/sawyer_xyz_env.py

*大小: 33.1 KB | Token: 9.3K*

```python
# metaworld/sawyer_xyz_env.py

from __future__ import annotations

import abc
import copy
import pickle
from functools import cached_property
from typing import Any, Callable, Literal, SupportsFloat

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Space
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias

from ppo_test.types import XYZ, EnvironmentStateDict, ObservationDict, Task
from ppo_test.utils import reward_utils
from scipy.spatial.transform import Rotation

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"


class SawyerMocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Sawyer Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06, -1.0, -1.0, -1.0, -1.0])
    mocap_high = np.array([0.2, 0.7, 0.6, 1.0, 1.0, 1.0, 1.0])
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 80,
    }

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


class SawyerXYZEnv(SawyerMocapBase, EzPickle):
    """The base environment for all Sawyer Mujoco envs that use mocap for XYZ control."""

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

    max_path_length: int = 1000
    """The maximum path length for the environment (the task horizon)."""

    TARGET_RADIUS: float = 0.05
    """Upper bound for distance from the target when checking for task completion."""

    class _Decorators:
        @classmethod
        def assert_task_is_set(cls, func: Callable) -> Callable:
            """Asserts that the task has been set in the environment before proceeding with the function call.
            To be used as a decorator for SawyerXYZEnv methods."""

            def inner(*args, **kwargs) -> Any:
                env = args[0]
                if not env._set_task_called:
                    raise RuntimeError(
                        "You must call env.set_task before using env." + func.__name__
                    )
                return func(*args, **kwargs)

            return inner

    def __init__(
        self,
        frame_skip: int = 5,
        hand_low: XYZ = (-0.2, 0.55, 0.05),
        hand_high: XYZ = (0.2, 0.75, 0.3),
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_name: str | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
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

        self._partially_observable: bool = True

        self.task_name = self.__class__.__name__
        self._obs_obj_max_len: int = 14
        self._set_task_called: bool = False
        
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
        
        self._did_see_sim_exception: bool = False
        self.init_left_pad: npt.NDArray[Any] = self.get_body_com("leftpad")
        self.init_right_pad: npt.NDArray[Any] = self.get_body_com("rightpad")

        self.action_space = Box(  # type: ignore
            np.array([-1, -1, -1, -1, -1, -1, -1, -1]),
            np.array([+1, +1, +1, +1, +1, +1, +1, +1]),
            dtype=np.float32,
        )

        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self.hand_init_quat: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._target_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._random_reset_space: Box | None = None  # OVERRIDE ME
        self.goal_space: Box | None = None  # OVERRIDE ME
        self._last_stable_obs: npt.NDArray[np.float64] | None = None

        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of _prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)
        # The observation size is 3 (pos) + 4 (quat) + 1 (gripper) + 3 (force) + 3 (torque) + 14 (obj_padded) = 28
        # Stacked observation is 28 * 2 + 3 (goal) = 59
        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = np.zeros(28, dtype=np.float64)

        self.task_name = self.__class__.__name__

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

    @staticmethod
    def _set_task_inner() -> None:
        """Helper method to set additional task data. To be overridden by subclasses as appropriate."""
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        pass

    def set_task(self, task: Task) -> None:
        """Sets the environment's task.

        Args:
            task: The task to set.
        """
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._freeze_rand_vec = True
        self._last_rand_vec = data["rand_vec"]
        del data["rand_vec"]
        new_observability = data["partially_observable"]
        if new_observability != self._partially_observable:
            # Force recomputation of the observation space
            # See https://docs.python.org/3/library/functools.html#functools.cached_property
            del self.sawyer_observation_space
        self._partially_observable = new_observability
        del data["partially_observable"]
        self._set_task_inner(**data)

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Adjusts the position of the mocap body from the given action.
        Moves each body axis in XYZ by the amount described by the action.

        Args:
            action: The action to apply (in offsets between :math:`[-1, 1]` for each axis in XYZ).
        """
        action = np.clip(action, -1, 1)
        pos_delta = action[:3]
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low[:3],
            self.mocap_high[:3],
        )
        self.data.mocap_pos = new_mocap_pos
        
        r_increment = Rotation.from_quat(action[3:7])
        
        current_mocap = self.data.mocap_quat[0]
        new_mocap_r = r_increment * Rotation.from_quat(current_mocap[[1,2,3,0]])
        new_mocap_quat = new_mocap_r.as_quat()[[3,0,1,2]]
        self.data.mocap_quat[0] = new_mocap_quat

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy() # 前 9 个分别对应 7 个关节角度和 2 个夹爪的控制量
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float64]:
        """Gets the position of a given site.

        Args:
            site_name: The name of the site to get the position of.

        Returns:
            Flat, 3 element array indicating site's location.
        """
        return self.data.site(site_name).xpos.copy()

    def _set_pos_site(self, name: str, pos: npt.NDArray[Any]) -> None:
        """Sets the position of a given site.

        Args:
            name: The site's name
            pos: Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        self.data.site(name).xpos = pos[:3]

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        """Retrieves site name(s) and position(s) corresponding to env targets."""
        assert self._target_pos is not None
        return [("goal", self._target_pos)]

    @property
    def touching_main_object(self) -> bool:
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            Whether the gripper is touching the object
        """
        return self.touching_object(self._get_id_main_object())

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

        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force

    def _get_id_main_object(self) -> int:
        return self.data.geom("peg").id

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Retrieves object position(s) from mujoco properties or instance vars.

        Returns:
            Flat array (usually 3 elements) representing the object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        """Retrieves object quaternion(s) from mujoco properties.

        Returns:
            Flat array (usually 4 elements) representing the object(s)' quaternion(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_pos_goal(self) -> npt.NDArray[Any]:
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:

        pos_hand = self.tcp_center
        quat_hand = self.get_endeff_quat() # Get hand quaternion

        finger_right, finger_left = (
            self.data.body("rightclaw"),
            self.data.body("leftclaw"),
        )
        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco

        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        try:
            pegHead_force, pegHead_torque = self.get_peghead_force_and_torque()
        except NotImplementedError:
            pegHead_force = np.zeros(3)
            pegHead_torque = np.zeros(3)
        return np.hstack((pos_hand, quat_hand, gripper_distance_apart, pegHead_force, pegHead_torque, obs_obj_padded))


    @cached_property
    def sawyer_observation_space(self) -> Box:
        obs_obj_max_len = self._obs_obj_max_len
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
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
        
        force_low = np.full(3, -np.inf, dtype=np.float64)
        force_high = np.full(3, np.inf, dtype=np.float64)

        torque_low = np.full(3, -np.inf, dtype=np.float64)
        torque_high = np.full(3, np.inf, dtype=np.float64)
        
        # Current observation: pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + pegHead_force (3) + pegHead_torque + obs_obj_padded (14) = 28
        # Goal: 3
        # Total: 28 + 28 + 3 = 59
        return Box(
            np.hstack(
                (
                    self._HAND_POS_SPACE.low, 
                    self._HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    self._HAND_POS_SPACE.low, 
                    self._HAND_QUAT_SPACE.low,
                    gripper_low,
                    force_low,
                    torque_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_POS_SPACE.high,
                    self._HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    self._HAND_POS_SPACE.high,
                    self._HAND_QUAT_SPACE.high,
                    gripper_high,
                    force_high,
                    torque_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

    def _get_obs(self) -> npt.NDArray[np.float64]:
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_obs_dict(self) -> ObservationDict:
        obs = self._get_obs()
        # The state_achieved_goal should be the end-effector position (3) + quaternion (4) = 7 elements
        # The current observation is pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + 3 (force) + obs_obj_padded (14) = 25
        # The previous observation is 25 elements
        # The goal is 3 elements
        # Total observation is 25 + 25 + 3 = 53
        # state_achieved_goal should be the current end-effector pos and quat, which are the first 7 elements of curr_obs
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[:7], # Updated to reflect hand pos and quat
        )
        
    @_Decorators.assert_task_is_set
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action:
        Returns:
            The (next_obs, reward, terminated, truncated, info) tuple.
        """
        assert len(action) == 8, f"Actions should be size 8, got {len(action)}"
        self.set_xyz_action(action[:7]) # Pass position and rotation actions
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            assert self._last_stable_obs is not None
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,  # termination flag always False
                False,
                {
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )
        mujoco.mj_forward(self.model, self.data)
        
        self._last_stable_obs = self._get_obs()
        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.sawyer_observation_space.high,
            a_min=self.sawyer_observation_space.low,
            dtype=np.float64,
        )
        
        assert isinstance(self._last_stable_obs, np.ndarray)
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        # step will never return a terminate==True if there is a success
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            False,
            truncate,
            info,
        )

    @abc.abstractmethod
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        """
        Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            Tuple of reward between 0 and 10 and a dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)
        """
        raise NotImplementedError

    def reset_model(self) -> npt.NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
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
        self.reset_model()
        obs, info = super().reset()
        assert obs is not None, "Observation should not be None after reset"
        curr_obs_len = 28
        self._prev_obs = obs[:curr_obs_len].copy()        
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

    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ) -> float:
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.

        Returns:
            the reward value
        """
        assert (
            self.obj_init_pos is not None
        ), "`obj_init_pos` must be initialized before calling this function."

        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, float(caging_xz))
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + float(reach)) / 2

        return caging_and_gripping
```

### ppo_test/types.py

*大小: 1.2 KB | Token: 331*

```python
from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import NotRequired, TypeAlias, TypedDict


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a `MetaWorldEnv`'s `set_task` method.
    """

    env_name: str
    data: bytes  # Contains env parameters like random_init and *a* goal


XYZ: TypeAlias = "Tuple[float, float, float]"
"""A 3D coordinate."""


class EnvironmentStateDict(TypedDict):
    state: dict[str, Any]
    mjb: str
    mocap: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


class ObservationDict(TypedDict):
    state_observation: npt.NDArray[np.float64]
    state_desired_goal: npt.NDArray[np.float64]
    state_achieved_goal: npt.NDArray[np.float64]


class InitConfigDict(TypedDict):
    obj_init_angle: NotRequired[float]
    obj_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class HammerInitConfigDict(TypedDict):
    hammer_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class StickInitConfigDict(TypedDict):
    stick_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]
```

### ppo_test/wrappers.py

*大小: 11.6 KB | Token: 3.3K*

```python
from __future__ import annotations

import base64

import gymnasium as gym
import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

from ppo_test.sawyer_xyz_env import SawyerXYZEnv
from ppo_test.types import Task


class OneHotWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: Env, task_idx: int, num_tasks: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        env_lb = env.observation_space.low
        env_ub = env.observation_space.high
        one_hot_ub = np.ones(num_tasks)
        one_hot_lb = np.zeros(num_tasks)

        self.one_hot = np.zeros(num_tasks)
        self.one_hot[task_idx] = 1.0

        self._observation_space = gym.spaces.Box(
            np.concatenate([env_lb, one_hot_lb]), np.concatenate([env_ub, one_hot_ub])
        )

    def observation(self, obs: NDArray) -> NDArray:
        return np.concatenate([obs, self.one_hot])


def _serialize_task(task: Task) -> dict:
    return {
        "env_name": task.env_name,
        "data": base64.b64encode(task.data).decode("ascii"),
    }


def _deserialize_task(task_dict: dict[str, str]) -> Task:
    assert "env_name" in task_dict and "data" in task_dict

    return Task(
        env_name=task_dict["env_name"], data=base64.b64decode(task_dict["data"])
    )


class RNNBasedMetaRLWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically include prev_action / reward / done info in the observation.
    For use with RNN-based meta-RL algorithms."""

    def __init__(self, env: Env, normalize_reward: bool = True):
        super().__init__(env)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        obs_flat_dim = int(np.prod(self.env.observation_space.shape))
        action_flat_dim = int(np.prod(self.env.action_space.shape))
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_flat_dim + action_flat_dim + 1 + 1,)
        )
        self._normalize_reward = normalize_reward

    def step(self, action):
        next_obs, reward, terminate, truncate, info = self.env.step(action)
        if self._normalize_reward:
            obs_reward = float(reward) / 10.0
        else:
            obs_reward = float(reward)

        recurrent_obs = np.concatenate(
            [
                next_obs,
                action,
                [obs_reward],
                [float(np.logical_or(terminate, truncate))],
            ]
        )
        return recurrent_obs, reward, terminate, truncate, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        assert isinstance(self.env.action_space, gym.spaces.Box)
        obs, info = self.env.reset(seed=seed, options=options)
        recurrent_obs = np.concatenate(
            [obs, np.zeros(self.env.action_space.shape), [0.0], [0.0]]
        )
        return recurrent_obs, info


class RandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically set / reset the environment to a random
    task."""

    tasks: list[Task]
    sample_tasks_on_reset: bool = True

    def _set_random_task(self):
        task_idx = self.np_random.choice(len(self.tasks))
        self.unwrapped.set_task(self.tasks[task_idx])

    def __init__(
        self,
        env: Env,
        tasks: list[Task],
        sample_tasks_on_reset: bool = True,
    ):
        super().__init__(env)
        self.unwrapped: SawyerXYZEnv
        self.tasks = tasks
        self.sample_tasks_on_reset = sample_tasks_on_reset

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.sample_tasks_on_reset:
            self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(self, *, seed: int | None = None, options: dict | None = None):
        self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def get_checkpoint(self) -> dict:
        return {
            "tasks": [_serialize_task(task) for task in self.tasks],
            "rng_state": self.np_random.bit_generator.state,
            "sample_tasks_on_reset": self.sample_tasks_on_reset,
            "env_rng_state": get_env_rng_checkpoint(self.unwrapped),
        }

    def load_checkpoint(self, ckpt: dict):
        assert "tasks" in ckpt
        assert "rng_state" in ckpt
        assert "sample_tasks_on_reset" in ckpt
        assert "env_rng_state" in ckpt

        self.tasks = [_deserialize_task(task) for task in ckpt["tasks"]]
        self.np_random.__setstate__(ckpt["rng_state"])
        self.sample_tasks_on_reset = ckpt["sample_tasks_on_reset"]
        set_env_rng(self.unwrapped, ckpt["env_rng_state"])


class PseudoRandomTaskSelectWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically reset the environment to a *pseudo*random task when explicitly called.

    Pseudorandom implies no collisions therefore the next task in the list will be used cyclically.
    However, the tasks will be shuffled every time the last task of the previous shuffle is reached.

    Doesn't sample new tasks on reset by default.
    """

    tasks: list[Task]
    current_task_idx: int
    sample_tasks_on_reset: bool = False

    def _set_pseudo_random_task(self):
        self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        if self.current_task_idx == 0:
            self.np_random.shuffle(self.tasks)  # pyright: ignore [reportArgumentType]
        self.unwrapped.set_task(self.tasks[self.current_task_idx])

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def __init__(
        self,
        env: Env,
        tasks: list[Task],
        sample_tasks_on_reset: bool = False,
    ):
        super().__init__(env)
        self.sample_tasks_on_reset = sample_tasks_on_reset
        self.tasks = tasks
        self.current_task_idx = -1

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.sample_tasks_on_reset:
            self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(self, *, seed: int | None = None, options: dict | None = None):
        self._set_pseudo_random_task()
        return self.env.reset(seed=seed, options=options)

    def get_checkpoint(self) -> dict:
        return {
            "tasks": [_serialize_task(task) for task in self.tasks],
            "current_task_idx": self.current_task_idx,
            "sample_tasks_on_reset": self.sample_tasks_on_reset,
            "env_rng_state": get_env_rng_checkpoint(self.unwrapped),
        }

    def load_checkpoint(self, ckpt: dict):
        assert "tasks" in ckpt
        assert "current_task_idx" in ckpt
        assert "sample_tasks_on_reset" in ckpt
        assert "env_rng_state" in ckpt

        self.tasks = [_deserialize_task(task) for task in ckpt["tasks"]]
        self.current_task_idx = ckpt["current_task_idx"]
        self.sample_tasks_on_reset = ckpt["sample_tasks_on_reset"]
        set_env_rng(self.unwrapped, ckpt["env_rng_state"])


class AutoTerminateOnSuccessWrapper(gym.Wrapper):
    """A Gymnasium Wrapper to automatically output a termination signal when the environment's task is solved.
    That is, when the 'success' key in the info dict is True.

    This is not the case by default in SawyerXYZEnv, because terminating on success during training leads to
    instability and poor evaluation performance. However, this behaviour is desired during said evaluation.
    Hence the existence of this wrapper.

    Best used *under* an AutoResetWrapper and RecordEpisodeStatistics and the like."""

    terminate_on_success: bool = True

    def __init__(self, env: Env):
        super().__init__(env)
        self.terminate_on_success = True

    def toggle_terminate_on_success(self, on: bool):
        self.terminate_on_success = on

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.terminate_on_success:
            terminated = info["success"] == 1.0
        return obs, reward, terminated, truncated, info


class NormalizeRewardsExponential(gym.Wrapper):
    def __init__(self, reward_alpha, env):
        super().__init__(env)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.0
        self._reward_var = 1.0

    def _update_reward_estimate(self, reward):
        self._reward_mean = (
            1 - self._reward_alpha
        ) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean
        )

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def step(self, action: NDArray):
        next_obs, reward, terminate, truncate, info = self.env.step(action)
        self._update_reward_estimate(reward)  # type: ignore
        reward = self._apply_normalize_reward(reward)  # type: ignore
        return next_obs, reward, terminate, truncate, info


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class CheckpointWrapper(gym.Wrapper):
    env_id: str

    def __init__(self, env: gym.Env, env_id: str):
        super().__init__(env)
        assert hasattr(self.env, "get_checkpoint") and callable(self.env.get_checkpoint)
        assert hasattr(self.env, "load_checkpoint") and callable(
            self.env.load_checkpoint
        )
        self.env_id = env_id

    def get_checkpoint(self) -> tuple[str, dict]:
        ckpt: dict = self.env.get_checkpoint()
        return (self.env_id, ckpt)

    def load_checkpoint(self, ckpts: list[tuple[str, dict]]) -> None:
        my_ckpt = None
        for env_id, ckpt in ckpts:
            if env_id == self.env_id:
                my_ckpt = ckpt
                break
        if my_ckpt is None:
            raise ValueError(
                f"Could not load checkpoint, no checkpoint found with id {self.env_id}. Checkpoint IDs: ",
                [env_id for env_id, _ in ckpts],
            )
        self.env.load_checkpoint(my_ckpt)


def get_env_rng_checkpoint(env: SawyerXYZEnv) -> dict[str, dict]:
    return {  # pyright: ignore [reportReturnType]
        "np_random_state": env.np_random.bit_generator.state,
        "action_space_rng_state": env.action_space.np_random.bit_generator.state,
        "obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
        "goal_space_rng_state": env.goal_space.np_random.bit_generator.state,  # type: ignore
    }


def set_env_rng(env: SawyerXYZEnv, state: dict[str, dict]) -> None:
    assert "np_random_state" in state
    assert "action_space_rng_state" in state
    assert "obs_space_rng_state" in state
    assert "goal_space_rng_state" in state

    env.np_random.bit_generator.state = state["np_random_state"]
    env.action_space.np_random.bit_generator.state = state["action_space_rng_state"]
    env.observation_space.np_random.bit_generator.state = state["obs_space_rng_state"]
    env.goal_space.np_random.bit_generator.state = state["goal_space_rng_state"]  # type: ignore
```

### src_test/0_init.py

*大小: 1.4 KB | Token: 362*

```python
import gymnasium as gym
import ppo_test
import time
import random
import mujoco

env_name = 'peg-insert-side-v3'

# env = metaworld.make_mt_envs(
#     'peg-insert-side-v3',
#     render_mode='human',
#     width=1080,
#     height=1920
# )

env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
env = env_class(render_mode='human', width=1080, height=1920)
benchmark = ppo_test.MT1(env_name)
task = benchmark.train_tasks[0]  # 使用第一个训练任务
env.set_task(task)

# env = gym.make('Meta-World/MT1', env_name=env_name, render_mode='human', width=1080, height=1920)

from ppo_test.policies import SawyerPegInsertionSideV3Policy
policy = SawyerPegInsertionSideV3Policy()

obs, info = env.reset()

# 6. 循环执行直到任务成功
done = False
count = 0

mujoco_env = env.unwrapped
mujoco_env.mujoco_renderer.viewer.cam.azimuth = 135
# mujoco_env.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
# mujoco_env.mujoco_renderer.viewer.cam.fixedcamid = 2

while count < 500 and not done:
    # 渲染环境
    env.render()

    # 根据当前观测值获取动作
    action = policy.get_action(obs)
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 检查任务是否成功
    if info['success'] > 0.5:
        print("任务成功！")
        done = True
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
env.close()
```

### src_test/1-fix_cam.py

*大小: 7.1 KB | Token: 1.8K*

```python
import gymnasium as gym
import ppo_test
import time

def inspect_env_structure(env):
    """检查环境结构以找到viewer"""
    print("环境检查:")
    print(f"环境类型: {type(env)}")
    print(f"Unwrapped类型: {type(env.unwrapped)}")
    
    # 检查环境属性
    attrs = [attr for attr in dir(env.unwrapped) if not attr.startswith('__')]
    viewer_attrs = [attr for attr in attrs if 'view' in attr.lower() or 'render' in attr.lower()]
    print(f"与viewer/render相关的属性: {viewer_attrs}")
    
    # 检查是否有特定属性
    check_attrs = ['viewer', '_viewer', '_viewers', 'mujoco_renderer', 'renderer']
    for attr in check_attrs:
        if hasattr(env.unwrapped, attr):
            value = getattr(env.unwrapped, attr)
            print(f"{attr}: {type(value)} = {value}")
    
    return env.unwrapped

def set_camera_view_v2(env, distance=1.5, azimuth=90.0, elevation=-30.0, lookat=None):
    """
    改进的相机设置函数，适用于新版本的gymnasium/mujoco
    """
    if lookat is None:
        lookat = [0.0, 0.6, 0.2]
    
    mujoco_env = env.unwrapped
    
    # 确保先渲染一次以初始化viewer
    if not hasattr(mujoco_env, '_initialized_viewer'):
        env.render()
        mujoco_env._initialized_viewer = True
    
    # 方法1: 通过mujoco_renderer访问
    if hasattr(mujoco_env, 'mujoco_renderer'):
        renderer = mujoco_env.mujoco_renderer
        print(f"Found mujoco_renderer: {type(renderer)}")
        if hasattr(renderer, 'viewer'):
            viewer = renderer.viewer
            if viewer and hasattr(viewer, 'cam'):
                print("Setting camera via mujoco_renderer.viewer")
                viewer.cam.distance = distance
                viewer.cam.azimuth = azimuth
                viewer.cam.elevation = elevation
                viewer.cam.lookat[:] = lookat
                return True
    
    # 方法2: 通过_viewers字典访问
    if hasattr(mujoco_env, '_viewers'):
        viewers = mujoco_env._viewers
        print(f"Found _viewers: {viewers}")
        if viewers:
            for mode, viewer in viewers.items():
                if viewer and hasattr(viewer, 'cam'):
                    print(f"Setting camera via _viewers[{mode}]")
                    viewer.cam.distance = distance
                    viewer.cam.azimuth = azimuth
                    viewer.cam.elevation = elevation
                    viewer.cam.lookat[:] = lookat
                    return True
    
    # 方法3: 查找所有可能的viewer属性
    for attr_name in dir(mujoco_env):
        if 'view' in attr_name.lower():
            try:
                attr_value = getattr(mujoco_env, attr_name)
                if attr_value and hasattr(attr_value, 'cam'):
                    print(f"Setting camera via {attr_name}")
                    attr_value.cam.distance = distance
                    attr_value.cam.azimuth = azimuth
                    attr_value.cam.elevation = elevation
                    attr_value.cam.lookat[:] = lookat
                    return True
            except:
                continue
    
    # 方法4: 尝试通过render_mode直接访问
    try:
        # 获取当前的viewer
        current_viewer = None
        if hasattr(mujoco_env, '_get_viewer'):
            current_viewer = mujoco_env._get_viewer('human')
        elif hasattr(mujoco_env, 'viewer'):
            current_viewer = mujoco_env.viewer
        
        if current_viewer and hasattr(current_viewer, 'cam'):
            print("Setting camera via direct viewer access")
            current_viewer.cam.distance = distance
            current_viewer.cam.azimuth = azimuth
            current_viewer.cam.elevation = elevation
            current_viewer.cam.lookat[:] = lookat
            return True
    except Exception as e:
        print(f"Direct viewer access failed: {e}")
    
    print("无法找到或设置viewer相机")
    return False

def create_camera_wrapper(env_name, camera_config=None):
    """
    创建一个能够正确设置相机的环境包装器
    """
    if camera_config is None:
        camera_config = {
            'distance': 1.2,
            'azimuth': 90.0,
            'elevation': -20.0,
            'lookat': [0.0, 0.6, 0.2]
        }
    
    class SmartCameraWrapper(gym.Wrapper):
        def __init__(self, env, camera_config):
            super().__init__(env)
            self.camera_config = camera_config
            self._camera_attempts = 0
            self._camera_success = False
            
        def render(self):
            # 先执行原始渲染
            result = self.env.render()
            
            # 如果还没成功设置相机，继续尝试
            if not self._camera_success and self._camera_attempts < 5:
                self._camera_attempts += 1
                print(f"尝试设置相机 (第{self._camera_attempts}次)")
                success = set_camera_view_v2(self.env, **self.camera_config)
                if success:
                    self._camera_success = True
                    print("相机设置成功!")
                elif self._camera_attempts == 1:
                    # 第一次失败时，打印环境结构信息
                    inspect_env_structure(self.env)
            
            return result
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            # 重置时不重置相机成功状态，因为viewer可能仍然存在
            return obs, info
        
        def step(self, action):
            # 每隔一段时间重新尝试设置相机
            result = self.env.step(action)
            if hasattr(self, '_step_count'):
                self._step_count += 1
            else:
                self._step_count = 1
                
            # 每50步尝试一次相机设置（以防viewer被重新创建）
            if self._step_count % 50 == 0 and not self._camera_success:
                set_camera_view_v2(self.env, **self.camera_config)
                
            return result
    
    # 创建基础环境
    base_env = ppo_test.make_mt_envs(
        env_name,
        render_mode='human',
        width=1080,
        height=1920
    )
    
    return SmartCameraWrapper(base_env, camera_config)


if __name__ == "__main__":
    print("方法1: 使用智能相机包装器")
    camera_config = {
        'distance': 1.5,
        'azimuth': 135.0,
        'elevation': -25.0,
        'lookat': [0.0, 0.6, 0.15]
    }
    
    try:
        env = create_camera_wrapper('peg-insert-side-v3', camera_config)
        
        from ppo_test.policies import SawyerPegInsertionSideV3Policy
        policy = SawyerPegInsertionSideV3Policy()
        
        obs, info = env.reset()
        
        for i in range(200):
            env.render()
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['success'] > 0.5:
                print("任务成功!")
                break
                
            time.sleep(0.02)
        
        env.close()
        
    except Exception as e:
        print(f"包装器方法失败: {e}")
```

### force_v4.py

*大小: 3.1 KB | Token: 756*

```python
if force_magnitude > 10:
                # 1. 获取peg当前的姿态（从mujoco四元数转换为scipy Rotation对象）
                peg_current_rotation = Rotation.from_quat(o_d["peg_rot"])
                # 2. 定义在peg局部坐标系下的力臂向量
                lever_arm_local = np.array([-1, 0.0, 0.0])
                # 3. 将力臂向量从局部坐标系转换到世界坐标系
                lever_arm_world = peg_current_rotation.apply(lever_arm_local)
                corrective_torque_vector = np.cross(lever_arm_world, force_vector)
                # 5. 限制修正速度，保证稳定性
                speed = np.deg2rad(1) # 最大修正角速度
                torque_magnitude = np.linalg.norm(corrective_torque_vector)
                if torque_magnitude > 1e-6: # 避免除以零
                    unit_torque_axis = corrective_torque_vector / torque_magnitude
                    # 如果计算出的旋转速度超过上限，则使用上限速度
                    if torque_magnitude > speed:
                        increment_rotvec = unit_torque_axis * speed
                    else:
                        increment_rotvec = corrective_torque_vector
                    r_correction = Rotation.from_rotvec(increment_rotvec)
                    self.ini_r = r_correction * self.ini_r
                    desir_pos = pos_curr
                    
                    print(f"Force Detected: {force_magnitude:.2f} N. Applying rotational correction.")
                    print(f"Corrected Target Euler: {r_correction.as_euler('xyz', degrees=True)}")
                    
                    
# ppo_test/sawyer_xyz_env.py

# ... (在 SawyerXYZEnv 类中) ...
def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
    # ... (前面的速度计算等逻辑保持不变) ...
    
    # --- 导纳控制逻辑 ---
    pos_deviation = np.zeros(3)
    r_correct = Rotation.from_quat([0, 0, 0, 1])

    if hasattr(self, '_get_pegHead_force') and callable(getattr(self, '_get_pegHead_force')):
        try:
            # 1. 获取外部力 F_ext
            force = self._get_pegHead_force()

            # 2. 位置导纳: (逻辑不变，刚度由初始化参数决定)
            damping_force = self.admittance_damping * tcp_vel
            net_force = force - damping_force
            safe_stiffness = np.where(self.admittance_stiffness == 0, 1e-6, self.admittance_stiffness)
            pos_deviation = net_force / safe_stiffness

            # 3. 旋转导纳 (关键修改点)
            # 我们只希望响应 YZ 平面的力和姿态
            # - 力 Fy (force[1]) 会导致绕 X 轴的旋转 (翻滚 Roll)，这是 YZ 平面内的姿态调整。
            # - 我们不再响应 Fx (force[0])，以保持 X 轴指向的刚性。
            # highlight-start
            rotation_vec = (1.0 / self.rotational_stiffness) * np.array([force[1], 0, 0])
            # highlight-end
            if np.linalg.norm(rotation_vec) > 1e-4:
                r_correct = Rotation.from_rotvec(rotation_vec)

        except Exception:
            pass # 如果出错则不修正
    
    # ... (后面的应用控制逻辑保持不变) ...
```

### plant.py

*大小: 1.0 KB | Token: 287*

```python
import ppo_test
import time
import numpy as np
import numpy.typing as npt
import random
import pandas as pd

from typing import Any, Tuple
from scipy.spatial.transform import Rotation, Slerp

from ppo_test.policies.action import Action
from ppo_test.policies.policy import Policy, assert_fully_parsed, move

env_name = 'peg-insert-side-v3'
env_class = ppo_test.env_dict.ALL_V3_ENVIRONMENTS[env_name]
env = env_class(render_mode='rgb_array', width=1280, height=720) # human or rgb_array
benchmark = ppo_test.MT1(env_name)
task = random.choice(benchmark.train_tasks)
env.set_task(task)
timestep = env.model.opt.timestep

print(f"Current simulation timestep: {timestep}")

obs, info = env.reset()

done = False
count = 0
while count < 500 and not done:
    env.render()
    action = np.zeros(7)
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    if info.get('success', 0.0) > 0.5:
        print("任务成功！")
        # time.sleep(1)
        break
    # time.sleep(0.01)
    count += 1

env.close()
```

### show.py

*大小: 3.5 KB | Token: 803*

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# mpl_toolkits.mplot3d is necessary for '3d' projection
from mpl_toolkits.mplot3d import Axes3D

def animate_force_vector(filepath="force_analysis.csv", episode_to_animate=1):
    """
    为特定轮次的数据创建一个力的3D矢量动画。

    Args:
        filepath (str): CSV数据文件的路径。
        episode_to_animate (int): 您希望生成动画的轮次（episode）编号。
    """
    # --- 1. 加载并准备数据 ---
    print("正在加载数据...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{filepath}'。")
        print("请确保CSV文件与此脚本位于同一目录下。")
        return

    # 筛选出要制作动画的特定轮次的数据
    episode_df = df[df['episode'] == episode_to_animate].reset_index()
    if episode_df.empty:
        print(f"错误: 在CSV文件中找不到轮次 {episode_to_animate} 的数据。")
        return

    # --- 2. 设置3D绘图区 ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 计算一个缩放因子，使最长的矢量长度为0.8，以获得最佳观感
    max_magnitude = df['magnitude'].max()
    scale_factor = 1.5

    # --- 3. 定义动画更新函数 ---
    # 这个函数会为动画的每一帧被调用
    def update(frame):
        ax.cla()  # 清除上一帧的图像

        # 获取当前帧的数据
        row = episode_df.iloc[frame]
        magnitude = row['magnitude']
        direction = np.array([row['direction_x'], row['direction_y'], row['direction_z']])

        # 根据力的大小计算矢量的显示长度
        length = (magnitude / max_magnitude) * scale_factor if max_magnitude > 0 else 0
        vector = direction * length

        # 绘制代表力的矢量（从原点出发）
        ax.quiver(0, 0, 0,  # 矢量起点
                  vector[0], vector[1], vector[2],  # 矢量终点
                  color='r',
                  arrow_length_ratio=0.15, # 箭头相对于杆的长度比例
                  label='Force Vector'
                  )

        # --- 设置图表样式 ---
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X Direction')
        ax.set_ylabel('Y Direction')
        ax.set_zlabel('Z Direction')

        # 设置一个固定的、更合适的观察角度
        ax.view_init(elev=30, azim=45)

        # 添加标题和信息文本
        ax.set_title(f"Force Vector Animation (Episode {episode_to_animate})")
        ax.text2D(0.05, 0.95, f"Step: {int(row['step'])}", transform=ax.transAxes, fontsize=12)
        ax.text2D(0.05, 0.90, f"Magnitude: {magnitude:.3f}", transform=ax.transAxes, fontsize=12)

    # --- 4. 创建并保存动画 ---
    num_frames = len(episode_df)
    # interval参数控制帧之间的延迟（毫秒），影响播放速度
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1 )

    output_filename = f'force_animation_episode_{episode_to_animate}.gif'
    print(f"正在保存动画到 {output_filename} ...")
    print("这可能需要一些时间，具体取决于您的数据量大小。")
    anim.save(output_filename, writer='pillow')
    print(f"动画保存成功！请查看 {output_filename}")
    plt.close()


if __name__ == '__main__':
    # 您可以在这里更改想制作动画的轮次编号
    EPISODE_NUMBER = 1
    animate_force_vector(episode_to_animate=EPISODE_NUMBER)
```

### test.py

*大小: 13.2 KB | Token: 3.6K*

```python
import re

from matplotlib.pyplot import plot
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

    def __init__(self, force_feedback_gain=1, force_threshold=15):

        super().__init__()
        self.force_feedback_gain = force_feedback_gain
        self.force_threshold = force_threshold
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_quat([0,1,0,1])

    def reset(self):
        print("Resetting policy stage to 1.")
        self.current_stage = 1
        self.gasp = False
        self.prev_r_error_rotvec = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.e_im = np.zeros(6)
        self.e_dot_im = np.zeros(6)
        self.ini_r = Rotation.from_quat([0,1,0,1])

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
        
        if self.current_stage == 4 and np.linalg.norm(o_d["pegHead_force"]) > 5:
            delta_pos, delta_rot_euler = self._pos_im(o_d["pegHead_force"], o_d["pegHead_torque"], delta_pos, delta_rot_euler)
        
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
            if np.linalg.norm(pos_curr[1:] - pos_hole[1:]) < 0.03: self.current_stage = 4
            return pos_hole + np.array([0.4, 0.0, 0.0]), self.ini_r
        if self.current_stage == 4:
            return pos_hole + np.array([0.1, 0.0, 0.0]), self.ini_r
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
```

### visualize_forces.py

*大小: 2.6 KB | Token: 644*

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_force_data(filepath="force_analysis.csv"):
    """
    读取力分析的CSV文件并生成可视化图表。
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"错误: 未找到文件 '{filepath}'。")
        print("请确保您已经运行了仿真脚本来生成CSV文件。")
        return

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    print(f"成功读取 {filepath}，开始生成图表...")

    # --- 图表1: 所有轮次中，力的大小随时间的变化 ---
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="step", y="magnitude", hue="episode", palette="viridis", legend="full")
    plt.title("Force Magnitude Over Time (All Episodes)", fontsize=16)
    plt.xlabel("Simulation Step")
    plt.ylabel("Force Magnitude")
    plt.legend(title="Episode")
    plt.tight_layout()
    plt.savefig("force_magnitude_vs_time.png")
    print("图表已保存: force_magnitude_vs_time.png")
    plt.close()

    # --- 图表2: 每个轮次中，力的分量随时间的变化 ---
    # 使用 melt 函数重塑数据，以便用 seaborn 进行分面绘图
    df_melted = df.melt(id_vars=['episode', 'step'], 
                        value_vars=['direction_x', 'direction_y', 'direction_z'],
                        var_name='component', 
                        value_name='value')

    g = sns.relplot(
        data=df_melted,
        x="step", y="value",
        hue="component", col="episode",
        kind="line", col_wrap=3,  # 每行显示的图表数量
        height=4, aspect=1.2,
        palette="bright"
    )
    g.fig.suptitle("Force Components Over Time by Episode", y=1.03, fontsize=16)
    g.set_axis_labels("Simulation Step", "Direction Component Value")
    g.set_titles("Episode {col_name}")
    g.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("force_components_vs_time.png")
    print("图表已保存: force_components_vs_time.png")
    plt.close()

    # --- 图表3: 每个轮次的最大作用力对比 ---
    plt.figure(figsize=(10, 6))
    max_forces = df.groupby('episode')['magnitude'].max().reset_index()
    sns.barplot(data=max_forces, x="episode", y="magnitude", palette="plasma")
    plt.title("Maximum Force Magnitude per Episode", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Maximum Force Magnitude")
    plt.tight_layout()
    plt.savefig("max_force_per_episode.png")
    print("图表已保存: max_force_per_episode.png")
    plt.close()

    print("\n所有图表生成完毕！")

if __name__ == "__main__":
    visualize_force_data()
```
