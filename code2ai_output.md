## 文件结构

```
📁 .
  📁 metaworld
    📁 policies
      📄 __init__.py
      📄 action.py
      📄 policy.py
      📄 sawyer_peg_insertion_side_v3_policy.py
    📁 utils
      📄 reward_utils.py
      📄 rotation.py
    📁 xml
      📄 basic_scene.xml
      📄 peg_block_dependencies.xml
      📄 peg_block.xml
      📄 peg_insert_dependencies.xml
      📄 sawyer_peg_insertion_side.xml
      📄 xyz_base_dependencies.xml
      📄 xyz_base.xml
    📄 __init__.py
    📄 asset_path_utils.py
    📄 env_dict.py
    📄 evaluation.py
    📄 sawyer_peg_insertion_side_v3.py
    📄 sawyer_xyz_env.py
    📄 types.py
    📄 wrappers.py
```

## 源文件

### metaworld/policies/__init__.py

*大小: 282 B | Token: 78*

```python
from metaworld.policies.sawyer_peg_insertion_side_v3_policy import (
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

### metaworld/policies/action.py

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

### metaworld/policies/policy.py

*大小: 2.3 KB | Token: 647*

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
    from_xyz: npt.NDArray[Any], to_xyz: npt.NDArray[Any], p: float
) -> npt.NDArray[Any]:
    """Computes action components that help move from 1 position to another.

    Args:
        from_xyz: The coordinates to move from (usually current position)
        to_xyz: The coordinates to move to
        p: constant to scale response

    Returns:
        Response that will decrease abs(to_xyz - from_xyz)
    """
    error = to_xyz - from_xyz
    response = p * error
    # if np.any(np.absolute(response) > 1.0):
    #     warnings.warn(
    #         "Constant(s) may be too high. Environments clip response to [-1, 1]"
    #     )

    return response


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

### metaworld/policies/sawyer_peg_insertion_side_v3_policy.py

*大小: 2.7 KB | Token: 756*

```python
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
            "quat_hand": obs[3:7],
            "gripper_distance_apart": obs[7],
            "peg_pos": obs[8:11],
            "peg_rot": obs[11:15],
            "unused_info_curr_obs": obs[15:22],
            "_prev_obs": obs[22:44],
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
```

### metaworld/utils/reward_utils.py

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

### metaworld/utils/rotation.py

*大小: 2.7 KB | Token: 758*

```python
"""Rotation utilities for quaternion and Euler angle conversions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def quat2euler(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion in [w, x, y, z] format
        
    Returns:
        Euler angles in [roll, pitch, yaw] format (radians)
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler2quat(euler: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles in [roll, pitch, yaw] format (radians)
        
    Returns:
        Quaternion in [w, x, y, z] format
    """
    roll, pitch, yaw = euler
    
    # Abbreviations for the various angular functions
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def normalize_quat(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a quaternion to unit length.
    
    Args:
        quat: Quaternion in [w, x, y, z] format
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    return quat / norm


def quat_multiply(q1: npt.NDArray[np.float64], q2: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Multiply two quaternions.
    
    Args:
        q1: First quaternion in [w, x, y, z] format
        q2: Second quaternion in [w, x, y, z] format
        
    Returns:
        Product quaternion in [w, x, y, z] format
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])
```

### metaworld/xml/basic_scene.xml

*大小: 3.1 KB | Token: 806*

```xml
<mujocoinclude>
    <option timestep='0.0025' iterations="50" tolerance="1e-10" solver="Newton" jacobian="dense" cone="elliptic"/>

    <asset>
        <!-- night sky -->
        <!-- <texture name="skybox" type="skybox" builtin="gradient" rgb1=".08 .09 .10" rgb2="0 0 0"
               width="800" height="800" mark="random" markrgb=".8 .8 .8"/> -->
        <texture type="skybox" builtin="gradient" rgb1=".50 .495 .48" rgb2=".50 .495 .48" width="32" height="32"/>
        <texture name="T_table" type="cube" file="./textures/wood2.png"/>
        <texture name="T_floor" type="2d" file="./textures/floor2.png"/>

        <material name="basic_floor" texture="T_floor" texrepeat="12 12" shininess=".3" specular="0.5"
                  reflectance="0.2"/>
        <material name="table_wood" texture="T_table" shininess=".3" specular="0.5"/>
        <material name="table_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>

        <mesh file="./table/tablebody.stl" name="tablebody" scale="1 1 1"/>
        <mesh file="./table/tabletop.stl" name="tabletop" scale="1 1 1"/>
    </asset>

    <asset>
        <texture name="T_wallmetal" type="cube" file="./textures/metal.png"/>
        <material name="wall_metal" texture="T_wallmetal" shininess="1" reflectance="1" specular=".5"/>
    </asset>

    <visual>
        <map fogstart="1.5" fogend="5" force="0.1" znear="0.01"/>
        <quality shadowsize="4096" offsamples="4"/>

        <headlight ambient="0.4 0.4 0.4"/>

    </visual>

    <worldbody>
        <light castshadow="false" directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='-1 -1 1'
               dir='1 1 -1'/>
        <light directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='1 -1 1' dir='-1 1 -1'/>
        <light castshadow="false" directional='true' diffuse='.3 .3 .3' specular='0.3 0.3 0.3' pos='0 1 1'
               dir='0 -1 -1'/>
        <body name="tablelink" pos="0 .6 0">
            <geom material="table_wood" group="1" type="box" size=".7 .4 .027" pos="0 0 -.027" conaffinity="0"
                  contype="0"/>
            <geom material="table_wood" group="1" mesh="tablebody" pos="0 0 -0.65" type="mesh" conaffinity="0"
                  contype="0"/>
            <geom material="table_col" group="4" pos="0.0 0.0 -0.46" size="0.7 0.4 0.46" type="box" conaffinity="1"
                  contype="0"/>
        </body>

        <body name="RetainingWall" pos="0.0 0.6 0.06">
            <geom material="wall_metal" type="box" size=".7 .01 .06" pos="0. -0.39 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".7 .01 .06" pos="0. 0.39 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".01 .38 .06" pos="-.69 0. 0." conaffinity="1" condim="3"
                  contype="0"/>
            <geom material="wall_metal" type="box" size=".01 .38 .06" pos=".69 0. 0." conaffinity="1" condim="3"
                  contype="0"/>
        </body>

        <geom name="floor" size="4 4 .1" pos="0 0 -.913" conaffinity="1" contype="1" type="plane" material="basic_floor"
              condim="3"/>

    </worldbody>

</mujocoinclude>
```

### metaworld/xml/peg_block_dependencies.xml

*大小: 1.2 KB | Token: 318*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>
        <texture name="T_peg_block_wood" type="cube" file="./textures/wood1.png"/>

      <material name="peg_block_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>
      <material name="peg_block_wood" texture="T_peg_block_wood" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_block_red" rgba=".55 0 0 1" shininess="1" reflectance=".7" specular=".5"/>

    </asset>
    <default>

      <default class="peg_block_base">
          <joint armature="0.001" damping="2" limited="true"/>
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <position ctrllimited="true" ctrlrange="0 1.57"/>
          <default class="peg_block_viz">
              <geom condim="4" type="mesh"/>
          </default>
          <default class="peg_block_col">
              <geom conaffinity="1" condim="3" contype="1" group="4" material="peg_block_col" solimp="0.99 0.99 0.01" solref="0.01 1"/>
          </default>
      </default>
    </default>

    <asset>
      <mesh file="./peg_block/block_inner.stl" name="block_inner"/>
        <mesh file="./peg_block/block_outer.stl" name="block_outer"/>
    </asset>

</mujocoinclude>
```

### metaworld/xml/peg_block.xml

*大小: 1.3 KB | Token: 340*

```xml
<mujocoinclude>
    <body childclass="peg_block_base">
      <geom material="peg_block_red" mesh="block_inner" pos="0 0 0.095"/>
      <geom material="peg_block_wood" mesh="block_outer" pos="0 0 0.1"/>
      <geom class="peg_block_col" pos="0 0 0.195" size="0.09 0.1 0.005" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0 0 0.05" size="0.09 0.096 0.05" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="-0.06 0 0.13" size="0.03 0.096 0.03" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0.06 0 0.13" size="0.03 0.096 0.03" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0 0 0.175" size="0.09 0.096 0.015" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>
      <geom class="peg_block_col" pos="-0.095 0 0.1" size="0.005 0.1 0.1" type="box" mass="1000"/>
      <site name="hole" pos="0 -.096 0.13" size="0.005" rgba="0 0.8 0 1"/>
      <site name="bottom_right_corner_collision_box_1" pos="0.1 -0.11 0.01" size="0.0001"/>
      <site name="top_left_corner_collision_box_1" pos="-0.1 -.15 0.096" size="0.0001"/>
      <site name="bottom_right_corner_collision_box_2" pos="0.1 -0.11 0.16" size="0.0001"/>
      <site name="top_left_corner_collision_box_2" pos="-0.1 -.17 0.19" size="0.0001"/>
    </body>
</mujocoinclude>
```

### metaworld/xml/peg_insert_dependencies.xml

*大小: 1006 B | Token: 252*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>
      <texture name="T_peg_wood" type="cube" file="./textures/wood1.png"/>

      <material name="peg_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0"/>
      <material name="peg_green" rgba="0 .5 0 1" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_black" rgba=".15 .15 .15 1" shininess="1" reflectance=".7" specular=".5"/>
      <material name="peg_wood" rgba=".55 .55 .55 1" texture="T_peg_wood" shininess="1" reflectance=".7" specular=".5"/>
    </asset>
    <default>
      <default class="peg_base">
          <!-- <joint armature="0.001" damping="2"/> -->
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <default class="peg_col">
              <geom conaffinity="1" condim="3" contype="1" group="4" material="peg_col" solimp="0.99 0.99 0.01" solref="0.01 1"/>
          </default>

      </default>
    </default>

</mujocoinclude>
```

### metaworld/xml/sawyer_peg_insertion_side.xml

*大小: 1.3 KB | Token: 345*

```xml
<mujoco>
    <include file="./basic_scene.xml"/>
    <include file="./peg_block_dependencies.xml"/>
    <include file="./peg_insert_dependencies.xml"/>
    <include file="./xyz_base_dependencies.xml"/>

    <worldbody>
      <include file="./xyz_base.xml"/>

        <body name="peg" pos="0 0.6 0.03">
          <inertial pos="0 0 0" mass="0.1" diaginertia="100000 100000 100000"/>
          <geom name="peg" euler="0 1.57 0" size="0.015 0.015 0.12" type="box" mass=".1" rgba="0.3 1 0.3 1" conaffinity="1" contype="1" group="1"/>
          <joint type="free" limited="false" damping="0.005"/>
          <site name="pegHead" pos="-0.1 0 0" size="0.005" rgba="0.8 0 0 1"/>
          <site name="pegEnd" pos="0.1 0 0" size="0.005" rgba="0.8 0 0 1"/>
          <site name="pegGrasp" pos=".03 .0 .01" size="0.005" rgba="0.8 0 0 1"/>
        </body>

        <body name="box" euler="0 0 1.57" pos="-0.3 0.6 0">
          <include file="./peg_block.xml"/>
        </body>
        <site name="goal" pos="0 0.6 0.05" size="0.01" rgba="0.8 0 0 1"/>

    </worldbody>

    <actuator>
        <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400"  user="1"/>
        <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400"  user="1"/>
    </actuator>

    <equality>
        <weld body1="mocap" body2="hand" solref="0.02 1"></weld>
    </equality>
</mujoco>
```

### metaworld/xml/xyz_base_dependencies.xml

*大小: 1.4 KB | Token: 366*

```xml
<mujocoinclude>
    <compiler angle="radian" inertiafromgeom="auto" inertiagrouprange="4 5"/>
    <asset>

      <material name="xyz_col" rgba="0.3 0.3 1.0 0.5" shininess="0" specular="0.5"/>

      <mesh file="./xyz_base/base.stl" name="base"/>
      <mesh file="./xyz_base/eGripperBase.stl" name="eGripperBase"/>
      <mesh file="./xyz_base/head.stl" name="head"/>
      <mesh file="./xyz_base/l0.stl" name="l0"/>
      <mesh file="./xyz_base/l1.stl" name="l1"/>
      <mesh file="./xyz_base/l2.stl" name="l2"/>
      <mesh file="./xyz_base/l3.stl" name="l3"/>
      <mesh file="./xyz_base/l4.stl" name="l4"/>
      <mesh file="./xyz_base/l5.stl" name="l5"/>
      <mesh file="./xyz_base/l6.stl" name="l6"/>
      <mesh file="./xyz_base/pedestal.stl" name="pedestal"/>
    </asset>

    <default>

      <default class="xyz_base">
          <joint armature="0.001" damping="2" limited="true"/>
          <geom conaffinity="0" contype="0" group="1" type="mesh"/>
          <position ctrllimited="true" ctrlrange="0 1.57"/>
          <default class="base_viz">
              <geom conaffinity="0" condim="4" contype="0" group="1" margin="0.001" solimp=".8 .9 .01" solref=".02 1" type="mesh"/>
          </default>
          <default class="base_col">
              <geom conaffinity="1" condim="4" contype="1" group="4" margin="0.001" material="xyz_col" solimp=".8 .9 .01" solref=".02 1"/>
          </default>
      </default>
    </default>

</mujocoinclude>
```

### metaworld/xml/xyz_base.xml

*大小: 19.1 KB | Token: 5.4K*

```xml
<mujocoinclude>
  <!--
  Usage:

  <mujoco>
  	<compiler meshdir="../meshes/sawyer" ...></compiler>
  	<include file="shared_config.xml"></include>
      (new stuff)
  	<worldbody>
  		<include file="sawyer_xyz_base.xml"></include>
          (new stuff)
  	</worldbody>
  </mujoco>
  -->

      <camera pos="0 0.5 1.5" name="topview" />
      <camera name="corner" mode="fixed" pos="-1.1 -0.4 0.6" xyaxes="-1 1 0 -0.2 -0.2 -1"/>
      <camera name="corner2" fovy="60" mode="fixed" pos="1.3 -0.2 1.1" euler="3.9 2.3 0.6"/>
      <camera name="corner3" fovy="45" mode="fixed" pos="0.9 0 1.5" euler="3.5 2.7 1"/>
      <!--<geom name="floor" type="plane" pos="0 0 -.9" size="10 10 10"-->
            <!--rgba="0 0 0 1" contype="15" conaffinity="15" />-->
      <!--<geom name="tableTop" type="box" pos="0 0.6 -0.45" size="0.4 0.2 0.45"
            rgba=".6 .6 .5 1" contype="15" conaffinity="15" />-->
      <!-- <geom name="tableTop" type="plane" pos="0 0.6 0" size="0.4 0.4 0.5" -->
            <!-- rgba=".6 .6 .5 1" contype="1" conaffinity="1" friction="2 0.1 0.002" material="light_wood_v3"/> -->

      <body name="base" childclass="xyz_base" pos="0 0 0">
          <site name="basesite" pos="0 0 0" size="0.01" />
          <inertial pos="0 0 0" mass="0" diaginertia="0 0 0" />
          <body name="controller_box" pos="0 0 0">
              <inertial pos="-0.325 0 -0.38" mass="46.64" diaginertia="1.71363 1.27988 0.809981" />
              <geom size="0.11 0.2 0.265" pos="-0.325 0 -0.38" type="box" rgba="0.2 0.2 0.2 1"/>
          </body>
          <body name="pedestal_feet" pos="0 0 0">
              <inertial pos="-0.1225 0 -0.758" mass="167.09" diaginertia="8.16095 9.59375 15.0785" />
              <geom size="0.385 0.35 0.155" pos="-0.1225 0 -0.758" type="box" rgba="0.2 0.2 0.2 1"
                    contype="0"
                    conaffinity="0"
              />
          </body>
          <body name="torso" pos="0 0 0">
              <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
              <geom size="0.05 0.05 0.05" type="box" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 1" />
          </body>
          <body name="pedestal" pos="0 0 0">
              <inertial pos="0 0 0" quat="0.659267 -0.259505 -0.260945 0.655692" mass="60.864" diaginertia="6.0869 5.81635 4.20915" />
              <geom pos="0.26 0.345 -0.91488" quat="0.5 0.5 -0.5 -0.5" type="mesh" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 1" mesh="pedestal" />
              <geom size="0.18 0.31" pos="-0.02 0 -0.29" type="cylinder" rgba="0.2 0.2 0.2 0" />
          </body>
          <body name="right_arm_base_link" pos="0 0 0">
              <inertial pos="-0.0006241 -2.8025e-05 0.065404" quat="-0.209285 0.674441 0.227335 0.670558" mass="2.0687" diaginertia="0.00740351 0.00681776 0.00672942" />
              <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="base" />
              <geom size="0.08 0.12" pos="0 0 0.12" type="cylinder" rgba="0.5 0.1 0.1 0" />
              <body name="right_l0" pos="0 0 0.08">
                  <inertial pos="0.024366 0.010969 0.14363" quat="0.894823 0.00899958 -0.170275 0.412573" mass="5.3213" diaginertia="0.0651588 0.0510944 0.0186218" />
                  <joint name="right_j0" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0503 3.0503" damping="10"/>
                  <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l0" />
                  <body name="head" pos="0 0 0.2965">
                      <inertial pos="0.0053207 -2.6549e-05 0.1021" quat="0.999993 7.08405e-05 -0.00359857 -0.000626247" mass="1.5795" diaginertia="0.0118334 0.00827089 0.00496574" />
                      <!-- <joint name="head_pan" pos="0 0 0" axis="0 0 1" limited="true" range="-5.0952 0.9064" damping="10"/> -->
                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="head" />
                      <!-- <geom size="0.18" pos="0 0 0.08" rgba="0.5 0.1 0.1 0" /> -->
                      <body name="screen" pos="0.03 0 0.105" quat="0.5 0.5 0.5 0.5">
                          <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                          <geom size="0.12 0.07 0.001" type="box" contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.2 0" />
                          <!-- <geom size="0.001" rgba="0.2 0.2 0.2 0" /> -->
                      </body>
                      <body name="head_camera" pos="0.0228027 0 0.216572" quat="0.342813 -0.618449 0.618449 -0.342813">
                          <inertial pos="0.0228027 0 0.216572" quat="0.342813 -0.618449 0.618449 -0.342813" mass="0" diaginertia="0 0 0" />
                          <site name="headsite" pos="0 0 0" size="0.01" />
                      </body>
                  </body>
                  <body name="right_torso_itb" pos="-0.055 0 0.22" quat="0.707107 0 -0.707107 0">
                      <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                  </body>
                  <body name="right_l1" pos="0.081 0.05 0.237" quat="0.5 -0.5 0.5 0.5">
                      <inertial pos="-0.0030849 -0.026811 0.092521" quat="0.424888 0.891987 0.132364 -0.0794296" mass="4.505" diaginertia="0.0224339 0.0221624 0.0097097" />
                      <!--<joint name="right_j1" pos="0 0 0" axis="0 0 1" limited="true" range="-3.8095 2.2736" damping="10"/>-->
                      <joint name="right_j1" pos="0 0 0" axis="0 0 1"
                             limited="true" range="-3.8 -0.5"
                             damping="10"/>
                      <!--<joint name="right_j1" pos="0 0 0" axis="0 0 1" limited="true" range="0.8095 2.2736" damping="10"/>-->
                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l1" />
                      <!-- <geom size="0.07" pos="0 0 0.1225" rgba="0.5 0.1 0.1 0" /> -->
                      <body name="right_l2" pos="0 -0.14 0.1425" quat="0.707107 0.707107 0 0">
                          <inertial pos="-0.00016044 -0.014967 0.13582" quat="0.707831 -0.0524761 0.0516007 0.702537" mass="1.745" diaginertia="0.0257928 0.025506 0.00292515" />
                          <joint name="right_j2" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0426 3.0426" damping="10"/>
                          <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l2" />
                          <geom size="0.06 0.17" pos="0 0 0.08" type="cylinder" rgba="0.5 0.1 0.1 0" />
                          <body name="right_l3" pos="0 -0.042 0.26" quat="0.707107 -0.707107 0 0">
                              <site name="armsite" pos="0 0 0" size="0.01" />
                              <inertial pos="-0.0048135 -0.0281 -0.084154" quat="0.902999 0.385391 -0.0880901 0.168247" mass="2.5097" diaginertia="0.0102404 0.0096997 0.00369622" />
                              <joint name="right_j3" pos="0 0 0" axis="0 0 1" limited="true" range="-3.0439 3.0439" damping="10"/>
                              <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l3" />
                              <!-- <geom size="0.06" pos="0 -0.01 -0.12" rgba="0.5 0.1 0.1 0" /> -->
                              <body name="right_l4" pos="0 -0.125 -0.1265" quat="0.707107 0.707107 0 0">
                                  <inertial pos="-0.0018844 0.0069001 0.1341" quat="0.803612 0.031257 -0.0298334 0.593582" mass="1.1136" diaginertia="0.0136549 0.0135493 0.00127353" />
                                  <joint name="right_j4" pos="0 0 0" axis="0 0 1" limited="true" range="-2.9761 2.9761" damping="10" />
                                  <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l4" />
                                  <geom size="0.045 0.15" pos="0 0 0.11" type="cylinder" rgba="0.5 0.1 0.1 0" />
                                  <body name="right_arm_itb" pos="-0.055 0 0.075" quat="0.707107 0 -0.707107 0">
                                      <inertial pos="0 0 0" mass="0.0001" diaginertia="1e-08 1e-08 1e-08" />
                                  </body>
                                  <body name="right_l5" pos="0 0.031 0.275" quat="0.707107 -0.707107 0 0">
                                      <inertial pos="0.0061133 -0.023697 0.076416" quat="0.404076 0.9135 0.0473125 0.00158335" mass="1.5625" diaginertia="0.00474131 0.00422857 0.00190672" />
                                      <joint name="right_j5" pos="0 0 0" axis="0 0 1" limited="true" range="-2.9761 2.9761" damping="10"/>
                                      <geom type="mesh" contype="0" conaffinity="0" group="1" rgba="0.5 0.1 0.1 1" mesh="l5" />
                                      <!-- <geom size="0.06" pos="0 0 0.1" rgba="0.5 0.1 0.1 0" /> -->
                                      <body name="right_hand_camera" pos="0.039552 -0.033 0.0695" quat="0.707107 0 0.707107 0">
                                          <inertial pos="0.039552 -0.033 0.0695" quat="0.707107 0 0.707107 0" mass="0" diaginertia="0 0 0" />
                                      </body>
                                      <body name="right_wrist" pos="0 0 0.10541" quat="0.707107 0.707107 0 0">
                                          <inertial pos="0 0 0.10541" quat="0.707107 0.707107 0 0" mass="0" diaginertia="0 0 0" />
                                      </body>
                                      <body name="right_l6" pos="0 -0.11 0.1053" quat="0.0616248 0.06163 -0.704416 0.704416">
                                          <inertial pos="-8.0726e-06 0.0085838 -0.0049566" quat="0.479044 0.515636 -0.513069 0.491322" mass="0.3292" diaginertia="0.000360258 0.000311068 0.000214974" />
                                          <joint name="right_j6" pos="0 0 0" axis="0 0 1" limited="true" range="-4.7124 4.7124" damping="10"/>
                                          <geom type="mesh" contype="4" conaffinity="2" group="1" rgba="0.5 0.1 0.1 1" mesh="l6" />
                                          <geom size="0.055 0.025" pos="0 0.015 -0.01" type="cylinder" rgba="0.5 0.1 0.1 0" />
                                          <body name="right_hand" pos="0 0 0.0245" quat="0.707107 0 0 0.707107">
                                              <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                                              <geom type="mesh" contype="1" conaffinity="1" group="1" rgba="0.5 0.1 0.1 1" pos= "0 0 0.03" mesh="eGripperBase" />

                                              <geom size="0.035 0.014" pos="0 0 0.015" type="cylinder" rgba="0 0 0 1"/>
                                              <!-- <geom size="0.035 0.015" pos="0 0 0.02" type="cylinder" rgba="0.2 0.2 0.2 0"/> -->

  <!--  ================= BEGIN GRIPPER ================= /-->
                                              <!-- <body name="hand" pos="0 0 0"
                                                    quat="-1 0 1 0">
                                                  <geom class="1" name="Geomclaw" type="box" size="0.01 0.04 0.01"/>
                                                      <body name="rightclaw" pos=".03 -.03 0.0" >
                                                          <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                                                          <geom
                                                                  name="rightclaw_it" condim="4" contype="2" conaffinity="2" class="1" mass="0.08" type="box" pos="0 0 0" size="0.025 0.005 0.02" rgba="0.0 1.0 0.0 1.0" friction="1 0.05 0.01"
                                                                  euler="0 0 0.2"
                                                          />
                                                          <joint name="rc_close" type="slide" pos="0 0 0" axis="0 1 0" range="0 .04" user="008" limited="true"/>
                                                          <site name="endeffector2" pos=".015 .01 0" size="0.008" rgba="0.0 0.0 0.0 0.0" />
                                                      </body>
                                                      <body name="leftclaw" pos=".03 .03 0">
                                                          <inertial diaginertia="0.1 0.1 0.1" mass="4" pos="-0.01 0 0"></inertial>
                                                          <geom
                                                                  name="leftclaw_it" condim="4" contype="2" conaffinity="2" class="1" type="box" mass="0.08" pos="0 0 0" size="0.025 0.005 0.02" rgba="0.0 1.0 0.0 1.0" friction="1 0.05 0.01"
                                                                  euler="0 0 -0.2"
                                                          />
                                                          <joint name="lc_close" type="slide" pos="0 0 0" axis="0 -1 0" range="-.04 0" user="008" limited="true"/>
                                                          <site name="endeffector" pos=".015 -.01 0" size="0.008" rgba="0.0 0.0 0.0 0.0"  />
                                                      </body>
                                              </body> -->
                                              <body name="hand" pos="0 0 0.12" quat="-1 0 1 0">
                                                  <camera name="behindGripper" mode="track" pos="0 0 -0.5" quat="0 1 0 0" fovy="60" />
                                                  <camera name="gripperPOV" mode="track" pos="0.04 -0.06 0" quat="-1 -1.3 0 0" fovy="90" />

                                                  <site name="endEffector" pos="0.04 0 0" size="0.01" rgba='1 1 1 0' />
                                                  <geom name="rail" type="box" pos="-0.05 0 0" density="7850" size="0.005 0.055 0.005"  rgba="0.5 0.5 0.5 1.0" condim="3" friction="2 0.1 0.002"   />

                                                  <!--IMPORTANT: For rougher contact with gripper, set higher friciton values for the other interacting objects -->
                                                  <body name="rightclaw" pos="0 -0.05 0" >

                                                      <geom class="base_col" name="rightclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="1 1 1 1.0"   />

                                                      <joint name="r_close" pos="0 0 0" axis="0 1 0" range= "0 0.04" armature="100" damping="1000" limited="true"  type="slide"/>
                                                      <!-- <joint name="r_close" pos="0 0 0" axis="0 1 0" range= "0 0.03" armature="100" damping="1000" limited="true"  type="slide"/>  -->

                                                      <!-- <site name="rightEndEffector" pos="0.0 0.005 0" size="0.044 0.008 0.012" type='box' /> -->

                                                      <!-- <site name="rightEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="rightEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
                                                      <body name="rightpad" pos ="0 .003 0" >
                                                          <geom name="rightpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="1 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002" contype="1" conaffinity="1" mass="1"/>
                                                      </body>

                                                  </body>

                                                  <body name="leftclaw" pos="0 0.05 0">
                                                      <geom class="base_col" name="leftclaw_it" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015"  rgba="0 1 1 1.0"  />
                                                      <joint name="l_close" pos="0 0 0" axis="0 1 0" range= "-0.03 0" armature="100" damping="1000" limited="true"  type="slide"/>
                                                      <!-- <site name="leftEndEffector" pos="0.0 -0.005 0" size="0.044 0.008 0.012" type='box' /> -->
                                                      <!-- <site name="leftEndEffector" pos="0.035 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/> -->
                                                      <site name="leftEndEffector" pos="0.045 0 0" size="0.01" rgba="1.0 0.0 0.0 1.0"/>
                                                      <body name="leftpad" pos ="0 -.003 0" >
                                                          <geom name="leftpad_geom" condim="4" margin="0.001" type="box" user="0" pos="0 0 0" size="0.045 0.003 0.015" rgba="0 1 1 1.0" solimp="0.95 0.99 0.01" solref="0.01 1" friction="2 0.1 0.002"  contype="1" conaffinity="1" />
                                                      </body>

                                                  </body>
                                              </body>
  <!--  ================= END GRIPPER ================= /-->
                                          </body>
                                      </body>
                                  </body>
                                  <body name="right_l4_2" pos="0 0 0">
                                      <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                                      <!-- <geom size="0.06" pos="0 0.01 0.26"
                                            rgba="0.2 0.2 0.2 0"
                                            contype="0"
                                            conaffinity="0"
                                      /> -->
                                  </body>
                              </body>
                          </body>
                          <body name="right_l2_2" pos="0 0 0">
                              <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                              <!-- <geom size="0.06" pos="0 0 0.26" rgba="0.2 0.2 0.2 0"
                                    contype="0"
                                    conaffinity="0"
                              /> -->
                          </body>
                      </body>
                      <body name="right_l1_2" pos="0 0 0">
                          <inertial pos="1e-08 1e-08 1e-08" quat="0.820473 0.339851 -0.17592 0.424708" mass="1e-08" diaginertia="1e-08 1e-08 1e-08" />
                          <geom size="0.07 0.07" pos="0 0 0.035" type="cylinder" rgba="0.2 0.2 0.2 0"/>
                      </body>
                  </body>
              </body>
          </body>
      </body>

      <body mocap="true" name="mocap" pos="0 0 0">
          <!--For debugging, set the alpha to 1-->
          <!--<geom conaffinity="0" contype="0" pos="0 0 0" rgba="0.5 0.5 0.5 1" size="0.1 0.02 0.02" type="box"></geom>-->
          <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0.0 0.5 0.5 0" size="0.01" type="sphere"></geom>
          <site name="mocap" pos="0 0 0" rgba="0.0 0.5 0.5 0" size="0.01" type="sphere"></site>
      </body>

</mujocoinclude>
```

### metaworld/__init__.py

*大小: 19.8 KB | Token: 5.6K*

```python
"""The public-facing Metaworld API."""

from __future__ import annotations

import abc
import pickle
from collections import OrderedDict
from functools import partial
from typing import Any, Literal

import gymnasium as gym 
import numpy as np
import numpy.typing as npt

# noqa: D104
from gymnasium.envs.registration import register

import metaworld.env_dict as _env_dict
from metaworld.env_dict import (
    ALL_V3_ENVIRONMENTS,
    ALL_V3_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE,
)
from metaworld.sawyer_xyz_env import SawyerXYZEnv 
from metaworld.types import Task 
from metaworld.wrappers import (
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
    reward_function_version: Literal["v1", "v2"] = "v2",
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
        reward_function_version=reward_function_version,
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

### metaworld/asset_path_utils.py

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

### metaworld/env_dict.py

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
from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.sawyer_peg_insertion_side_v3 import SawyerPegInsertionSideEnvV3

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

### metaworld/evaluation.py

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

### metaworld/sawyer_peg_insertion_side_v3.py

*大小: 9.0 KB | Token: 2.5K*

```python
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.asset_path_utils import full_V3_path_for
from metaworld.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.types import InitConfigDict
from metaworld.utils import reward_utils


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

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        # Updated observation parsing for new 47-dimensional observation space
        # obs structure: hand_pos(3) + quat_hand(4) + gripper_distance_apart(1) + peg_pos(3) + peg_rot(4) + unused_info(7) + _prev_obs(22) + goal_pos(3) = 47
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

        info = {
            "success": success,
            "near_object": near_object,
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": reward,
            "insertion_phase": self.insertion_phase,
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

        self.maxPlacingDist = (
            np.linalg.norm(
                np.array(
                    [self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]
                )
                - np.array(self._target_pos)
            )
            + self.heightTarget
        )

        # 重置插入阶段
        self.insertion_phase = "approach"
        
        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        tcp = self.tcp_center
        # Updated observation parsing for new 47-dimensional observation space
        obj = obs[8:11]  # peg_pos, index changed from 4:7 to 8:11
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
```

### metaworld/sawyer_xyz_env.py

*大小: 34.1 KB | Token: 9.6K*

```python
"""Base classes for all the envs."""

from __future__ import annotations

import copy
import pickle
from functools import cached_property
from typing import Any, Callable, Literal, SupportsFloat

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias

from metaworld.types import XYZ, EnvironmentStateDict, ObservationDict, Task
from metaworld.utils import reward_utils, rotation

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

    max_path_length: int = 500
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
        action_scale: float = 1.0 / 100,
        action_rot_scale: float = 0.1,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_name: str | None = None,
        reward_function_version: str | None = None,
        width: int = 480,
        height: int = 480,
    ) -> None:
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack((mocap_low, [-1.0, -1.0, -1.0, -1.0]))
        self.mocap_high = np.hstack((mocap_high, [1.0, 1.0, 1.0, 1.0]))
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.num_resets: int = 0
        self.current_seed: int | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        self.width = width
        self.height = height

        self._partially_observable: bool = True

        self.task_name = self.__class__.__name__

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
            np.array([-1, -1, -1, -1, -1, -1, -1]), # Extended action space to 7 dimensions
            np.array([+1, +1, +1, +1, +1, +1, +1]), # Extended action space to 7 dimensions
            dtype=np.float32,
        )
        self._obs_obj_max_len: int = 14
        self._set_task_called: bool = False
        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._target_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._random_reset_space: Box | None = None  # OVERRIDE ME
        self.goal_space: Box | None = None  # OVERRIDE ME
        self._last_stable_obs: npt.NDArray[np.float64] | None = None

        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of _prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)
        # The observation size is 3 (pos) + 4 (quat) + 1 (gripper) + 14 (obj_padded) = 22
        # Stacked observation is 22 * 2 + 3 (goal) = 47
        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = np.zeros(22, dtype=np.float64)

        self.task_name = self.__class__.__name__

        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
            render_mode,
            camera_id,
            camera_name,
            reward_function_version,
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
        pos_delta = action[:3] * self.action_scale # Split action for position
        quat_delta_euler = action[3:6] * self.action_rot_scale # Split action for rotation (Euler angles)

        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low[:3], # Clip position only
            self.mocap_high[:3], # Clip position only
        )
        self.data.mocap_pos = new_mocap_pos

        # Convert current mocap_quat to Euler, add delta, convert back to quat
        current_quat = self.data.mocap_quat[0]
        current_euler = rotation.quat2euler(current_quat)
        new_euler = current_euler + quat_delta_euler
        new_quat = rotation.euler2quat(new_euler)
        
        # Normalize new_quat to ensure it's a unit quaternion
        new_quat = new_quat / np.linalg.norm(new_quat)

        self.data.mocap_quat = new_quat[None] # Apply new quaternion

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
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
        return self.data.geom("objGeom").id

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
        """Combines the end effector's {pos, closed amount} and the object(s)' {pos, quat} into a single flat observation.

        Note: The goal's position is *not* included in this.

        Returns:
            The flat observation array (18 elements)
        """

        pos_hand = self.get_endeff_pos()
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
        return np.hstack((pos_hand, quat_hand, gripper_distance_apart, obs_obj_padded)) # Include quat_hand

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            The flat observation array (47 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_obs_dict(self) -> ObservationDict:
        obs = self._get_obs()
        # The state_achieved_goal should be the end-effector position (3) + quaternion (4) = 7 elements
        # The current observation is pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + obs_obj_padded (14) = 22
        # The previous observation is 22 elements
        # The goal is 3 elements
        # Total observation is 22 + 22 + 3 = 47
        # state_achieved_goal should be the current end-effector pos and quat, which are the first 7 elements of curr_obs
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[:7], # Updated to reflect hand pos and quat
        )

    @cached_property
    def sawyer_observation_space(self) -> Box:
        obs_obj_max_len = 14
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
        
        # Current observation: pos_hand (3) + quat_hand (4) + gripper_distance_apart (1) + obs_obj_padded (14) = 22
        # Previous observation: 22
        # Goal: 3
        # Total: 22 + 22 + 3 = 47
        return Box(
            np.hstack(
                (
                    self._HAND_POS_SPACE.low, # Changed to _HAND_POS_SPACE
                    self._HAND_QUAT_SPACE.low, # Added _HAND_QUAT_SPACE
                    gripper_low,
                    obj_low,
                    self._HAND_POS_SPACE.low, # Changed to _HAND_POS_SPACE
                    self._HAND_QUAT_SPACE.low, # Added _HAND_QUAT_SPACE
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_POS_SPACE.high, # Changed to _HAND_POS_SPACE
                    self._HAND_QUAT_SPACE.high, # Added _HAND_QUAT_SPACE
                    gripper_high,
                    obj_high,
                    self._HAND_POS_SPACE.high, # Changed to _HAND_POS_SPACE
                    self._HAND_QUAT_SPACE.high, # Added _HAND_QUAT_SPACE
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

    @_Decorators.assert_task_is_set
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action: The action to take. Must be a 7 element array of floats.

        Returns:
            The (next_obs, reward, terminated, truncated, info) tuple.
        """
        assert len(action) == 7, f"Actions should be size 7, got {len(action)}" # Changed to 7
        self.set_xyz_action(action[:6]) # Pass position and rotation actions
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
                {  # info
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

    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        """Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            Tuple of reward between 0 and 10 and a dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
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
        assert obs is not None, "Observation should not be None after reset" # Added assertion
        # Update _prev_obs and obs based on new observation size (22 for curr_obs)
        self._prev_obs = obs[:22].copy()
        obs[22:44] = self._prev_obs
        obs = obs.astype(np.float64)
        return obs, info

    def _reset_hand(self, steps: int = 50) -> None:
        """Resets the hand position.

        Args:
            steps: The number of steps to take to reset the hand.
        """
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0]) # Set to identity quaternion
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

### metaworld/types.py

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

### metaworld/wrappers.py

*大小: 11.6 KB | Token: 3.3K*

```python
from __future__ import annotations

import base64

import gymnasium as gym
import numpy as np
from gymnasium import Env
from numpy.typing import NDArray

from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.types import Task


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
