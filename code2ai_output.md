# 项目导出

**文件数量**: 27  
**总大小**: 211.6 KB  
**Token 数量**: 57.2K  
**生成时间**: 2025/7/24 16:44:44

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
    📄 2-test_mujoco.py
    📄 3-control.py
    📄 controllers.py
    📄 coordinate_systems.py
    📄 demo_position_control.py
    📄 force_extractor.py
    📄 params.py
    📄 policy.py
    📄 standalone_peg_insert.py
    📄 success_checker.py
    📄 wrapper.py
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

*大小: 1.2 KB | Token: 332*

```python
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


class Action:
    """Represents an action to be taken in an environment.

    Once initialized, fields can be assigned as if the action
    is a dictionary. Once filled, the corresponding array is
    available as an instance variable.
    """

    def __init__(self, structure: dict[str, npt.NDArray[Any] | int]) -> None:
        """Action.

        Args:
            structure: Map from field names to output array indices
        """
        self._structure = structure
        self.array = np.zeros(len(self), dtype=np.float32)

    def __len__(self) -> int:
        return sum(
            [1 if isinstance(idx, int) else len(idx) for idx in self._structure.items()]
        )

    def __getitem__(self, key) -> npt.NDArray[np.float32]:
        assert key in self._structure, (
            "This action's structure does not contain %s" % key
        )
        return self.array[self._structure[key]]

    def __setitem__(self, key: str, value) -> None:
        assert key in self._structure, f"This action's structure does not contain {key}"
        self.array[self._structure[key]] = value
```

### metaworld/policies/policy.py

*大小: 2.3 KB | Token: 645*

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
    if np.any(np.absolute(response) > 1.0):
        warnings.warn(
            "Constant(s) may be too high. Environments clip response to [-1, 1]"
        )

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

*大小: 2.1 KB | Token: 602*

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
            "gripper_distance_apart": obs[3],
            "peg_pos": obs[4:7],
            "peg_rot": obs[7:11],
            "goal_pos": obs[-3:],
            "unused_info_curr_obs": obs[11:18],
            "_prev_obs": obs[18:36],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = self._grab_effort(o_d)

        return action.array

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

*大小: 17.4 KB | Token: 4.9K*

```python
# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utilities for computing rotations in 3D space.

Many methods borrow heavily or entirely from transforms3d: https://github.com/matthew-brett/transforms3d
They have mostly been modified to support batched operations.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

"""
Rotations
=========
Note: these have caused many subtle bugs in the past.
Be careful while updating these methods and while using them in clever ways.
See MuJoCo documentation here: http://mujoco.org/book/modeling.html#COrientation
Conventions
-----------
    - All functions accept batches as well as individual rotations
    - All rotation conventions match respective MuJoCo defaults
    - All angles are in radians
    - Matricies follow LR convention
    - Euler Angles are all relative with 'xyz' axes ordering
    - See specific representation for more information
Representations
---------------
Euler
    There are many euler angle frames -- here we will strive to use the default
        in MuJoCo, which is eulerseq='xyz'.
    This frame is a relative rotating frame, about x, y, and z axes in order.
        Relative rotating means that after we rotate about x, then we use the
        new (rotated) y, and the same for z.
Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:
            q0 = cos(angle / 2)
            q1 = sin(angle / 2) * x
            q2 = sin(angle / 2) * y
            q3 = sin(angle / 2) * z
        This is also sometimes called qw, qx, qy, qz.
    Note that quaternions are ambiguous, because we can represent a rotation by
        angle about vector <x, y, z> and -angle about vector <-x, -y, -z>.
        To choose between these, we pick "first nonzero positive", where we
        make the first nonzero element of the quaternion positive.
    This can result in mismatches if you're converting an quaternion that is not
        "first nonzero positive" to a different representation and back.
Axis Angle
    (Not currently implemented)
    These are very straightforward.  Rotation is angle about a unit vector.
XY Axes
    (Not currently implemented)
    We are given x axis and y axis, and z axis is cross product of x and y.
Z Axis
    This is NOT RECOMMENDED.  Defines a unit vector for the Z axis,
        but rotation about this axis is not well defined.
    Instead pick a fixed reference direction for another axis (e.g. X)
        and calculate the other (e.g. Y = Z cross-product X),
        then use XY Axes rotation instead.
SO3
    (Not currently implemented)
    While not supported by MuJoCo, this representation has a lot of nice features.
    We expect to add support for these in the future.
TODO / Missing
--------------
    - Rotation integration or derivatives (e.g. velocity conversions)
    - More representations (SO3, etc)
    - Random sampling (e.g. sample uniform random rotation)
    - Performance benchmarks/measurements
    - (Maybe) define everything as to/from matricies, for simplicity
"""

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def euler2mat(euler: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts euler angles to rotation matrices.

    Args:
        euler: the euler angles. Can be batched and stored in any (nested) iterable.

    Returns:
        Rotation matrices corresponding to the euler angles, in double precision.
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shaped euler {euler}"

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts euler angles to quaternions.

    Args:
        euler: the euler angles. Can be batched and stored in any (nested) iterable.

    Returns:
        Quaternions corresponding to the euler angles, in double precision.
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts rotation matrices to euler angles.

    Args:
        mat: a 3D rotation matrix. Can be batched and stored in any (nested) iterable.

    Returns:
        Euler angles corresponding to the rotation matrices, in double precision.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


def mat2quat(mat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts rotation matrices to quaternions.

    Args:
        mat: a 3D rotation matrix. Can be batched and stored in any (nested) iterable.

    Returns:
        Quaternions corresponding to the rotation matrices, in double precision.
    """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), f"Invalid shape matrix {mat}"

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=["multi_index"])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts quaternions to euler angles.

    Args:
        quat: the quaternion. Can be batched and stored in any (nested) iterable.

    Returns:
        Euler angles corresponding to the quaternions, in double precision.
    """
    return mat2euler(quat2mat(quat))


def subtract_euler(
    e1: npt.NDArray[Any], e2: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Subtracts two euler angles.

    Args:
        e1: the first euler angles. Can be batched.
        e2: the second euler angles. Can be batched.

    Returns:
        Euler angles corresponding to the difference between e1 and e2, in double precision.
    """
    assert e1.shape == e2.shape
    assert e1.shape[-1] == 3
    q1 = euler2quat(e1)
    q2 = euler2quat(e2)
    q_diff = quat_mul(q1, quat_conjugate(q2))
    return quat2euler(q_diff)


def quat2mat(quat: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Converts quaternions to rotation matrices.

    Args:
        quat: the quaternion. Can be batched and stored in any (nested) iterable.

    Returns:
        Rotation matrices corresponding to the quaternions, in double precision.
    """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, f"Invalid shape quat {quat}"

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Returns the conjugate of a quaternion.

    Args:
        q: the quaternion. Can be batched.

    Returns:
        The conjugate of the quaternion.
    """
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q


def quat_mul(q0: npt.NDArray[Any], q1: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Multiplies two quaternions.

    Args:
        q0: the first quaternion. Can be batched.
        q1: the second quaternion. Can be batched.

    Returns:
        The product of `q0` and `q1`.
    """
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def quat_rot_vec(q: npt.NDArray[Any], v0: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Rotates a vector by a quaternion.

    Args:
        q: the quaternion.
        v0: the vector.

    Returns:
        The rotated vector.
    """
    q_v0 = np.array([0, v0[0], v0[1], v0[2]])
    q_v = quat_mul(q, quat_mul(q_v0, quat_conjugate(q)))
    v = q_v[1:]
    return v


def quat_identity() -> npt.NDArray[np.int_]:
    """Returns the identity quaternion."""
    return np.array([1, 0, 0, 0])


def quat2axisangle(quat: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], float]:
    """Converts a quaternion to an axis-angle representation.

    Args:
        quat: the quaternion.

    Returns:
        The axis-angle representation of `quat` as an `(axis, angle)` tuple.
    """
    theta = 0.0
    axis = np.array([0, 0, 1])
    sin_theta = np.linalg.norm(quat[1:])

    if sin_theta > 0.0001:
        theta = 2 * np.arcsin(sin_theta)
        theta *= 1 if quat[0] >= 0 else -1
        axis = quat[1:] / sin_theta

    return axis, theta


def euler2point_euler(euler: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert euler angles to 2D points on the unit circle for each one.

    Args:
        euler: the euler angles. Can optionally have 1 batch dimension.

    Returns:
        2D points on the unit circle for each axis, returned as [`sin_x`, `sin_y`, `sin_z`, `cos_x`, `cos_y`, `cos_z`].
    """
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 3
    _euler_sin = np.sin(_euler)
    _euler_cos = np.cos(_euler)
    return np.concatenate([_euler_sin, _euler_cos], axis=-1)


def point_euler2euler(euler: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert 2D points on the unit circle for each axis to euler angles.

    Args:
        euler: 2D points on the unit circle for each axis, stored as [`sin_x`, `sin_y`, `sin_z`, `cos_x`, `cos_y`, `cos_z`].
            Can optionally have 1 batch dimension.

    Returns:
        The corresponding euler angles expressed as scalars.
    """
    _euler = euler.copy()
    if len(_euler.shape) < 2:
        _euler = np.expand_dims(_euler, 0)
    assert _euler.shape[1] == 6
    angle = np.arctan(_euler[..., :3] / _euler[..., 3:])
    angle[_euler[..., 3:] < 0] += np.pi
    return angle


def quat2point_quat(quat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert the quaternion's angle to 2D points on the unit circle for each axis in 3D space.

    Args:
        quat: the quaternion. Can optionally have 1 batch dimension.

    Returns:
        A quaternion with its angle expressed as 2D points on the unit circle for each axis in 3D space, returned as
            [`sin_x`, `sin_y`, `sin_z`, `cos_x`, `cos_y`, `cos_z`, `quat_axis_x`, `quat_axis_y`, `quat_axis_z`].
    """
    # Should be in qw, qx, qy, qz
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 4
    angle = np.arccos(_quat[:, [0]]) * 2
    xyz = _quat[:, 1:]
    xyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (xyz / np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([np.sin(angle), np.cos(angle), xyz], axis=-1)


def point_quat2quat(quat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Convert 2D points on the unit circle for each axis to quaternions.

    Args:
        quat: A quaternion with its angle expressed as 2D points on the unit circle for each axis in 3D space, stored as
            [`sin_x`, `sin_y`, `sin_z`, `cos_x`, `cos_y`, `cos_z`, `quat_axis_x`, `quat_axis_y`, `quat_axis_z`].
            Can optionally have 1 batch dimension.

    Returns:
        The quaternion with its angle expressed as a scalar.
    """
    _quat = quat.copy()
    if len(_quat.shape) < 2:
        _quat = np.expand_dims(_quat, 0)
    assert _quat.shape[1] == 5
    angle = np.arctan(_quat[:, [0]] / _quat[:, [1]])
    qw = np.cos(angle / 2)

    qxyz = _quat[:, 2:]
    qxyz[np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5] = (qxyz * np.sin(angle / 2))[
        np.squeeze(np.abs(np.sin(angle / 2))) >= 1e-5
    ]
    return np.concatenate([qw, qxyz], axis=-1)


def normalize_angles(angles: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Puts angles in [-pi, pi] range."""
    angles = angles.copy()
    if angles.size > 0:
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        assert -np.pi - 1e-6 <= angles.min() and angles.max() <= np.pi + 1e-6
    return angles


def round_to_straight_angles(angles: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Returns closest angle modulo 90 degrees."""
    angles = np.round(angles / (np.pi / 2)) * (np.pi / 2)
    return normalize_angles(angles)


def get_parallel_rotations() -> list[npt.NDArray[Any]]:
    mult90 = [0, np.pi / 2, -np.pi / 2, np.pi]
    parallel_rotations: list[npt.NDArray] = []
    for euler in itertools.product(mult90, repeat=3):
        canonical = mat2euler(euler2mat(euler))
        canonical = np.round(canonical / (np.pi / 2))
        if canonical[0] == -2:
            canonical[0] = 2
        if canonical[2] == -2:
            canonical[2] = 2
        canonical *= np.pi / 2
        if all([(canonical != rot).any() for rot in parallel_rotations]):
            parallel_rotations.append(canonical)
    assert len(parallel_rotations) == 24
    return parallel_rotations
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
        return _init_each_env( [misc]
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
                split, [arg-type]
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

*大小: 8.3 KB | Token: 2.3K*

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
    """
    Motivation for V3:
        V1 was difficult to solve because the observation didn't say where
        to insert the peg (the hole's location). Furthermore, the hole object
        could be initialized in such a way that it severely restrained the
        sawyer's movement.
    Changelog from V1 to V3:
        - (8/21/20) Updated to Byron's XML
        - (7/7/20) Removed 1 element vector. Replaced with 3 element position
            of the hole (for consistency with other environments)
        - (6/16/20) Added a 1 element vector to the observation. This vector
            points from the end effector to the hole in the Y direction.
            i.e. (self._target_pos - pos_hand)[1]
        - (6/16/20) Used existing goal_low and goal_high values to constrain
            the hole's position, as opposed to hand_low and hand_high
    """

    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
        height: int = 480,
        width: int = 480,
        reward_function_version: str | None = None,  # 添加这行

    ) -> None:
        hand_init_pos = (0, 0.6, 0.2)

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
            reward_function_version=reward_function_version,  # 添加这行

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

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[4:7]

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

        return self._get_obs()

    def compute_reward(
        self, action: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        assert self._target_pos is not None and self.obj_init_pos is not None
        tcp = self.tcp_center
        obj = obs[4:7]
        obj_head = self._get_site_pos("pegHead")
        tcp_opened: float = obs[3]
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
```

### metaworld/sawyer_xyz_env.py

*大小: 32.0 KB | Token: 9.0K*

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
from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias

from metaworld.types import XYZ, EnvironmentStateDict, ObservationDict, Task
from metaworld.utils import reward_utils

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"


class SawyerMocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Sawyer Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])
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
        mocap_pos, mocap_quat = state
        self.set_state(mocap_pos, mocap_quat)

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

    _HAND_SPACE = Box(
        np.array([-0.525, 0.348, -0.0525]),
        np.array([+0.525, 1.025, 0.7]),
        dtype=np.float64,
    )
    """Bounds for hand position."""

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
        action_rot_scale: float = 1.0,
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
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.num_resets: int = 0
        self.current_seed: int | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        self.width = width
        self.height = height

        # TODO Probably needs to be removed
        self.discrete_goal_space: Box | None = None
        self.discrete_goals: list = []
        self.active_discrete_goal: int | None = None

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
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
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

        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = self._get_curr_obs_combined_no_goal()

        self.task_name = self.__class__.__name__

        EzPickle.__init__(
            self,
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
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos = new_mocap_pos
        self.data.mocap_quat = np.array([1, 0, 1, 0])

    def discretize_goal_space(self, goals: list) -> None:
        """Discretizes the goal space into a Discrete space.
        Current disabled and callign it will stop execution.

        Args:
            goals: List of goals to discretize
        """
        assert False, "Discretization is not supported at the moment."
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

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
        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            The flat observation array (39 elements)
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
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[3:-3],
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
        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    self._HAND_SPACE.high,
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
            action: The action to take. Must be a 4 element array of floats.

        Returns:
            The (next_obs, reward, terminated, truncated, info) tuple.
        """
        assert len(action) == 4, f"Actions should be size 4, got {len(action)}"
        self.set_xyz_action(action[:3])
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
                False,
                False,  # termination flag always False
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
        self._prev_obs = obs[:18].copy()
        obs[18:36] = self._prev_obs
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
            self.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
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

### src_test/0_init.py

*大小: 1.5 KB | Token: 381*

```python
import gymnasium as gym
import metaworld
import time
import random
import mujoco

env_name = 'peg-insert-side-v3'
# 1
# env = metaworld.make_mt_envs(
#     'peg-insert-side-v3',
#     render_mode='human',
#     width=1080,
#     height=1920
# )

# env_class = metaworld.env_dict.ALL_V3_ENVIRONMENTS[env_name]
# env = env_class(render_mode='human', width=1080, height=1920)
# benchmark = metaworld.MT1(env_name)
# task = benchmark.train_tasks[0]  # 使用第一个训练任务
# env.set_task(task)

env = gym.make('Meta-World/MT1', env_name=env_name, render_mode='human', width=1080, height=1920)

# 5. 实例化专家策略
from metaworld.policies import SawyerPegInsertionSideV3Policy
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
    # info['success'] 在成功时会返回 1.0
    if info['success'] > 0.5:
        print("任务成功！")
        # done = True
        
    time.sleep(0.02)
    count += 1

print(f"最终信息: {info}")
env.close()
```

### src_test/1-fix_cam.py

*大小: 7.1 KB | Token: 1.8K*

```python
import gymnasium as gym
import metaworld
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
    base_env = metaworld.make_mt_envs(
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
        
        from metaworld.policies import SawyerPegInsertionSideV3Policy
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

### src_test/2-test_mujoco.py

*大小: 14.1 KB | Token: 3.6K*

```python
import mujoco
from mujoco import viewer
import cv2
import numpy as np
import time
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation

class MetaWorldMuJoCoAdapter:
    """
    将MuJoCo环境适配为MetaWorld策略可以使用的形式
    """
    
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 创建渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 初始化环境状态
        self._initialize_env_state()
        
        # 观测相关
        self._obs_obj_max_len = 14
        self._prev_obs = None
        
        # 任务相关
        self._target_pos = np.array([-0.3, 0.6, 0.0])  # 目标位置
        self.obj_init_pos = np.array([0, 0.6, 0.02])   # 物体初始位置
        self.hand_init_pos = np.array([0, 0.6, 0.2])   # 手的初始位置
        
        # 环境参数
        self.max_path_length = 500
        self.curr_path_length = 0
        
    def _initialize_env_state(self):
        """初始化环境状态"""
        # 设置初始关节位置
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置机器人初始姿态
        if self.model.nq > 0:
            # Sawyer机器人的初始关节角度
            init_qpos = np.array([0.0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
            if len(init_qpos) <= self.model.nq:
                self.data.qpos[:len(init_qpos)] = init_qpos
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.curr_path_length = 0
        
        # 重置mujoco数据
        mujoco.mj_resetData(self.model, self.data)
        self._initialize_env_state()
        
        # 设置物体位置（可以添加随机化）
        self._set_object_position(self.obj_init_pos)
        
        # 设置目标位置
        self._set_target_position(self._target_pos)
        
        # 计算初始观测
        obs = self._get_obs()
        self._prev_obs = obs[:18].copy()
        
        return obs
    
    def _set_object_position(self, pos: np.ndarray):
        """设置物体位置"""
        # 查找物体的关节ID
        try:
            # 尝试找到物体相关的关节
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and 'obj' in joint_name.lower():
                    # 假设物体是自由关节
                    if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_addr = self.model.jnt_qposadr[i]
                        self.data.qpos[qpos_addr:qpos_addr+3] = pos
                        break
            else:
                # 如果没有找到关节，尝试直接设置body位置
                for i in range(self.model.nbody):
                    body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and 'peg' in body_name.lower():
                        self.data.xpos[i] = pos
                        break
        except Exception as e:
            print(f"设置物体位置时出错: {e}")
    
    def _set_target_position(self, pos: np.ndarray):
        """设置目标位置"""
        try:
            # 查找目标site
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name and 'goal' in site_name.lower():
                    self.data.site_xpos[i] = pos
                    break
        except Exception as e:
            print(f"设置目标位置时出错: {e}")
    
    def _get_endeff_pos(self) -> np.ndarray:
        """获取末端执行器位置"""
        try:
            # 查找手部body
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'hand' in body_name.lower():
                    return self.data.xpos[i].copy()
            
            # 如果没找到，返回默认位置
            return np.array([0.0, 0.6, 0.2])
        except:
            return np.array([0.0, 0.6, 0.2])
    
    def _get_gripper_distance(self) -> float:
        """获取夹爪张开程度"""
        try:
            # 查找左右夹爪
            left_pos = None
            right_pos = None
            
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    if 'left' in body_name.lower() and 'claw' in body_name.lower():
                        left_pos = self.data.xpos[i]
                    elif 'right' in body_name.lower() and 'claw' in body_name.lower():
                        right_pos = self.data.xpos[i]
            
            if left_pos is not None and right_pos is not None:
                distance = np.linalg.norm(right_pos - left_pos)
                return np.clip(distance / 0.1, 0.0, 1.0)
            else:
                return 0.5  # 默认值
        except:
            return 0.5
    
    def _get_object_pos(self) -> np.ndarray:
        """获取物体位置"""
        try:
            # 查找物体site
            for i in range(self.model.nsite):
                site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                if site_name and 'peg' in site_name.lower() and 'grasp' in site_name.lower():
                    return self.data.site_xpos[i].copy()
            
            # 如果没找到site，查找body
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'peg' in body_name.lower():
                    return self.data.xpos[i].copy()
            
            return self.obj_init_pos.copy()
        except:
            return self.obj_init_pos.copy()
    
    def _get_object_quat(self) -> np.ndarray:
        """获取物体四元数"""
        try:
            # 查找物体body的旋转
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and 'peg' in body_name.lower():
                    # 从旋转矩阵获取四元数
                    xmat = self.data.xmat[i].reshape(3, 3)
                    return Rotation.from_matrix(xmat).as_quat()
            
            return np.array([1, 0, 0, 0])  # 默认四元数
        except:
            return np.array([1, 0, 0, 0])
    
    def _get_obs(self) -> np.ndarray:
        """获取观测值，格式与MetaWorld兼容"""
        # 获取末端执行器位置
        pos_hand = self._get_endeff_pos()
        
        # 获取夹爪状态
        gripper_distance = self._get_gripper_distance()
        
        # 获取物体位置和方向
        obj_pos = self._get_object_pos()
        obj_quat = self._get_object_quat()
        
        # 构建物体观测（填充到固定长度）
        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_info = np.hstack([obj_pos, obj_quat])
        obs_obj_padded[:len(obj_info)] = obj_info
        
        # 当前观测
        curr_obs = np.hstack([pos_hand, gripper_distance, obs_obj_padded])
        
        # 帧堆叠
        if self._prev_obs is None:
            self._prev_obs = curr_obs.copy()
        
        # 目标位置（MetaWorld策略需要）
        goal_pos = self._target_pos.copy()
        
        # 完整观测：当前 + 前一帧 + 目标
        obs = np.hstack([curr_obs, self._prev_obs, goal_pos])
        
        # 更新前一帧观测
        self._prev_obs = curr_obs.copy()
        
        return obs.astype(np.float64)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步仿真"""
        # 应用动作到控制器
        self._apply_action(action)
        
        # 执行仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新路径长度
        self.curr_path_length += 1
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励和完成状态
        reward, info = self._compute_reward_and_info(obs, action)
        
        # 检查终止条件
        terminated = info.get('success', False)
        truncated = self.curr_path_length >= self.max_path_length
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: np.ndarray):
        """将MetaWorld动作应用到MuJoCo控制器"""
        if len(action) >= 4:
            # 前3个是位置增量，第4个是夹爪控制
            pos_delta = action[:3] * 0.01  # 缩放因子
            gripper_action = action[3]
            
            # 应用位置控制（通过mocap或者直接控制关节）
            # 这里需要根据你的具体模型来调整
            try:
                # 如果有mocap body
                if self.model.nmocap > 0:
                    current_pos = self.data.mocap_pos[0].copy()
                    new_pos = current_pos + pos_delta
                    # 限制在工作空间内
                    new_pos = np.clip(new_pos, 
                                    [-0.5, 0.4, 0.05], 
                                    [0.5, 1.0, 0.5])
                    self.data.mocap_pos[0] = new_pos
                    self.data.mocap_quat[0] = [1, 0, 1, 0]
                
                # 应用夹爪控制
                if self.model.nu > 0:
                    # 找到夹爪控制器
                    self.data.ctrl[-2:] = [gripper_action, -gripper_action]
                    
            except Exception as e:
                print(f"应用动作时出错: {e}")
    
    def _compute_reward_and_info(self, obs: np.ndarray, action: np.ndarray) -> Tuple[float, dict]:
        """计算奖励和信息"""
        # 简化的奖励函数
        obj_pos = self._get_object_pos()
        target_pos = self._target_pos
        
        # 距离奖励
        obj_to_target = np.linalg.norm(obj_pos - target_pos)
        success = obj_to_target < 0.07
        
        # 基本奖励
        reward = -obj_to_target
        if success:
            reward += 10.0
        
        info = {
            'success': float(success),
            'obj_to_target': obj_to_target,
            'reward': reward
        }
        
        return reward, info


def create_custom_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """创建自定义相机"""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    
    # 设置为自由相机
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.5
    camera.azimuth = 135
    camera.elevation = -25
    camera.lookat = np.array([0.0, 0.6, 0.15])
    
    return camera


def main(xml_path):
    """主程序"""
    # 创建适配器
    env_adapter = MetaWorldMuJoCoAdapter(xml_path)
    
    # 创建策略
    from metaworld.policies import SawyerPegInsertionSideV3Policy
    policy = SawyerPegInsertionSideV3Policy()
    
    # 创建自定义相机
    camera = create_custom_camera(env_adapter.model)
    
    # 重置环境
    obs = env_adapter.reset()
    
    print("开始仿真...")
    print(f"观测维度: {len(obs)}")
    
    # 使用被动viewer进行可视化
    with viewer.launch_passive(env_adapter.model, env_adapter.data) as v:
        # 设置相机
        v.cam.distance = 1.5
        v.cam.azimuth = 135
        v.cam.elevation = -25
        v.cam.lookat = [0.0, 0.6, 0.15]
        
        step_count = 0
        while step_count < 1000:
            # 获取策略动作
            try:
                action = policy.get_action(obs)
                
                # 执行步骤
                obs, reward, terminated, truncated, info = env_adapter.step(action)
                
                # 同步viewer
                v.sync()
                
                # 打印信息
                if step_count % 50 == 0:
                    print(f"Step {step_count}: Reward={reward:.3f}, "
                          f"Success={info['success']}, "
                          f"Distance={info['obj_to_target']:.3f}")
                
                # 检查完成
                if terminated or truncated:
                    print(f"Episode finished at step {step_count}")
                    if info['success']:
                        print("任务成功完成！")
                    # 重置环境
                    obs = env_adapter.reset()
                    step_count = 0
                    continue
                
                step_count += 1
                time.sleep(0.02)  # 控制仿真速度
                
            except KeyboardInterrupt:
                print("用户中断")
                break
            except Exception as e:
                print(f"仿真出错: {e}")
                break


def test_environment(xml_path):
    
    try:
        env_adapter = MetaWorldMuJoCoAdapter(xml_path)
        print("✓ 环境创建成功")
        
        obs = env_adapter.reset()
        print(f"✓ 环境重置成功，观测维度: {len(obs)}")
        
        # 测试动作
        action = np.array([0.01, 0.0, 0.0, 0.5])
        obs, reward, terminated, truncated, info = env_adapter.step(action)
        print(f"✓ 步骤执行成功，奖励: {reward}")
        
        # 测试策略
        from metaworld.policies import SawyerPegInsertionSideV3Policy
        policy = SawyerPegInsertionSideV3Policy()
        action = policy.get_action(obs)
        print(f"✓ 策略调用成功，动作: {action}")
        
        print("所有测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    xml_path = "./metaworld/xml/sawyer_peg_insertion_side.xml"
    # 首先测试环境
    if test_environment(xml_path):
        # 如果测试通过，运行主程序
        main(xml_path)
    else:
        print("请检查XML路径和依赖项")
```

### src_test/3-control.py

*大小: 37.7 KB | Token: 9.7K*

```python
import numpy as np
import mujoco
import gymnasium as gym
import metaworld
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time

@dataclass
class ControlParams:
    """力位混合控制参数"""
    # 位置控制参数
    kp_pos: float = 1000.0  # 位置刚度
    kd_pos: float = 100.0   # 位置阻尼
    
    # 姿态控制参数
    kp_rot: float = 500.0   # 姿态刚度
    kd_rot: float = 50.0    # 姿态阻尼
    
    # 力控制参数
    kp_force: float = 0.001  # 力增益
    ki_force: float = 0.0001 # 力积分增益
    kp_torque: float = 0.0005 # 力矩增益
    ki_torque: float = 0.00005 # 力矩积分增益
    force_deadzone: float = 0.5  # 力死区(N)
    torque_deadzone: float = 0.1 # 力矩死区(Nm)
    max_force: float = 10.0     # 最大允许力(N)
    max_torque: float = 2.0     # 最大允许力矩(Nm)
    
    # 几何参数（基于XML分析）
    peg_radius: float = 0.015    # peg半径
    hole_radius: float = 0.025   # hole半径（估计）
    insertion_tolerance: float = 0.008  # 径向容差
    min_insertion_depth: float = 0.06   # 最小插入深度
    
    # 插入策略参数
    approach_distance: float = 0.08  # 接近距离
    alignment_distance: float = 0.03 # 对齐距离
    max_orientation_error: float = 0.2 # 最大姿态误差(弧度)
    
    # 控制频率
    position_control_freq: float = 500.0  # Hz
    force_control_freq: float = 1000.0    # Hz
    
    # 切换阈值
    switch_distance: float = 0.05  # 距离目标多远时切换到力控制(m)

class TaskCoordinateSystem:
    """任务坐标系：以hole为基准建立坐标系"""
    
    def __init__(self, hole_pos: np.ndarray, hole_orientation: np.ndarray):
        self.hole_pos = hole_pos.copy()
        self.hole_orientation = hole_orientation.copy()
        
        # 构建任务坐标系：Y轴为插入方向（指向hole内部，即Y负方向）
        self.y_axis = -self.hole_orientation / np.linalg.norm(self.hole_orientation)
        
        # 构建正交的X、Z轴（XZ平面为径向平面）
        if abs(self.y_axis[2]) < 0.9:
            self.z_axis = np.cross(self.y_axis, [0, 0, 1])
        else:
            self.z_axis = np.cross(self.y_axis, [1, 0, 0])
        if np.linalg.norm(self.z_axis) < 1e-6:
            self.z_axis = np.array([0, 0, 1])
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.x_axis = np.cross(self.y_axis, self.z_axis)
        
        # 旋转矩阵：世界坐标系 -> 任务坐标系
        self.rotation_matrix = np.column_stack([self.x_axis, self.y_axis, self.z_axis])
        
        # 任务坐标系的目标姿态（peg应该对齐的方向）
        self.target_rotation = Rotation.from_matrix(self.rotation_matrix.T)
        
        print(f"任务坐标系建立：")
        print(f"  X轴（径向1）: {self.x_axis}")
        print(f"  Y轴（插入方向）: {self.y_axis}")
        print(f"  Z轴（径向2）: {self.z_axis}")
        
    def world_to_task(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标转任务坐标"""
        relative_pos = world_pos - self.hole_pos
        return self.rotation_matrix.T @ relative_pos
    
    def task_to_world(self, task_pos: np.ndarray) -> np.ndarray:
        """任务坐标转世界坐标"""
        world_relative = self.rotation_matrix @ task_pos
        return world_relative + self.hole_pos
    
    def world_force_to_task(self, world_force: np.ndarray) -> np.ndarray:
        """世界坐标系下的力转换到任务坐标系"""
        return self.rotation_matrix.T @ world_force
    
    def task_force_to_world(self, task_force: np.ndarray) -> np.ndarray:
        """任务坐标系下的力转换到世界坐标系"""
        return self.rotation_matrix @ task_force
    
    def get_orientation_error(self, current_rotation: Rotation) -> np.ndarray:
        """计算当前姿态与目标姿态的误差"""
        # 计算相对旋转
        relative_rotation = self.target_rotation * current_rotation.inv()
        
        # 转换为轴角表示
        axis_angle = relative_rotation.as_rotvec()
        
        # 转换到任务坐标系
        return self.rotation_matrix.T @ axis_angle

class InsertionSuccessChecker:
    """插入成功检测器"""
    
    def __init__(self, params: ControlParams, task_coord_system: TaskCoordinateSystem):
        self.params = params
        self.task_coord_system = task_coord_system
        
    def check_insertion_success(self, peg_head_pos: np.ndarray, peg_pos: np.ndarray, 
                              peg_rotation: Rotation) -> Dict:
        """检查插入成功状态"""
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        peg_center_task = self.task_coord_system.world_to_task(peg_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 计算径向距离（在XZ平面上的距离）
        radial_distance = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        
        # 计算插入深度（Y方向，负值表示插入）
        insertion_depth = max(0, -peg_head_task[1])
        
        # 检查是否在hole内（径向约束）
        is_inside_hole = radial_distance <= self.params.insertion_tolerance
        
        # 检查是否达到最小插入深度
        sufficient_depth = insertion_depth >= self.params.min_insertion_depth
        
        # 检查peg是否大致对齐
        center_radial_distance = np.sqrt(peg_center_task[0]**2 + peg_center_task[2]**2)
        is_aligned = center_radial_distance <= self.params.insertion_tolerance * 2
        
        # 检查姿态是否对齐
        orientation_magnitude = np.linalg.norm(orientation_error)
        is_orientation_aligned = orientation_magnitude <= self.params.max_orientation_error
        
        success = is_inside_hole and sufficient_depth and is_aligned and is_orientation_aligned
        
        return {
            'success': success,
            'insertion_depth': insertion_depth,
            'radial_distance': radial_distance,
            'orientation_error': orientation_magnitude,
            'is_inside_hole': is_inside_hole,
            'sufficient_depth': sufficient_depth,
            'is_aligned': is_aligned,
            'is_orientation_aligned': is_orientation_aligned,
            'peg_head_task_pos': peg_head_task,
            'peg_center_task_pos': peg_center_task,
            'orientation_error_vec': orientation_error
        }

class Enhanced6DOFController:
    """增强的6DOF力位混合控制器"""
    
    def __init__(self, params: ControlParams):
        self.params = params
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"
        
    def reset(self):
        """重置控制器状态"""
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        self.last_time = 0.0
        self.control_mode = "position"
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_orientation: np.ndarray,  # 姿态误差向量
                       target_orientation: np.ndarray,   # 目标姿态误差（通常为0）
                       current_force: np.ndarray,
                       target_force: np.ndarray,
                       current_torque: np.ndarray,
                       target_torque: np.ndarray,
                       selection_matrix_pos: np.ndarray,  # 位置选择矩阵
                       selection_matrix_rot: np.ndarray,  # 姿态选择矩阵
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算6DOF力位混合控制输出
        
        Returns:
            Tuple[位置控制输出, 姿态控制输出]
        """
        
        # 位置误差
        pos_error = target_pos - current_pos
        
        # 姿态误差
        rot_error = target_orientation - current_orientation
        
        # 力误差
        force_error = target_force - current_force
        torque_error = target_torque - current_torque
        
        # 力积分（仅在力控制方向）
        force_mask = 1 - np.diag(selection_matrix_pos)
        torque_mask = 1 - np.diag(selection_matrix_rot)
        
        self.force_integral += force_error * force_mask * dt
        self.torque_integral += torque_error * torque_mask * dt
        
        # 限制积分器防止积分饱和
        self.force_integral = np.clip(self.force_integral, -1.0, 1.0)
        self.torque_integral = np.clip(self.torque_integral, -0.5, 0.5)
        
        # 位置控制输出
        pos_control = self.params.kp_pos * pos_error
        rot_control = self.params.kp_rot * rot_error
        
        # 力控制输出（转换为位置/姿态增量）
        force_control = (self.params.kp_force * force_error + 
                        self.params.ki_force * self.force_integral)
        torque_control = (self.params.kp_torque * torque_error + 
                         self.params.ki_torque * self.torque_integral)
        
        # 应用死区
        force_control = np.where(np.abs(current_force) > self.params.force_deadzone,
                                force_control, 0)
        torque_control = np.where(np.abs(current_torque) > self.params.torque_deadzone,
                                 torque_control, 0)
        
        # 混合控制输出
        pos_output = (selection_matrix_pos @ pos_control + 
                     (np.eye(3) - selection_matrix_pos) @ force_control)
        rot_output = (selection_matrix_rot @ rot_control + 
                     (np.eye(3) - selection_matrix_rot) @ torque_control)
        
        return pos_output, rot_output

class ForceExtractor:
    """改进的力信息提取器"""
    
    def __init__(self, env):
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        # 查找peg相关的几何体ID
        self.peg_geom_ids = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'peg' in geom_name.lower():
                self.peg_geom_ids.append(i)
        
        print(f"找到peg几何体ID: {self.peg_geom_ids}")
    
    def get_contact_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取peg上的接触力和力矩"""
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        contact_count = 0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 检查是否涉及peg
            if contact.geom1 in self.peg_geom_ids or contact.geom2 in self.peg_geom_ids:
                # 获取接触力
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                
                # 接触力的前3个分量是法向力和切向力
                contact_force = c_array[:3]
                
                # 转换到世界坐标系
                contact_frame = contact.frame.reshape(3, 3)
                world_force = contact_frame @ contact_force
                
                # 计算力矩（简化计算，假设力作用在接触点）
                contact_pos = contact.pos
                peg_pos = self.env.unwrapped._get_pos_objects()
                r = contact_pos - peg_pos
                contact_torque = np.cross(r, world_force)
                
                total_force += world_force
                total_torque += contact_torque
                contact_count += 1
        
        return total_force, total_torque

class HybridControlWrapper(gym.Wrapper):
    """改进的力位混合控制包装器"""
    
    def __init__(self, env, control_params: ControlParams = None):
        super().__init__(env)
        self.params = control_params or ControlParams()
        self.force_extractor = ForceExtractor(env)
        self.controller = Enhanced6DOFController(self.params)
        
        # 任务坐标系和成功检测器（将在reset时初始化）
        self.task_coord_system = None
        self.success_checker = None
        
        # 控制状态
        self.peg_grasped = False
        self.control_phase = "approach"  # "approach", "align", "insert"
        self.insertion_stage = "move_to_front"  # "move_to_front", "align_orientation", "insert"
        
        # 记录数据
        self.episode_data = {
            'forces': [],
            'torques': [],
            'positions': [],
            'orientations': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'insertion_status': [],
            'box_positions': []
        }
        
        # 初始状态
        self.initial_box_pos = None
        
    def reset(self, **kwargs):
        """重置环境并初始化坐标系"""
        obs, info = self.env.reset(**kwargs)
        
        # 重置控制器
        self.controller.reset()
        self.peg_grasped = False
        self.control_phase = "approach"
        self.insertion_stage = "move_to_front"
        
        # 获取hole位置和方向
        hole_pos, hole_orientation = self._get_hole_info()
        self.task_coord_system = TaskCoordinateSystem(hole_pos, hole_orientation)
        self.success_checker = InsertionSuccessChecker(self.params, self.task_coord_system)
        
        # 记录初始状态
        self.initial_box_pos = self._get_box_position()
        
        # 重置记录数据
        self.episode_data = {
            'forces': [],
            'torques': [],
            'positions': [],
            'orientations': [],
            'contact_forces': [],
            'control_outputs': [],
            'phases': [],
            'insertion_status': [],
            'box_positions': []
        }
        
        print(f"环境重置完成")
        print(f"Hole位置: {hole_pos}")
        print(f"插入方向: {hole_orientation}")
        
        return obs, info
    
    def _get_hole_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取hole的位置和方向"""
        try:
            # 直接使用hole site的位置
            hole_site_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_SITE, 'hole')
            hole_pos = self.env.unwrapped.data.site_xpos[hole_site_id].copy()
            
            # 获取box的方向来确定hole的开口方向
            box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_BODY, 'box')
            box_quat = self.env.unwrapped.data.xquat[box_body_id].copy()
            
            # 根据XML，hole的开口朝向Y轴正方向（在box坐标系中）
            rotation = Rotation.from_quat(box_quat)
            hole_direction = rotation.apply(np.array([0, 1, 0]))  # Y正方向为hole开口方向
            
        except:
            # 备用方案：使用目标位置估算
            print("Warning: 无法直接获取hole信息，使用估算值")
            hole_pos = self.env.unwrapped._target_pos.copy() if hasattr(self.env.unwrapped, '_target_pos') and self.env.unwrapped._target_pos is not None else np.array([-0.3, 0.504, 0.13])
            hole_direction = np.array([0, 1, 0])
        
        return hole_pos, hole_direction
    
    def _get_box_position(self) -> np.ndarray:
        """获取box的当前位置"""
        try:
            box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                           mujoco.mjtObj.mjOBJ_BODY, 'box')
            return self.env.unwrapped.data.xpos[box_body_id].copy()
        except:
            return np.zeros(3)
    
    def _get_peg_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取peg的中心位置和头部位置"""
        # peg中心位置
        peg_center = self.env.unwrapped._get_pos_objects()
        
        # peg头部位置（通过pegHead site获取）
        try:
            peg_head_site_id = mujoco.mj_name2id(self.env.unwrapped.model, 
                                               mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
            peg_head = self.env.unwrapped.data.site_xpos[peg_head_site_id].copy()
        except:
            # 备用方案：基于peg的方向估算头部位置
            peg_quat = self.env.unwrapped._get_quat_objects()
            rotation = Rotation.from_quat(peg_quat)
            # 根据XML，pegHead在peg的-X方向0.1m处
            peg_head = peg_center + rotation.apply(np.array([-0.1, 0, 0]))
        
        return peg_center, peg_head
    
    def _get_peg_rotation(self) -> Rotation:
        """获取peg的当前姿态"""
        peg_quat = self.env.unwrapped._get_quat_objects()
        return Rotation.from_quat(peg_quat)
    
    def _get_gripper_distance(self) -> float:
        """从观测中获取夹爪距离，用于判断是否成功抓取"""
        obs = self.env.unwrapped._get_obs()
        return obs[3]
    
    def _detect_grasp(self) -> bool:
        """检测是否抓取了peg"""
        gripper_distance = self._get_gripper_distance()
        peg_center, _ = self._get_peg_positions()
        hand_pos = self.env.unwrapped.get_endeff_pos()
        
        distance_to_peg = np.linalg.norm(hand_pos - peg_center)
        
        return gripper_distance < 0.3 and distance_to_peg < 0.05
    
    def _update_control_phase(self):
        """更新控制阶段"""
        peg_center, peg_head = self._get_peg_positions()
        hole_pos, _ = self._get_hole_info()
        
        if not self.peg_grasped:
            self.control_phase = "approach"
            self.insertion_stage = "move_to_front"
        else:
            distance_to_hole = np.linalg.norm(peg_head - hole_pos)
            
            if distance_to_hole > self.params.approach_distance:
                self.control_phase = "approach"
                self.insertion_stage = "move_to_front"
            elif distance_to_hole > self.params.alignment_distance:
                self.control_phase = "align"
                self.insertion_stage = "align_orientation"
            else:
                self.control_phase = "insert"
                self.insertion_stage = "insert"
    
    def step(self, action):
        """执行一步，应用力位混合控制"""
        # 检测抓取状态
        self.peg_grasped = self._detect_grasp()
        
        # 更新控制阶段
        self._update_control_phase()
        
        # 获取当前状态
        peg_center, peg_head = self._get_peg_positions()
        peg_rotation = self._get_peg_rotation()
        hole_pos, hole_orientation = self._get_hole_info()
        contact_force, contact_torque = self.force_extractor.get_contact_forces_and_torques()
        
        # 根据控制阶段选择控制策略
        if self.control_phase == "approach":
            # 纯位置控制阶段：移动到hole前方
            modified_action = self._apply_approach_control(action, peg_head, hole_pos)
        elif self.control_phase == "align":
            # 对齐阶段：调整姿态
            modified_action = self._apply_alignment_control(
                action, peg_head, peg_rotation, hole_pos, contact_force, contact_torque)
        elif self.control_phase == "insert":
            # 插入阶段：力位混合控制
            modified_action = self._apply_insertion_control(
                action, peg_head, peg_rotation, hole_pos, contact_force, contact_torque)
        else:
            modified_action = action.copy()
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 检查插入状态
        insertion_status = self.success_checker.check_insertion_success(
            peg_head, peg_center, peg_rotation)
        
        # 记录数据
        self._record_data(peg_center, peg_head, peg_rotation, contact_force, 
                         contact_torque, modified_action, insertion_status)
        
        # 添加额外的评估指标
        info.update(self._compute_evaluation_metrics(insertion_status))
        
        return obs, reward, terminated, truncated, info
    
    def _apply_approach_control(self, action, peg_head_pos, hole_pos):
        """接近阶段控制：移动到hole前方"""
        # 计算hole前方位置
        hole_front_pos = hole_pos + self.task_coord_system.y_axis * self.params.approach_distance
        
        # 简单位置控制
        direction = hole_front_pos - peg_head_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.001:
            direction = direction / distance
            modified_action = action.copy()
            scale = min(0.1, distance * 2.0)  # 距离越近速度越慢
            modified_action[:3] += direction * scale
            return modified_action
        
        return action.copy()
    
    def _apply_alignment_control(self, action, peg_head_pos, peg_rotation, hole_pos, 
                               contact_force, contact_torque):
        """对齐阶段控制：调整位置和姿态"""
        if self.task_coord_system is None:
            return action.copy()
        
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)
        
        # 获取姿态误差
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 目标位置：hole前方一点
        target_pos_task = hole_pos_task + np.array([0, self.params.alignment_distance, 0])
        target_orientation = np.zeros(3)  # 目标姿态误差为0
        
        # 目标力和力矩
        target_force_task = np.array([0, 0, 0])  # 对齐阶段不需要大的力
        target_torque_task = np.array([0, 0, 0])
        
        # 获取选择矩阵
        selection_matrix_pos, selection_matrix_rot = self._get_selection_matrices()
        
        # 计算控制输出
        dt = 1.0 / self.params.force_control_freq
        pos_output, rot_output = self.controller.compute_control(
            peg_head_task, target_pos_task, orientation_error, target_orientation,
            contact_force_task, target_force_task, contact_torque_task, target_torque_task,
            selection_matrix_pos, selection_matrix_rot, dt)
        
        # 转换回世界坐标系并应用
        pos_output_world = self.task_coord_system.task_force_to_world(pos_output)
        
        modified_action = action.copy()
        pos_scale = 0.02
        modified_action[:3] += pos_output_world * pos_scale
        
        return modified_action
    
    def _apply_insertion_control(self, action, peg_head_pos, peg_rotation, hole_pos, 
                               contact_force, contact_torque):
        """插入阶段控制：力位混合控制"""
        if self.task_coord_system is None:
            return action.copy()
        
        # 转换到任务坐标系
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        hole_pos_task = self.task_coord_system.world_to_task(hole_pos)
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)
        
        # 获取姿态误差
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 目标位置：深入hole内部
        target_pos_task = hole_pos_task + np.array([0, -self.params.min_insertion_depth, 0])
        target_orientation = np.zeros(3)  # 目标姿态误差为0
        
        # 目标力和力矩
        target_force_task = np.array([0, 1.0, 0])  # Y方向允许推力
        target_torque_task = np.array([0, 0, 0])
        
        # 获取选择矩阵
        selection_matrix_pos, selection_matrix_rot = self._get_selection_matrices()
        
        # 计算控制输出
        dt = 1.0 / self.params.force_control_freq
        pos_output, rot_output = self.controller.compute_control(
            peg_head_task, target_pos_task, orientation_error, target_orientation,
            contact_force_task, target_force_task, contact_torque_task, target_torque_task,
            selection_matrix_pos, selection_matrix_rot, dt)
        
        # 转换回世界坐标系并应用
        pos_output_world = self.task_coord_system.task_force_to_world(pos_output)
        
        modified_action = action.copy()
        pos_scale = 0.01
        modified_action[:3] += pos_output_world * pos_scale
        
        return modified_action
    
    def _get_selection_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取位置和姿态的选择矩阵"""
        # 默认配置：径向方向（XZ）做力控制，插入方向（Y）做位置控制
        # 姿态控制：径向旋转用力控制，轴向旋转用位置控制
        selection_matrix_pos = np.diag([0, 1, 0])  # Y方向位置控制，XZ方向力控制
        selection_matrix_rot = np.diag([0, 1, 1])  # YZ方向姿态控制，X方向力矩控制
        return selection_matrix_pos, selection_matrix_rot
    
    def _record_data(self, peg_center, peg_head, peg_rotation, contact_force, 
                    contact_torque, control_output, insertion_status):
        """记录实验数据"""
        self.episode_data['positions'].append(peg_center.copy())
        self.episode_data['orientations'].append(peg_rotation.as_quat().copy())
        self.episode_data['contact_forces'].append(contact_force.copy())
        self.episode_data['torques'].append(contact_torque.copy())
        self.episode_data['control_outputs'].append(control_output.copy())
        self.episode_data['phases'].append(f"{self.control_phase}_{self.insertion_stage}")
        self.episode_data['insertion_status'].append(insertion_status.copy())
        self.episode_data['box_positions'].append(self._get_box_position())
    
    def _compute_evaluation_metrics(self, insertion_status: Dict) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 计算最大接触力和力矩
        if self.episode_data['contact_forces']:
            forces = np.array(self.episode_data['contact_forces'])
            torques = np.array(self.episode_data['torques'])
            metrics['max_contact_force'] = np.max(np.linalg.norm(forces, axis=1))
            metrics['avg_contact_force'] = np.mean(np.linalg.norm(forces, axis=1))
            metrics['max_contact_torque'] = np.max(np.linalg.norm(torques, axis=1))
            metrics['avg_contact_torque'] = np.mean(np.linalg.norm(torques, axis=1))
        else:
            metrics['max_contact_force'] = 0.0
            metrics['avg_contact_force'] = 0.0
            metrics['max_contact_torque'] = 0.0
            metrics['avg_contact_torque'] = 0.0
        
        # 计算box移动距离
        if self.initial_box_pos is not None:
            current_box_pos = self._get_box_position()
            box_displacement = np.linalg.norm(current_box_pos - self.initial_box_pos)
            metrics['box_displacement'] = box_displacement
            metrics['environment_damage'] = box_displacement > 0.01
        else:
            metrics['box_displacement'] = 0.0
            metrics['environment_damage'] = False
        
        # 插入相关指标
        metrics['insertion_depth'] = insertion_status['insertion_depth']
        metrics['radial_distance'] = insertion_status['radial_distance']
        metrics['orientation_error'] = insertion_status['orientation_error']
        metrics['insertion_success'] = insertion_status['success']
        metrics['is_inside_hole'] = insertion_status['is_inside_hole']
        metrics['sufficient_depth'] = insertion_status['sufficient_depth']
        metrics['is_aligned'] = insertion_status['is_aligned']
        metrics['is_orientation_aligned'] = insertion_status['is_orientation_aligned']
        
        return metrics

class SimplePolicy:
    """改进的简单策略"""
    
    def __init__(self):
        self.phase = "reach"
        self.grasp_threshold = 0.05
        
    def get_action(self, obs):
        """生成动作"""
        hand_pos = obs[:3]
        gripper_distance = obs[3]
        obj_pos = obs[4:7]
        goal_pos = obs[-3:]
        
        action = np.zeros(4)
        
        # 计算到目标的距离
        hand_to_obj = np.linalg.norm(hand_pos - obj_pos)
        obj_to_goal = np.linalg.norm(obj_pos - goal_pos)
        
        if self.phase == "reach":
            # 移动到物体
            if hand_to_obj > self.grasp_threshold:
                direction = (obj_pos - hand_pos) / max(hand_to_obj, 1e-6)
                action[:3] = direction * 0.1
                action[3] = -1  # 打开夹爪
            else:
                self.phase = "grasp"
                
        elif self.phase == "grasp":
            # 抓取物体
            action[:3] = (obj_pos - hand_pos) * 0.5
            action[3] = 1  # 关闭夹爪
            
            if gripper_distance < 0.3 and hand_to_obj < 0.03:
                self.phase = "transport"
                
        elif self.phase == "transport":
            # 运输到目标位置附近（不直接插入）
            target_offset = 0.08  # 停在hole前方8cm
            adjusted_goal = goal_pos + np.array([0, target_offset, 0])  # 假设Y是插入方向
            
            if np.linalg.norm(obj_pos - adjusted_goal) > 0.02:
                direction = (adjusted_goal - obj_pos) / max(np.linalg.norm(adjusted_goal - obj_pos), 1e-6)
                action[:3] = direction * 0.05  # 更慢的速度
                action[3] = 1  # 保持夹爪关闭
            else:
                # 到达目标附近，进行细微调整
                action[:3] = (adjusted_goal - obj_pos) * 1.0
                action[3] = 1
        
        # 限制动作范围
        action[:3] = np.clip(action[:3], -1, 1)
        action[3] = np.clip(action[3], -1, 1)
        
        return action

class SelectionMatrixExperiment:
    """改进的方向选择矩阵实验类"""
    
    def __init__(self, env_name: str = 'peg-insert-side-v3'):
        self.env_name = env_name
        self.results = []
    
    def create_env(self, selection_matrix_func=None):
        """创建带有指定选择矩阵的环境"""
        ml1 = metaworld.ML1(self.env_name, seed=42)
        # 不使用render以提高速度
        env = ml1.train_classes[self.env_name](render_mode='human')
        task = ml1.train_tasks[0]
        env.set_task(task)
        
        # 包装为混合控制环境
        hybrid_env = HybridControlWrapper(env)
        
        # 替换选择矩阵函数
        if selection_matrix_func:
            hybrid_env._get_selection_matrices = selection_matrix_func
        
        return hybrid_env
    
    def define_selection_matrices(self):
        """定义不同的方向选择矩阵配置"""
        matrices = {
            "纯位置控制": lambda: (np.eye(3), np.eye(3)),
            "Y位置XZ力": lambda: (np.diag([0, 1, 0]), np.diag([0, 1, 1])),  # 推荐配置
            "XZ位置Y力": lambda: (np.diag([1, 0, 1]), np.diag([1, 0, 0])),
            "纯力控制": lambda: (np.zeros((3, 3)), np.zeros((3, 3))),
            "位置控制_力矩控制": lambda: (np.eye(3), np.zeros((3, 3))),
            "力控制_姿态控制": lambda: (np.zeros((3, 3)), np.eye(3)),
            "混合控制1": lambda: (np.diag([0, 1, 0]), np.diag([0, 0, 1])),
            "混合控制2": lambda: (np.diag([1, 1, 0]), np.diag([0, 1, 0])),
        }
        return matrices
    
    def run_experiment(self, num_episodes: int = 2, max_steps: int = 800):
        """运行选择矩阵对比实验"""
        matrices = self.define_selection_matrices()
        
        print(f"开始6DOF方向选择矩阵实验，测试{len(matrices)}种配置...")
        
        for matrix_name, matrix_func in matrices.items():
            print(f"\n测试配置: {matrix_name}")
            
            env = self.create_env(matrix_func)
            episode_results = []
            
            for episode in range(num_episodes):
                obs, info = env.reset()
                policy = SimplePolicy()
                
                episode_data = {
                    'matrix_name': matrix_name,
                    'episode': episode,
                    'total_reward': 0,
                    'success': False,
                    'metrics': {}
                }
                
                for step in range(max_steps):
                    env.render()
                    action = policy.get_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_data['total_reward'] += reward
                    
                    if terminated or truncated:
                        break
                
                # 记录最终指标
                episode_data['success'] = info.get('insertion_success', False)
                episode_data['metrics'] = {
                    'max_contact_force': info.get('max_contact_force', 0),
                    'max_contact_torque': info.get('max_contact_torque', 0),
                    'box_displacement': info.get('box_displacement', 0),
                    'insertion_depth': info.get('insertion_depth', 0),
                    'radial_distance': info.get('radial_distance', 0),
                    'orientation_error': info.get('orientation_error', 0),
                    'environment_damage': info.get('environment_damage', False),
                    'is_inside_hole': info.get('is_inside_hole', False),
                    'is_orientation_aligned': info.get('is_orientation_aligned', False)
                }
                
                episode_results.append(episode_data)
                print(f"  Episode {episode}: Success={episode_data['success']}, "
                      f"Depth={episode_data['metrics']['insertion_depth']:.3f}, "
                      f"Radial={episode_data['metrics']['radial_distance']:.3f}, "
                      f"Orient={episode_data['metrics']['orientation_error']:.3f}")
            
            # 计算平均结果
            avg_results = self._compute_average_results(episode_results)
            self.results.append(avg_results)
            
            env.close()
    
    def _compute_average_results(self, episode_results):
        """计算平均结果"""
        if not episode_results:
            return {}
        
        avg_result = {
            'matrix_name': episode_results[0]['matrix_name'],
            'num_episodes': len(episode_results),
            'success_rate': np.mean([r['success'] for r in episode_results]),
            'avg_reward': np.mean([r['total_reward'] for r in episode_results]),
            'avg_max_contact_force': np.mean([r['metrics']['max_contact_force'] for r in episode_results]),
            'avg_max_contact_torque': np.mean([r['metrics']['max_contact_torque'] for r in episode_results]),
            'avg_box_displacement': np.mean([r['metrics']['box_displacement'] for r in episode_results]),
            'avg_insertion_depth': np.mean([r['metrics']['insertion_depth'] for r in episode_results]),
            'avg_radial_distance': np.mean([r['metrics']['radial_distance'] for r in episode_results]),
            'avg_orientation_error': np.mean([r['metrics']['orientation_error'] for r in episode_results]),
            'damage_rate': np.mean([r['metrics']['environment_damage'] for r in episode_results]),
            'inside_hole_rate': np.mean([r['metrics']['is_inside_hole'] for r in episode_results]),
            'orientation_aligned_rate': np.mean([r['metrics']['is_orientation_aligned'] for r in episode_results])
        }
        
        return avg_result
    
    def print_results(self):
        """打印实验结果"""
        print("\n" + "="*120)
        print("6DOF方向选择矩阵实验结果汇总")
        print("="*120)
        
        # 表头
        header = f"{'配置名称':<20} {'成功率':<8} {'插入深度':<10} {'径向距离':<10} {'姿态误差':<10} {'孔内率':<8} {'姿态对齐率':<12} {'接触力':<10}"
        print(header)
        print("-"*120)
        
        # 数据行
        for result in self.results:
            row = (f"{result['matrix_name']:<20} "
                   f"{result['success_rate']:<8.2f} "
                   f"{result['avg_insertion_depth']:<10.3f} "
                   f"{result['avg_radial_distance']:<10.3f} "
                   f"{result['avg_orientation_error']:<10.3f} "
                   f"{result['inside_hole_rate']:<8.2f} "
                   f"{result['orientation_aligned_rate']:<12.2f} "
                   f"{result['avg_max_contact_force']:<10.1f}")
            print(row)
        
        print("-"*120)
        
        # 找出最优配置
        if self.results:
            best_success = max(self.results, key=lambda x: x['success_rate'])
            best_precision = min(self.results, key=lambda x: (x['avg_radial_distance'] + x['avg_orientation_error']) if x['success_rate'] > 0 else float('inf'))
            
            print(f"\n最高成功率: {best_success['matrix_name']} ({best_success['success_rate']:.2f})")
            if best_precision['success_rate'] > 0:
                print(f"最高精度: {best_precision['matrix_name']} (综合误差: {best_precision['avg_radial_distance'] + best_precision['avg_orientation_error']:.3f})")

def demo_enhanced_hybrid_control():
    """演示改进的6DOF力位混合控制"""
    print("开始改进的6DOF力位混合控制演示...")
    
    # 创建实验环境
    experiment = SelectionMatrixExperiment()
    
    # 运行实验
    experiment.run_experiment(num_episodes=3, max_steps=400)
    
    # 显示结果
    experiment.print_results()

if __name__ == "__main__":
    demo_enhanced_hybrid_control()
```

### src_test/controllers.py

*大小: 2.4 KB | Token: 677*

```python
# controllers.py
import numpy as np
from typing import Tuple
from params import ControlParams

class Enhanced6DOFController:
    """增强的6DOF力位混合控制器"""
    
    def __init__(self, params: ControlParams):
        self.params = params
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        
    def reset(self):
        self.force_integral = np.zeros(3)
        self.torque_integral = np.zeros(3)
        
    def compute_control(self, 
                       current_pos: np.ndarray,
                       target_pos: np.ndarray,
                       current_orientation_error: np.ndarray,
                       target_orientation_error: np.ndarray,
                       current_force: np.ndarray,
                       target_force: np.ndarray,
                       current_torque: np.ndarray,
                       target_torque: np.ndarray,
                       selection_matrix_pos: np.ndarray,
                       selection_matrix_rot: np.ndarray,
                       dt: float) -> Tuple[np.ndarray, np.ndarray]:

        pos_error = target_pos - current_pos
        rot_error = target_orientation_error - current_orientation_error
        
        force_error = target_force - current_force
        torque_error = target_torque - current_torque
        
        force_mask = 1 - np.diag(selection_matrix_pos)
        torque_mask = 1 - np.diag(selection_matrix_rot)
        
        self.force_integral += force_error * force_mask * dt
        self.torque_integral += torque_error * torque_mask * dt
        
        self.force_integral = np.clip(self.force_integral, -1.0, 1.0)
        self.torque_integral = np.clip(self.torque_integral, -0.5, 0.5)
        
        pos_control = self.params.kp_pos * pos_error
        rot_control = self.params.kp_rot * rot_error
        
        force_control_delta = (self.params.kp_force * force_error + 
                               self.params.ki_force * self.force_integral)
        torque_control_delta = (self.params.kp_torque * torque_error + 
                                self.params.ki_torque * self.torque_integral)
        
        pos_output = (selection_matrix_pos @ pos_control + 
                     (np.eye(3) - selection_matrix_pos) @ force_control_delta)
        rot_output = (selection_matrix_rot @ rot_control + 
                     (np.eye(3) - selection_matrix_rot) @ torque_control_delta)
        
        return pos_output, rot_output
```

### src_test/coordinate_systems.py

*大小: 1.8 KB | Token: 485*

```python
# coordinate_systems.py
import numpy as np
from scipy.spatial.transform import Rotation

class TaskCoordinateSystem:
    """任务坐标系：以hole为基准建立坐标系"""
    
    def __init__(self, hole_pos: np.ndarray, hole_orientation: np.ndarray):
        self.hole_pos = hole_pos.copy()
        
        # Y轴为插入方向（指向hole内部，即Y负方向）
        self.y_axis = -hole_orientation / np.linalg.norm(hole_orientation)
        
        # 构建正交的X、Z轴
        if abs(self.y_axis[2]) < 0.9:
            self.z_axis = np.cross(self.y_axis, [0, 0, 1])
        else:
            self.z_axis = np.cross(self.y_axis, [1, 0, 0])
        self.z_axis /= np.linalg.norm(self.z_axis)
        self.x_axis = np.cross(self.y_axis, self.z_axis)
        
        # 旋转矩阵：世界坐标系 -> 任务坐标系
        self.rotation_matrix = np.column_stack([self.x_axis, self.y_axis, self.z_axis])
        self.target_rotation = Rotation.from_matrix(self.rotation_matrix.T)
        
    def world_to_task(self, world_pos: np.ndarray) -> np.ndarray:
        relative_pos = world_pos - self.hole_pos
        return self.rotation_matrix.T @ relative_pos
    
    def task_to_world(self, task_pos: np.ndarray) -> np.ndarray:
        world_relative = self.rotation_matrix @ task_pos
        return world_relative + self.hole_pos
    
    def world_force_to_task(self, world_force: np.ndarray) -> np.ndarray:
        return self.rotation_matrix.T @ world_force
    
    def task_force_to_world(self, task_force: np.ndarray) -> np.ndarray:
        return self.rotation_matrix @ task_force
    
    def get_orientation_error(self, current_rotation: Rotation) -> np.ndarray:
        relative_rotation = self.target_rotation * current_rotation.inv()
        axis_angle = relative_rotation.as_rotvec()
        return self.rotation_matrix.T @ axis_angle
```

### src_test/demo_position_control.py

*大小: 3.7 KB | Token: 928*

```python
# demo_position_control.py
import metaworld
import numpy as np
import time

# 从本地模块导入
from wrapper import HybridControlWrapper
from policy import SimplePolicy
from params import ControlParams

# 1. 创建环境
ml1 = metaworld.ML1('peg-insert-side-v3', seed=42) 
env = ml1.train_classes['peg-insert-side-v3'](render_mode='human')
task = ml1.train_tasks[0]
env.set_task(task)

# 2. 使用修改后的控制参数
control_params = ControlParams()
hybrid_env = HybridControlWrapper(env, control_params)

# 3. 定义并设置选择矩阵函数（纯位置控制）
def pure_position_control_matrices():
    """返回位置和姿态都为纯位置控制的选择矩阵"""
    pos_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes position controlled
    rot_selection = np.eye(3)  # [1, 1, 1] on diag -> All axes orientation controlled
    return pos_selection, rot_selection
    
hybrid_env.set_selection_matrices_func(pure_position_control_matrices)
print("控制模式: 纯位置控制 (选择矩阵为单位阵)")

# 4. 初始化策略和环境
policy = SimplePolicy()
obs, info = hybrid_env.reset()
policy.reset()

# 5. 添加初始状态检查
print(f"初始状态:")
print(f"  Hand position: {obs[:3]}")
print(f"  Object position: {obs[4:7]}")
print(f"  Goal position: {env.unwrapped._target_pos}")
print(f"  Distance to object: {np.linalg.norm(obs[:3] - obs[4:7]):.3f}")

# 6. 运行一个episode
max_steps = 800
prev_obj_pos = obs[4:7].copy()
stuck_counter = 0

for step in range(max_steps):
    # 渲染环境
    hybrid_env.render()
    
    # 从简单策略获取高级动作
    action = policy.get_action(obs, hybrid_env.unwrapped)
    
    # 执行一步
    obs, reward, terminated, truncated, info = hybrid_env.step(action)
    
    # 检查是否卡住
    current_obj_pos = obs[4:7]
    if np.linalg.norm(current_obj_pos - prev_obj_pos) < 0.001:
        stuck_counter += 1
    else:
        stuck_counter = 0
    prev_obj_pos = current_obj_pos.copy()
    
    # 详细的调试信息
    if (step + 1) % 50 == 0:
        hand_pos = obs[:3]
        obj_pos = obs[4:7]
        goal_pos = env.unwrapped._target_pos
        
        print(f"\nStep: {step+1}")
        print(f"  Policy phase: {policy.phase}")
        print(f"  Insertion phase: {info.get('insertion_phase', 'unknown')}")
        print(f"  Hand pos: [{hand_pos[0]:.3f}, {hand_pos[1]:.3f}, {hand_pos[2]:.3f}]")
        print(f"  Object pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        print(f"  Goal pos: [{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}]")
        print(f"  Hand-Object dist: {np.linalg.norm(hand_pos - obj_pos):.3f}")
        print(f"  Object-Goal dist: {np.linalg.norm(obj_pos - goal_pos):.3f}")
        print(f"  Distance to target: {info.get('distance_to_target', 0):.3f}")
        print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
        print(f"  Success: {info.get('success', False)}")
        print(f"  Insertion depth: {info.get('insertion_depth', 0):.3f}")
        print(f"  Stuck counter: {stuck_counter}")

    # 安全检查：如果长时间卡住，重置环境
    if stuck_counter > 100:
        print("\n警告：检测到长时间卡住，重置环境...")
        obs, info = hybrid_env.reset()
        policy.reset()
        stuck_counter = 0
        continue

    # 如果任务成功，提前结束
    if info.get('success', False):
        print("\n*** 任务成功! ***")
        time.sleep(3) # 暂停3秒查看结果
        break

    if terminated or truncated:
        print(f"\n任务结束: terminated={terminated}, truncated={truncated}")
        break
        
    time.sleep(0.02) # 减慢渲染速度，便于观察
    
print("\n演示结束。")
hybrid_env.close()
```

### src_test/force_extractor.py

*大小: 1.5 KB | Token: 414*

```python
# force_extractor.py
import numpy as np
import mujoco
from typing import Tuple

class ForceExtractor:
    """从仿真中提取接触力和力矩"""
    
    def __init__(self, env):
        self.env = env
        self.model = env.unwrapped.model
        self.data = env.unwrapped.data
        
        self.peg_geom_ids = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and 'peg' in geom_name.lower():
                self.peg_geom_ids.append(i)
    
    def get_contact_forces_and_torques(self) -> Tuple[np.ndarray, np.ndarray]:
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            if contact.geom1 in self.peg_geom_ids or contact.geom2 in self.peg_geom_ids:
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force = c_array[:3]
                
                contact_frame = contact.frame.reshape(3, 3)
                world_force = contact_frame @ contact_force
                
                peg_pos = self.env.unwrapped._get_pos_objects()
                r = contact.pos - peg_pos
                contact_torque = np.cross(r, world_force)
                
                total_force += world_force
                total_torque += contact_torque
        
        return total_force, total_torque
```

### src_test/params.py

*大小: 1.2 KB | Token: 275*

```python
# params.py
from dataclasses import dataclass

@dataclass
class ControlParams:
    """力位混合控制参数"""
    # 位置控制参数 - 大幅降低增益
    kp_pos: float = 10.0   # 从1000.0降到10.0
    kd_pos: float = 1.0    # 从100.0降到1.0
    
    # 姿态控制参数 - 降低增益
    kp_rot: float = 5.0    # 从500.0降到5.0
    kd_rot: float = 0.5    # 从50.0降到0.5
    
    # 力控制参数 - 保持较小
    kp_force: float = 0.001
    ki_force: float = 0.0001
    kp_torque: float = 0.0005
    ki_torque: float = 0.00005
    force_deadzone: float = 0.5
    torque_deadzone: float = 0.1
    
    # 几何参数
    peg_radius: float = 0.015
    hole_radius: float = 0.025
    insertion_tolerance: float = 0.008
    min_insertion_depth: float = 0.06
    
    # 插入策略参数
    approach_distance: float = 0.0    # 接近阶段的距离阈值
    alignment_distance: float = 0.9   # 对齐阶段目标位置距离hole的距离
    max_orientation_error: float = 0.15  # 允许的最大姿态误差(弧度)
    
    # 控制频率
    position_control_freq: float = 500.0
    force_control_freq: float = 1000.0
    
    # 切换阈值
    switch_distance: float = 0.05
```

### src_test/policy.py

*大小: 4.5 KB | Token: 1.1K*

```python
# policy.py
import numpy as np

class SimplePolicy:
    """提供高级运动指令的简单策略"""
    def __init__(self):
        self.phase = "reach"
        self.grasp_threshold = 0.05
        self.transport_threshold = 0.1  # 新增：运输阶段的距离阈值
        
    def reset(self):
        self.phase = "reach"

    def get_action(self, obs, env_unwrapped):
        hand_pos = obs[:3]
        gripper_open = obs[3] > 0.5 # 简化为布尔值
        obj_pos = obs[4:7]
        goal_pos = env_unwrapped._target_pos

        action = np.zeros(4)
        hand_to_obj_dist = np.linalg.norm(hand_pos - obj_pos)
        obj_to_goal_dist = np.linalg.norm(obj_pos - goal_pos)

        if self.phase == "reach":
            # 更温和的接近动作
            direction = obj_pos - hand_pos
            distance = np.linalg.norm(direction)
            if distance > 0.001:  # 避免除零
                action[:3] = direction / distance * min(distance * 5.0, 1.0)  # 限制速度
            action[3] = -1  # 打开夹爪
            
            if hand_to_obj_dist < self.grasp_threshold:
                self.phase = "grasp"
        
        elif self.phase == "grasp":
            # 保持在物体附近
            direction = obj_pos - hand_pos
            distance = np.linalg.norm(direction)
            if distance > 0.001:
                action[:3] = direction / distance * min(distance * 2.0, 0.5)  # 更温和的移动
            action[3] = 1 # 关闭夹爪
            
            # 检测是否抓取成功
            if hand_to_obj_dist < 0.04 and not gripper_open:
                self.phase = "transport"

        elif self.phase == "transport":
            # 更智能的运输策略 - 分为接近和插入两个子阶段
            goal_pos = env_unwrapped._target_pos
            
            # 计算peg head到goal的距离（更准确的距离计算）
            try:
                # 尝试获取peg head位置
                import mujoco
                peg_head_site_id = mujoco.mj_name2id(env_unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
                peg_head_pos = env_unwrapped.data.site_xpos[peg_head_site_id].copy()
                head_to_goal_dist = np.linalg.norm(peg_head_pos - goal_pos)
            except:
                # 如果获取失败，使用peg center位置
                print("Can't get the pos of peg head")
                head_to_goal_dist = obj_to_goal_dist
            
            if head_to_goal_dist > 0.10:  # 距离较远时，快速接近
                direction = goal_pos - obj_pos
                distance = np.linalg.norm(direction)
                if distance > 0.001:
                    # 根据距离调整速度
                    speed_factor = min(distance * 1.5, 0.6)
                    action[:3] = direction / distance * speed_factor
                    
            elif head_to_goal_dist > 0.05:  # 中等距离，准备插入
                # 这个阶段需要更精确的对齐
                direction = goal_pos - obj_pos
                distance = np.linalg.norm(direction)
                if distance > 0.001:
                    # 缓慢接近，准备插入
                    action[:3] = direction / distance * min(distance * 5.0, 0.3)
                    
            else:  # 非常接近时，进行插入动作
                # 获取hole的方向进行插入
                try:
                    hole_site_id = mujoco.mj_name2id(env_unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'hole')
                    hole_pos = env_unwrapped.data.site_xpos[hole_site_id].copy()
                    
                    # 计算插入方向（从peg位置指向hole内部）
                    insertion_direction = hole_pos - obj_pos
                    distance = np.linalg.norm(insertion_direction)
                    if distance > 0.001:
                        # 沿着插入方向缓慢移动
                        action[:3] = insertion_direction / distance * 0.2
                except:
                    print("获取失败，使用默认的慢速接近")
                    # 如果获取失败，使用默认的慢速接近
                    direction = goal_pos - obj_pos
                    distance = np.linalg.norm(direction)
                    if distance > 0.001:
                        action[:3] = direction / distance * 0.1
                    
            action[3] = 1 # 保持抓取

        # 额外的安全限制
        action[:3] = np.clip(action[:3], -0.8, 0.8)  # 限制最大动作幅度
        
        return action
```

### src_test/standalone_peg_insert.py

*大小: 7.9 KB | Token: 2.0K*

```python
#!/usr/bin/env python3
"""
完全独立的Peg Insert Side环境测试脚本
直接使用MuJoCo环境而不依赖metaworld模块
"""

import numpy as np
import time
import mujoco

class SimplePegInsertEnv:
    """简化版的Peg Insert Side环境"""
    
    def __init__(self, render_mode='human'):
        # 加载模型
        import os
        xml_path = os.path.join(os.path.dirname(__file__), 'xml/sawyer_peg_insertion_side.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化参数
        self.render_mode = render_mode
        self.viewer = None
        self.max_steps = 500
        self.current_step = 0
        
        # 动作空间: [dx, dy, dz, grip]
        self.action_space = np.array([[-1, -1, -1, -1], [1, 1, 1, 1]])
        
        # 观测空间维度
        self.obs_dim = 18  # 简化观测空间
        
        # 目标位置
        self.target_pos = np.array([-0.3, 0.6, 0.13])
        
        # 重置环境
        self.reset()
        
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 设置初始状态
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        
        # 设置手部初始位置
        self.data.qpos[0:3] = [0, 0.6, 0.2]  # 手部位置
        self.data.qpos[3] = 0.05  # 夹爪初始张开
        
        # 设置peg初始位置
        self.data.qpos[9:12] = [0, 0.6, 0.02]  # peg位置
        
        # 前向动力学计算
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        """获取观测值"""
        # 手部位置
        hand_pos = self.data.body('hand').xpos.copy()
        
        # 夹爪开合度
        gripper_state = self.data.qpos[3]
        
        # peg位置
        peg_pos = self.data.body('peg').xpos.copy()
        
        # peg四元数
        peg_quat = self.data.body('peg').xquat.copy()
        
        # 组合观测值
        obs = np.concatenate([
            hand_pos,           # 3: 手部位置
            [gripper_state],    # 1: 夹爪状态
            peg_pos,            # 3: peg位置
            peg_quat,           # 4: peg四元数
            hand_pos,           # 3: 重复手部位置（用于帧堆叠）
            [gripper_state],    # 1: 重复夹爪状态
            peg_pos,            # 3: 重复peg位置
        ])
        
        return obs
        
    def step(self, action):
        """执行一步动作"""
        self.current_step += 1
        
        # 确保动作在范围内
        action = np.clip(action, self.action_space[0], self.action_space[1])
        
        # 应用动作到机械臂
        # 控制手部位置
        self.data.ctrl[0] = action[0] * 0.1  # dx
        self.data.ctrl[1] = action[1] * 0.1  # dy
        self.data.ctrl[2] = action[2] * 0.1  # dz
        
        # 控制夹爪
        self.data.ctrl[3] = action[3]  # grip
        
        # 执行仿真
        mujoco.mj_step(self.model, self.data)
        
        # 获取新状态
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._compute_reward(obs)
        
        # 检查是否完成
        terminated = self._check_success(obs)
        truncated = self.current_step >= self.max_steps
        
        # 信息
        info = {
            'success': terminated,
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
        
    def _compute_reward(self, obs):
        """计算奖励"""
        hand_pos = obs[0:3]
        peg_pos = obs[4:7]
        target_pos = self.target_pos
        
        # 计算peg到目标的距离
        peg_to_target = np.linalg.norm(peg_pos - target_pos)
        
        # 计算手到peg的距离
        hand_to_peg = np.linalg.norm(hand_pos - peg_pos)
        
        # 基础奖励：peg接近目标
        reward = 1.0 - min(float(peg_to_target / 0.5), 1.0)
        
        # 额外奖励：手接近peg（抓取阶段）
        if hand_to_peg < 0.1:
            reward += 0.5
            
        # 成功奖励
        if peg_to_target < 0.07:
            reward += 5.0
            
        return reward
        
    def _check_success(self, obs):
        """检查是否成功"""
        peg_pos = obs[4:7]
        target_pos = self.target_pos
        peg_to_target = np.linalg.norm(peg_pos - target_pos)
        return peg_to_target < 0.07
        
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
                
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()

class SimpleController:
    """简单的控制器"""
    
    def __init__(self):
        self.kp = 5.0
        
    def compute_control(self, current_pos, target_pos):
        """计算位置控制输出"""
        pos_error = target_pos - current_pos
        control_output = self.kp * pos_error
        return np.clip(control_output, -1.0, 1.0)

def demo_simple_control():
    """简单控制演示"""
    print("=== 简单位置控制演示 ===")
    
    # 创建环境实例
    env = SimplePegInsertEnv(render_mode='human')
    
    obs, info = env.reset()
    episode_reward = 0
    max_steps = 500
    
    print("环境初始化完成")
    
    # 初始化控制器
    controller = SimpleController()
    
    for step in range(max_steps):
        env.render()
        
        # 获取状态信息
        hand_pos = obs[0:3]
        peg_pos = obs[4:7]
        target_pos = env.target_pos
        
        # 简单的三阶段控制策略
        hand_to_peg_dist = np.linalg.norm(hand_pos - peg_pos)
        peg_to_target_dist = np.linalg.norm(peg_pos - target_pos)
        
        if hand_to_peg_dist > 0.05:
            # 第一阶段：接近peg
            target_pos_ctrl = peg_pos + np.array([0, 0, 0.1])
            grip_action = -1.0  # 张开夹爪
        elif peg_to_target_dist > 0.1:
            # 第二阶段：运输到目标上方
            target_pos_ctrl = target_pos + np.array([0, 0, 0.1])
            grip_action = 1.0   # 闭合夹爪
        else:
            # 第三阶段：向下插入
            target_pos_ctrl = target_pos - np.array([0, 0, 0.1])
            grip_action = 1.0   # 保持闭合
            
        # 计算控制输出
        control_output = controller.compute_control(hand_pos, target_pos_ctrl)
        
        # 组合动作为最终控制输入
        action = np.concatenate([control_output, [grip_action]])
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.3f}, "
                  f"PegDist={peg_to_target_dist:.3f}, "
                  f"Success={info.get('success', False)}")
            
        if info.get('success', False):
            print(f"任务成功! 总奖励: {episode_reward:.3f}")
            time.sleep(2)
            break
            
        if terminated or truncated:
            break
            
        time.sleep(0.02)
    
    env.close()
    return episode_reward, info.get('success', False)

def main():
    """主函数"""
    print("基于MuJoCo的简化Peg Insert环境演示")
    print("==================================")
    
    try:
        print("\n运行简单位置控制演示...")
        reward, success = demo_simple_control()
        print(f"\n结果: 奖励={reward:.3f}, 成功={success}")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### src_test/success_checker.py

*大小: 1.8 KB | Token: 484*

```python
# success_checker.py
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict
from params import ControlParams
from coordinate_systems import TaskCoordinateSystem

class InsertionSuccessChecker:
    """插入成功检测器"""
    
    def __init__(self, params: ControlParams, task_coord_system: TaskCoordinateSystem):
        self.params = params
        self.task_coord_system = task_coord_system
        
    def check_insertion_success(self, peg_head_pos: np.ndarray, peg_pos: np.ndarray, 
                              peg_rotation: Rotation) -> Dict:
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        peg_center_task = self.task_coord_system.world_to_task(peg_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        radial_distance = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        insertion_depth = max(0, -peg_head_task[1])
        
        is_inside_hole = radial_distance <= self.params.insertion_tolerance
        sufficient_depth = insertion_depth >= self.params.min_insertion_depth
        
        center_radial_distance = np.sqrt(peg_center_task[0]**2 + peg_center_task[2]**2)
        is_aligned = center_radial_distance <= self.params.insertion_tolerance * 2
        
        orientation_magnitude = np.linalg.norm(orientation_error)
        is_orientation_aligned = orientation_magnitude <= self.params.max_orientation_error
        
        success = is_inside_hole and sufficient_depth and is_aligned and is_orientation_aligned
        
        return {
            'success': success,
            'insertion_depth': insertion_depth,
            'radial_distance': radial_distance,
            # ... (其他返回信息可以按需添加)
        }
```

### src_test/wrapper.py

*大小: 10.3 KB | Token: 2.7K*

```python
# wrapper.py
import gymnasium as gym
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Callable

# 从本地模块导入
from params import ControlParams
from force_extractor import ForceExtractor
from controllers import Enhanced6DOFController
from coordinate_systems import TaskCoordinateSystem
from success_checker import InsertionSuccessChecker

class HybridControlWrapper(gym.Wrapper):
    """力位混合控制包装器"""
    
    def __init__(self, env, control_params: ControlParams = None):
        super().__init__(env)
        self.params = control_params or ControlParams()
        self.force_extractor = ForceExtractor(env)
        self.controller = Enhanced6DOFController(self.params)
        
        self.task_coord_system = None
        self.success_checker = None
        
        # 插入控制的阶段状态
        self.insertion_phase = "approach"  # approach -> align -> insert
        self.phase_start_time = 0
        self._get_selection_matrices_func = None

    def set_selection_matrices_func(self, func: Callable[[], Tuple[np.ndarray, np.ndarray]]):
        self._get_selection_matrices_func = func

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.controller.reset()
        self.insertion_phase = "approach"
        self.phase_start_time = 0
        
        hole_pos, hole_orientation = self._get_hole_info()
        self.task_coord_system = TaskCoordinateSystem(hole_pos, hole_orientation)
        self.success_checker = InsertionSuccessChecker(self.params, self.task_coord_system)
        
        return obs, info

    def _get_hole_info(self) -> Tuple[np.ndarray, np.ndarray]:
        hole_site_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'hole')
        hole_pos = self.env.unwrapped.data.site_xpos[hole_site_id].copy()
        box_body_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'box')
        box_quat = self.env.unwrapped.data.xquat[box_body_id].copy()
        rotation = Rotation.from_quat(box_quat)
        hole_direction = rotation.apply(np.array([0, 1, 0]))
        return hole_pos, hole_direction

    def _get_peg_state(self) -> Tuple[np.ndarray, np.ndarray, Rotation]:
        peg_center = self.env.unwrapped._get_pos_objects()
        peg_head_site_id = mujoco.mj_name2id(self.env.unwrapped.model, mujoco.mjtObj.mjOBJ_SITE, 'pegHead')
        peg_head = self.env.unwrapped.data.site_xpos[peg_head_site_id].copy()
        peg_quat = self.env.unwrapped._get_quat_objects()
        peg_rotation = Rotation.from_quat(peg_quat)
        return peg_center, peg_head, peg_rotation

    def _determine_insertion_phase(self, peg_head_pos: np.ndarray, peg_rotation: Rotation, 
                                 contact_force: np.ndarray) -> str:
        """根据当前状态确定插入阶段"""
        hole_pos, _ = self._get_hole_info()
        distance_to_hole = np.linalg.norm(peg_head_pos - hole_pos)
        
        # 转换到任务坐标系检查对齐情况
        peg_head_task = self.task_coord_system.world_to_task(peg_head_pos)
        orientation_error = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 检查径向偏差和姿态误差
        radial_error = np.sqrt(peg_head_task[0]**2 + peg_head_task[2]**2)
        orientation_magnitude = np.linalg.norm(orientation_error)
        
        # 检查是否有接触力
        has_contact = np.linalg.norm(contact_force) > 0.5
        
        if self.insertion_phase == "approach":
            # 接近阶段：距离hole较远时
            if distance_to_hole < self.params.approach_distance:
                return "align"
            return "approach"
            
        elif self.insertion_phase == "align":
            # 对齐阶段：调整位置和姿态
            if (radial_error < self.params.insertion_tolerance and 
                orientation_magnitude < self.params.max_orientation_error and
                distance_to_hole < self.params.alignment_distance):
                return "insert"
            return "align"
            
        elif self.insertion_phase == "insert":
            # 插入阶段：保持插入状态
            if has_contact or -peg_head_task[1] > 0.01:  # 已经有接触或已经插入
                return "insert"
            else:
                return "align"  # 如果失去接触，回到对齐阶段
                
        return self.insertion_phase

    def _get_target_position(self, peg_head_pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        """根据插入阶段确定目标位置"""
        hole_pos, hole_orientation = self._get_hole_info()
        
        if self.insertion_phase == "approach":
            # 接近阶段：跟随上层策略，但不要太接近hole
            target_world = hole_pos + np.array([0,10,0])
            # 融合上层策略的意图
            target_world += action[:3] * 0.02
            
        elif self.insertion_phase == "align":
            # 对齐阶段：目标位置在hole前方一小段距离
            target_world = hole_pos + hole_orientation * self.params.alignment_distance
            
        elif self.insertion_phase == "insert":
            # 插入阶段：目标位置在hole内部
            target_world = hole_pos + hole_orientation * self.params.min_insertion_depth
            
        return target_world

    def step(self, action):
        if self.task_coord_system is None:
            raise RuntimeError("环境未重置，请先调用reset()")

        peg_center, peg_head, peg_rotation = self._get_peg_state()
        contact_force, contact_torque = self.force_extractor.get_contact_forces_and_torques()
        
        # 更新插入阶段
        old_phase = self.insertion_phase
        self.insertion_phase = self._determine_insertion_phase(peg_head, peg_rotation, contact_force)
        
        if old_phase != self.insertion_phase:
            print(f"插入阶段切换: {old_phase} -> {self.insertion_phase}")
        
        # 将接触力/力矩转换到任务坐标系
        contact_force_task = self.task_coord_system.world_force_to_task(contact_force)
        contact_torque_task = self.task_coord_system.world_force_to_task(contact_torque)

        # 获取当前状态在任务坐标系下的表示
        peg_head_task = self.task_coord_system.world_to_task(peg_head)
        orientation_error_task = self.task_coord_system.get_orientation_error(peg_rotation)
        
        # 根据阶段确定目标位置
        target_pos_world = self._get_target_position(peg_head, action)
        target_pos_task = self.task_coord_system.world_to_task(target_pos_world)
        
        # 目标姿态始终是对齐到hole
        target_orientation_error_task = np.zeros(3)
        
        # 根据阶段设置目标力/力矩
        if self.insertion_phase == "insert":
            # 插入阶段：Y方向施加插入力，其他方向保持0力
            target_force_task = np.array([0, 5.0, 0])  # Y方向插入力
            target_torque_task = np.zeros(3)
        else:
            # 其他阶段：所有力/力矩都为0
            target_force_task = np.zeros(3)
            target_torque_task = np.zeros(3)

        # 根据阶段调整选择矩阵
        if self.insertion_phase == "insert":
            # 插入阶段：Y方向改为力控制
            selection_pos = np.diag([1, 0, 1])  # X,Z位置控制，Y力控制
            selection_rot = np.eye(3)  # 姿态仍然位置控制
        else:
            # 其他阶段：纯位置控制
            selection_pos, selection_rot = self._get_selection_matrices_func()

        # 计算控制器输出
        dt = self.env.unwrapped.dt
        pos_out, rot_out = self.controller.compute_control(
            peg_head_task, target_pos_task,
            orientation_error_task, target_orientation_error_task,
            contact_force_task, target_force_task,
            contact_torque_task, target_torque_task,
            selection_pos, selection_rot, dt
        )

        # 将控制器输出转换回世界坐标系
        pos_correction_world = self.task_coord_system.task_force_to_world(pos_out)
        rot_correction_world = self.task_coord_system.task_force_to_world(rot_out)

        # 根据阶段调整控制器权重
        if self.insertion_phase == "approach":
            # 接近阶段：主要依赖上层策略
            weight_upper = 0.8
            weight_lower = 0.2
        elif self.insertion_phase == "align":
            # 对齐阶段：增加底层控制器权重
            weight_upper = 0.3
            weight_lower = 0.7
        elif self.insertion_phase == "insert":
            # 插入阶段：主要依赖底层控制器
            weight_upper = 0.1
            weight_lower = 0.9
        
        # 限制控制器输出幅度
        pos_correction_world = np.clip(pos_correction_world, -0.2, 0.2)
        rot_correction_world = np.clip(rot_correction_world, -0.2, 0.2)
        
        # 融合上层策略和底层控制器
        modified_action = action.copy()
        modified_action[:3] = (weight_upper * action[:3] + 
                              weight_lower * pos_correction_world)
        
        # 限制最终动作幅度
        modified_action[:3] = np.clip(modified_action[:3], -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        # 更新info
        final_peg_center, final_peg_head, final_peg_rotation = self._get_peg_state()
        insertion_status = self.success_checker.check_insertion_success(
            final_peg_head, final_peg_center, final_peg_rotation)
        info.update(insertion_status)
        
        # 添加阶段信息到info
        info['insertion_phase'] = self.insertion_phase
        info['target_pos_world'] = target_pos_world
        info['distance_to_target'] = np.linalg.norm(final_peg_head - target_pos_world)
        
        # 调试信息
        if hasattr(self, '_debug_step_count'):
            self._debug_step_count += 1
        else:
            self._debug_step_count = 1
            
        if self._debug_step_count % 100 == 0:
            hole_pos, _ = self._get_hole_info()
            print(f"Debug - Phase: {self.insertion_phase}, "
                  f"Distance to hole: {np.linalg.norm(final_peg_head - hole_pos):.3f}, "
                  f"Contact force: {np.linalg.norm(contact_force):.3f}")
        
        return obs, reward, terminated, truncated, info
```
