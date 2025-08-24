from __future__ import annotations

from typing import Any, NamedTuple, Tuple, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias, TypedDict


class Task(NamedTuple):
    env_name: str
    data: bytes  # Contains env parameters like random_init and *a* goal

RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"

XYZ: TypeAlias = "Tuple[float, float, float]"
"""A 3D coordinate."""

class EnvironmentStateDict(TypedDict):
    state: dict[str, Any]
    mjb: str
    mocap: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
