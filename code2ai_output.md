# 项目导出

**文件数量**: 10  
**总大小**: 97.0 KB  
**Token 数量**: 26.7K  
**生成时间**: 2025/8/14 09:28:02

## 文件结构

```
📁 .
  📁 ppo_test
    📁 policies
      📄 __init__.py
    📁 utils
      📄 reward_utils.py
    📄 __init__.py
    📄 asset_path_utils.py
    📄 env_dict.py
    📄 sawyer_peg_insertion_side_v3.py
    📄 sawyer_xyz_env.py
    📄 types.py
    📄 wrappers.py
  📄 train_rl.py
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

### ppo_test/sawyer_peg_insertion_side_v3.py

*大小: 15.2 KB | Token: 3.9K*

```python
# metaworld/sawyer_peg_insertion_side_v3.py

from __future__ import annotations

from typing import Any, Tuple

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from ppo_test.asset_path_utils import full_V3_path_for
from ppo_test.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from ppo_test.utils import reward_utils

box_raw = 17
quat_box = Rotation.from_euler('xyz', [0, 0, 90+box_raw], degrees=True).as_quat()[[3,0, 1, 2]]

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

        self.hand_init_pos = np.array([0, 0.6, 0.2])
        self.hand_init_quat = Rotation.from_euler('xyz', [0,90,0], degrees=True).as_quat()[[1, 2, 3, 0]]
        
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

    @property
    def model_name(self) -> str:
        return full_V3_path_for("sawyer_peg_insertion_side.xml")
    
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
        self._goal_pos = pos_box + Rotation.from_euler('xyz', [0,0,box_raw], degrees=True).apply(np.array([0.03, 0.0, 0.13]))
        
        self.model.site("goal").pos = self._goal_pos

        self.objHeight = self.get_body_com("peg").copy()[2]
                
        return self._get_obs()

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        obj = obs[14:17]
        assert self._goal_pos is not None and self.obj_init_pos is not None        
        tcp_open: float = obs[7] 
        tcp = self.tcp_center
                
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        
        # 使用优化后的奖励计算
        reward, stage_rewards = self.compute_reward_test(action, obj)
        
        # 获取插入信息
        insertion_info = self.get_insertion_info()
        
        assert self.obj_init_pos is not None
        grasp_success = float(
            tcp_to_obj < 0.02
            and (tcp_open > 0)
            and (obj[2] - 0.01 > self.obj_init_pos[2])
        )
        
        # 成功条件：插入深度达到目标
        success = float(insertion_info["insertion_depth"] >= 0.1)  # 5cm插入深度
        # success =  float(stage_rewards["approach"] == 1)
                    
        info = {
            "success": success,
            "grasp_success": grasp_success,
            "stage_rewards": stage_rewards,
            "insertion_depth": insertion_info["insertion_depth"],
            "unscaled_reward": reward,
        }

        return reward, info

    def compute_reward_test(
        self, action: npt.NDArray[Any], obj: npt.NDArray[Any]
    ) -> tuple[float, dict[str, float]]:
        """
        优化的分阶段奖励函数
        阶段1: 接近peg
        阶段2: 抓取peg
        阶段3: 对准hole
        阶段4: 插入
        """
        
        tcp = self.tcp_center
        obs = self._get_obs()
        tcp_opened = obs[7]
        
        # 获取关键位置
        obj_head = self._get_site_pos("pegHead")
        insertion_info = self.get_insertion_info()
        hole_pos = insertion_info["hole_pos"]
        hole_orientation = insertion_info["hole_orientation"]
        insertion_depth = insertion_info["insertion_depth"]

        # 初始化任务阶段状态（如果不存在）
        if not hasattr(self, 'task_phase'):
            self.task_phase = 'approach'
            self.max_alignment_achieved = 0.0
            self.insertion_started = False
        # ========== 阶段1: 接近Peg ==========
        tcp_to_obj = float(np.linalg.norm(obj - tcp))
        approach_margin = float(np.linalg.norm(self.obj_init_pos - self.hand_init_pos))
        
        approach_reward = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.02),  # 目标是接近到2cm以内
            margin=approach_margin,
            sigmoid="long_tail"
        )
        
        # ========== 阶段2: 抓取Peg ==========
        # 使用现有的gripper caging reward函数
        grasp_reward = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.0075,
            pad_success_thresh=0.03,
            xz_thresh=0.005,
            high_density=True,
        )
        
        # 检查是否已经抓取成功
        obj_lifted = obj[2] - self.obj_init_pos[2] > 0.01  # 物体抬起超过1cm
        if tcp_to_obj < 0.08 and tcp_opened > 0 and obj_lifted:
            grasp_reward = 1.0
            if self.task_phase == 'approach':
                self.task_phase = 'grasp'

        # ========== 阶段3: 对准Hole ==========
        # 计算peg head到hole入口的距离
        head_to_hole = obj_head - hole_pos
        
        # 横向对准（垂直于hole方向的距离）
        lateral_offset = head_to_hole - np.dot(head_to_hole, hole_orientation) * hole_orientation
        lateral_distance = np.linalg.norm(lateral_offset)
        
        # 纵向对准（沿hole方向，但在hole前方）
        longitudinal_distance = np.dot(head_to_hole, hole_orientation)
        
        # 计算当前对准质量
        current_alignment = 0.0
        if grasp_reward > 0.8:  # 只有抓取成功后才计算对准
            # 横向对准奖励
            lateral_alignment = reward_utils.tolerance(
                lateral_distance,
                bounds=(0, 0.005),  # 横向误差小于5mm
                margin=0.1,
                sigmoid="long_tail"
            )
            
            # 纵向位置奖励（在hole前方0-3cm的位置最佳）
            longitudinal_alignment = reward_utils.tolerance(
                abs(longitudinal_distance),
                bounds=(0, 0.03),  
                margin=0.1,
                sigmoid="long_tail"
            )
            
            current_alignment = reward_utils.hamacher_product(
                lateral_alignment, 
                longitudinal_alignment
            )
            
            # 更新最大对准值
            self.max_alignment_achieved = max(self.max_alignment_achieved, current_alignment)
            
            # 状态转换：进入对准阶段
            if self.task_phase == 'grasp' and current_alignment > 0.5:
                self.task_phase = 'alignment'
        
        # 对准奖励（根据阶段决定）
        if not self.insertion_started:
            # 未开始插入时，使用当前对准值
            alignment_reward = current_alignment
        else:
            # 已开始插入，使用历史最大值，避免插入时的下降影响
            alignment_reward = self.max_alignment_achieved
        
        # ========== 阶段4: 插入 ==========
        insertion_reward = 0.0
        
        # 检查是否可以开始插入
        if self.max_alignment_achieved > 0.7 or self.insertion_started:
            # 一旦开始插入，就保持插入状态
            if insertion_depth > 0.001:  # 检测到开始插入（1mm以上）
                self.insertion_started = True
                self.task_phase = 'insertion'
            
            if self.insertion_started:
                # 插入深度奖励
                insertion_reward = reward_utils.tolerance(
                    insertion_depth,
                    bounds=(0.10, 0.10),  # 目标插入深度5-10cm
                    margin=0.1,
                    sigmoid="gaussian"
                )
                
                # 插入过程中的对准保持奖励（独立计算，不影响alignment_reward）
                insertion_alignment_bonus = 0.0
                if lateral_distance < 0.01:  # 插入时的对准容差可以更宽松
                    insertion_alignment_bonus = 0.2 * (1.0 - lateral_distance / 0.01)
                insertion_reward = min(1.0, insertion_reward + insertion_alignment_bonus)
        
        stage_weights = {"approach": 1, "grasp": 0, "alignment": 0, "insertion": 0}
            
        
        # 计算加权总奖励
        total_reward = (
            stage_weights["approach"] * approach_reward +
            stage_weights["grasp"] * grasp_reward +
            stage_weights["alignment"] * alignment_reward +
            stage_weights["insertion"] * insertion_reward
        ) / sum(stage_weights.values())
        
        # 成功奖励
        if insertion_depth >= 0.05:
            total_reward = 10.0
        
        # 阶段奖励字典（用于调试和监控）
        stage_rewards = {
            "approach": approach_reward,
            "grasp": grasp_reward,
            "alignment": alignment_reward,
            "insertion": insertion_reward,
            "lateral_distance": lateral_distance,
            "insertion_depth": insertion_depth,
            "task_phase": self.task_phase,
            "max_alignment": self.max_alignment_achieved,
            "insertion_started": self.insertion_started
        }

        
        labels = ["App", "Grasp", "Align", "Insert", "Lat", "Long", "Depth"]
        values = [approach_reward, grasp_reward, alignment_reward, insertion_reward, lateral_distance, longitudinal_distance, insertion_depth]
        print(" ".join(f"{v:6.3f}" for v in values))
        print(" ".join(f"{l:^6}" for l in labels))  # 可选：打印标签行

        return total_reward, stage_rewards

    def get_hole_info(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """获取hole的位置和朝向信息。
        朝向由从 'hole' 站点指向 'goal' 站点的单位向量表示。
        """
        hole_pos = self._get_site_pos("hole")
        goal_pos = self._get_site_pos("goal")

        # 计算从 hole 指向 goal 的向量
        orientation_vec = goal_pos - hole_pos
        
        hole_orientation = orientation_vec / np.linalg.norm(orientation_vec)
            
        return hole_pos, hole_orientation

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
```

### ppo_test/sawyer_xyz_env.py

*大小: 31.7 KB | Token: 8.9K*

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
        
        # self.action_space = Box(  # type: ignore
        #     np.array([-1, -1, -1, -1, -1, -1, -1, -1]),
        #     np.array([+1, +1, +1, +1, +1, +1, +1, +1]),
        #     dtype=np.float32,
        # )

        self.action_space = Box(  # type: ignore
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float32,
        )

        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self.hand_init_quat: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._goal_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
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
        
        # r_increment = Rotation.from_quat(action[3:7])
        
        # current_mocap = self.data.mocap_quat[0]
        # new_mocap_r = r_increment * Rotation.from_quat(current_mocap[[1,2,3,0]])
        # new_mocap_quat = new_mocap_r.as_quat()[[3,0,1,2]]
        # self.data.mocap_quat[0] = new_mocap_quat

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
        assert self._goal_pos is not None
        return [("goal", self._goal_pos)]

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
        assert isinstance(self._goal_pos, np.ndarray)
        assert self._goal_pos.ndim == 1
        return self._goal_pos

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
        assert len(action) == 4, f"Actions should be size 8, got {len(action)}"
        self.set_xyz_action(action[:3]) # Pass position and rotation actions
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

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

### train_rl.py

*大小: 2.8 KB | Token: 586*

```python
import os
import ppo_test
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. 定义常量和配置 ---
ENV_NAME = 'peg-insert-side-v3'

# 日志和模型保存路径
LOG_DIR = "rl_logs"
MODEL_SAVE_DIR = "rl_models"
BEST_MODEL_SAVE_PATH = "rl_models" # 对应 evaluate_rl.py 中的路径
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

# 训练总步数 (这是一个关键的超参数，需要根据训练效果调整)
# 对于第一阶段，可以先从一个较小的值开始，如 20万 或 50万
TOTAL_TIMESTEPS = 500_000

# --- 2. 创建训练和评估环境 ---
print("正在创建环境...")
# 训练环境，不需要渲染以加快速度
train_env = ppo_test.make_mt_envs(name=ENV_NAME)

# 评估环境，用于在训练过程中周期性地测试模型性能
# EvalCallback 将使用这个环境来确定哪个模型是 "best_model"
eval_env = ppo_test.make_mt_envs(name=ENV_NAME)
print("环境创建完成。")

# --- 3. 设置评估回调 ---
# EvalCallback 是一个强大的工具，它会:
# 1. 定期 (eval_freq) 在评估环境 (eval_env) 上运行模型。
# 2. 评估 N 次 (n_eval_episodes)。
# 3. 如果当前模型效果是历史最佳，则将其保存到 best_model_save_path。
# 4. deterministic=True 表示使用确定性策略进行评估，更准确地反映模型学习到的策略。
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_SAVE_PATH,
    log_path=LOG_DIR,
    eval_freq=10000, # 每 10000 步评估一次
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

# --- 4. 初始化或加载模型 ---
# 这里我们总是从头开始创建一个新模型，因为这是训练脚本
# "MlpPolicy" 适用于基于向量/数值的观察空间 (state-based observation)
print("正在初始化 PPO 模型...")
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,  # 打印训练过程中的信息
    tensorboard_log=LOG_DIR, # 启用 TensorBoard 日志
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.0,
    clip_range=0.2
)
print("模型初始化完成。")

# --- 5. 开始训练 ---
print("\n--- 开始训练 ---")
# model.learn 会执行整个训练过程
# callback 参数让我们能在训练中途执行特定操作，如保存最佳模型
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback
)
print("\n--- 训练完成 ---")

# --- 6. 保存最终模型 ---
final_model_path = os.path.join(MODEL_SAVE_DIR, f"ppo_{ENV_NAME}_final.zip")
model.save(final_model_path)
print(f"最终模型已保存至: {final_model_path}")
print(f"最佳模型已保存在 '{BEST_MODEL_SAVE_PATH}/' 目录下，可用于 evaluate_rl.py。")

# --- 7. 关闭环境 ---
train_env.close()
eval_env.close()
```
