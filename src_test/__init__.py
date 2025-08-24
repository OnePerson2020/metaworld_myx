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
from ppo_test.sawyer_peg_insertion_side_v4 import SawyerPegInsertionSideEnvV4

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
    env_cls: type[SawyerPegInsertionSideEnvV4],
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