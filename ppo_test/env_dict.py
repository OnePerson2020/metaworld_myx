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