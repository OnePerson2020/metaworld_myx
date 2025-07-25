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
