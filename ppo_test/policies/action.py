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
