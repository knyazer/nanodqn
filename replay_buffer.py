import numpy as np
from typing import Dict, NamedTuple, Any, Tuple
from gymnasium import spaces


class ReplayBufferSample(NamedTuple):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.pos = 0
        self.full = False

        obs_shape = self._get_shape(observation_space)
        action_shape = self._get_shape(action_space)

        self.observations = np.zeros(
            (buffer_size, *obs_shape), dtype=observation_space.dtype
        )
        self.next_observations = np.zeros(
            (buffer_size, *obs_shape), dtype=observation_space.dtype
        )
        self.actions = np.zeros((buffer_size, *action_shape), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def _get_shape(self, space: spaces.Space) -> Tuple:
        if isinstance(space, spaces.Box):
            return space.shape
        elif isinstance(space, spaces.Discrete):
            return (1,)
        elif isinstance(space, spaces.MultiDiscrete):
            return (int(len(space.nvec)),)
        elif isinstance(space, spaces.MultiBinary):
            return (int(space.n),)
        else:
            raise NotImplementedError(f"Unsupported space type: {type(space)}")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: Dict[str, Any],
    ) -> None:
        # Handle vectorized environment (multiple parallel environments)
        for i in range(len(obs)):
            self._add_single(obs[i], next_obs[i], action[i], reward[i], done[i])

    def _add_single(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        # Store transition in the buffer
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()

        if isinstance(self.action_space, spaces.Discrete):
            self.actions[self.pos] = np.array([action]).copy()
        else:
            self.actions[self.pos] = np.array(action).copy()

        self.rewards[self.pos] = np.array([reward]).copy()
        self.dones[self.pos] = np.array([done]).copy()

        # Update buffer position
        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSample:
        # Calculate the indices to sample
        upper_bound = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, upper_bound, size=batch_size)

        return ReplayBufferSample(
            observations=self.observations[indices],
            next_observations=self.next_observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
        )
