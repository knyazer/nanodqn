from typing import Dict, NamedTuple, Any, Tuple
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, Bool, Float, Int


class ReplayBufferSample(eqx.Module):
    observations: jnp.ndarray
    next_observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


class ReplayBuffer(eqx.Module):
    buffer_size: int

    observations: Float[Array, "bs obs_size"]
    next_observations: Float[Array, "bs obs_size"]
    actions: Float[Array, "bs act_size"] | Int[Array, "bs act_size"]
    rewards: Float[Array, "bs 1"]
    dones: Bool[Array, "bs 1"]

    pos: Int[Array, ""]
    full: Bool[Array, ""]

    def __init__(
        self,
        buffer_size: int,
        example_obs: Any,
        example_act: Any,
    ):
        self.buffer_size = buffer_size

        self.pos = jnp.zeros((), dtype=jnp.int32)  # noqa
        self.full = jnp.zeros((), dtype=jnp.bool)  # noqa

        self.observations = jnp.zeros((buffer_size, *example_obs.shape), example_obs.dtype)
        self.next_observations = jnp.zeros((buffer_size, *example_obs.shape), example_obs.dtype)
        self.actions = jnp.zeros((buffer_size, *example_act.shape), example_act.dtype)
        self.rewards = jnp.zeros((buffer_size, 1), dtype=jnp.float32)
        self.dones = jnp.zeros((buffer_size, 1), dtype=jnp.bool)

    def add(
        self,
        obs: Float[Array, "obs_size"],
        next_obs: Float[Array, "obs_size"],
        action: Float[Array, "act_size"],
        reward: Float[Array, ""],
        done: Bool[Array, ""],
    ):
        self = eqx.tree_at(lambda s: s.observations, self, self.observations.at[self.pos].set(obs))
        self = eqx.tree_at(
            lambda s: s.next_observations, self, self.next_observations.at[self.pos].set(next_obs)
        )
        self = eqx.tree_at(lambda s: s.actions, self, self.actions.at[self.pos].set(action))
        self = eqx.tree_at(lambda s: s.rewards, self, self.rewards.at[self.pos].set(reward))
        self = eqx.tree_at(lambda s: s.dones, self, self.dones.at[self.pos].set(done))

        # Update buffer position
        self = eqx.tree_at(
            lambda s: s.pos, self, jnp.min(jnp.array([self.pos + 1, self.buffer_size]))
        )

        # Check if full
        is_full = jnp.logical_or(self.full, self.pos >= self.buffer_size)
        self = eqx.tree_at(lambda s: s.full, self, is_full)
        return self

    def sample(self, key) -> ReplayBufferSample:
        # Calculate the indices to sample
        upper_bound = jax.lax.cond(self.full, lambda: self.buffer_size, lambda: self.pos)
        indices = jr.randint(key, (), 0, upper_bound)

        return ReplayBufferSample(
            observations=self.observations[indices],
            next_observations=self.next_observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
        )

    def filled(self):
        return self.full
