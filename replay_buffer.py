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
    masks: jnp.ndarray


class ReplayBuffer(eqx.Module):
    buffer_size: int

    observations: Float[Array, "bs obs_size"]
    next_observations: Float[Array, "bs obs_size"]
    actions: Float[Array, "bs act_size"] | Int[Array, "bs act_size"]
    rewards: Float[Array, "bs 1"]
    dones: Bool[Array, "bs 1"]
    masks: Bool[Array, "bs n_nets"]

    pos: Int[Array, ""]
    full: Bool[Array, ""]

    # --------------------------------------------------------------------- #
    # construction
    # --------------------------------------------------------------------- #
    @staticmethod
    def make(buffer_size: int, example_obs: Any, example_act: Any, mask_size: int):
        observations = jnp.zeros((buffer_size, *example_obs.shape), example_obs.dtype)
        next_observations = jnp.zeros((buffer_size, *example_obs.shape), example_obs.dtype)
        actions = jnp.zeros((buffer_size, *example_act.shape), example_act.dtype)
        rewards = jnp.zeros((buffer_size, 1), dtype=jnp.float32)
        dones = jnp.zeros((buffer_size, 1), dtype=jnp.bool_)  # ← jnp.bool_ (no deprecation)
        masks = jnp.zeros((buffer_size, mask_size), dtype=jnp.bool_)

        pos = jnp.zeros((), dtype=jnp.int32)
        full = jnp.zeros((), dtype=jnp.bool_)

        return ReplayBuffer(
            buffer_size, observations, next_observations, actions, rewards, dones, masks, pos, full
        )

    def add(
        self,
        obs: Float[Array, "k obs_size"],
        next_obs: Float[Array, "k obs_size"],
        action: Float[Array, "k act_size"] | Int[Array, "k"],  # may arrive as (k,)
        reward: Float[Array, "k"],  # may arrive as (k,)
        done: Bool[Array, "k"],  # may arrive as (k,)
        mask: Bool[Array, "k n_nets"],
    ):
        k = obs.shape[0]  # batch size

        # ------------------------------------------------------------------ #
        # 1. Make sure every tensor has the exact trailing dims expected by
        #    the buffer.  We reshape instead of broadcast so shape
        #    mismatches fail early and clearly.
        # ------------------------------------------------------------------ #
        reward = reward.reshape(k, 1)
        done = done.reshape(k, 1)

        # `self.actions` may be (bs,) for discrete envs, or (bs, act_size)
        # for continuous / one-hot.  We reshape whatever comes in to match.
        act_tail_shape = self.actions.shape[1:]  # ()  or (act_size,)
        action = action.reshape(k, *act_tail_shape)

        # ------------------------------------------------------------------ #
        # 2. Indices that handle wrap-around
        # ------------------------------------------------------------------ #
        idx = jnp.mod(self.pos + jnp.arange(k), self.buffer_size)  # (k,)

        # ------------------------------------------------------------------ #
        # 3. Scatter-update
        # ------------------------------------------------------------------ #
        new_observations = self.observations.at[idx].set(obs)
        new_next_observations = self.next_observations.at[idx].set(next_obs)
        new_actions = self.actions.at[idx].set(action)
        new_rewards = self.rewards.at[idx].set(reward)
        new_dones = self.dones.at[idx].set(done)
        new_masks = self.masks.at[idx].set(mask)

        # ------------------------------------------------------------------ #
        # 4. Advance cursor and (maybe) flip `full`
        # ------------------------------------------------------------------ #
        new_pos = jnp.mod(self.pos + k, self.buffer_size)
        new_full = jnp.logical_or(self.full, (self.pos + k) >= self.buffer_size)

        return ReplayBuffer(
            self.buffer_size,
            new_observations,
            new_next_observations,
            new_actions,
            new_rewards,
            new_dones,
            new_masks,
            new_pos,
            new_full,
        )

    def sample(self, key, n) -> ReplayBufferSample:
        upper_bound = jax.lax.cond(self.full, lambda: self.buffer_size, lambda: self.pos)

        indices = jr.randint(key, (n,), 0, upper_bound)

        return ReplayBufferSample(
            observations=self.observations[indices],
            next_observations=self.next_observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices],
            masks=self.masks[indices],
        )
