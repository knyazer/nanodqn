import os
import random
import time
from dataclasses import dataclass, make_dataclass, field
from typing import Any

import flax
import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from replay_buffer import ReplayBuffer

args = None  # placeholder


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                step_trigger=lambda s: s % 50000 == 49999,  # every 50k steps
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def create_config_dataclass(cfg: DictConfig) -> Any:
    fields = []
    for key, value in cfg.items():
        field_type = type(value)
        if value is None:
            field_type = Any
        fields.append((key, field_type, field(default=value)))

    return make_dataclass("Config", fields)()


def update(q_network, q_state, data):
    q_next_target = q_network.apply(
        q_state.target_params, data.next_observations
    )  # (batch_size, num_actions)
    q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
    next_q_value = data.rewards + (1 - data.dones) * args.gamma * q_next_target

    def mse_loss(params):
        q_pred = q_network.apply(params, data.observations)  # (batch_size, num_actions)
        q_pred = q_pred[jnp.arange(args.batch_size), data.actions.squeeze()]  # (batch_size,)
        return ((q_pred - next_q_value) ** 2).mean(), q_pred

    (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
    q_state = q_state.apply_gradients(grads=grads)
    return q_state, loss_value


class DQN:
    def __init__(self, key, envs, obs):
        self.q_network = QNetwork(action_dim=envs.single_action_space.n)
        breakpoint()
        self._last_mean_rs = 0
        self.q_network.apply = jax.jit(self.q_network.apply)
        self.q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=self.q_network.init(key, obs),
            target_params=self.q_network.init(key, obs),
            tx=optax.adamw(learning_rate=args.learning_rate),
        )

        self.rb = None

    def step(self, global_step):
        if global_step % args.train_frequency == 0:
            data = self.rb.sample(args.batch_size)
            self.q_state, loss = update(self.q_network, self.q_state, data)

        if global_step % args.target_network_frequency == 0:
            self.q_state = self.q_state.replace(
                target_params=optax.incremental_update(
                    self.q_state.params, self.q_state.target_params, args.tau
                )
            )

    def action(self, global_step, envs, obs):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = jnp.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = self.q_network.apply(self.q_state.params, obs)
            actions = q_values.argmax(axis=-1)
        return actions

    def report(self, global_step, infos, *, wandb=None, tqdm_bar=None):
        rs, ls = [], []
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    rs.append(info["episode"]["r"])
                    ls.append(info["episode"]["l"])

            current_mean = jnp.array(rs).mean()
            if self._last_mean_rs == 0:
                self._last_mean_rs = current_mean
            self._last_mean_rs = current_mean * 0.05 + 0.95 * self._last_mean_rs

            if args.track:  # post updates into wandb
                if wandb is None:
                    raise ValueError(
                        "Trying to track, while haven't passed wandb= kwarg to report!"
                    )
                wandb.log(
                    {
                        "charts/episodic_return": jnp.array(rs).mean(),
                        "charts/episodic_length": jnp.array(ls).mean(),
                    },
                    commit=False,
                )

        if global_step % 200 == 0 and tqdm_bar is not None:
            tqdm_bar.set_description_str(f"Reward: {float(self._last_mean_rs):.2f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to dynamically created dataclass
    global args
    args = create_config_dataclass(cfg)

    run_name = f"{args.env_id}_{hex(int(time.time()) % 65536)}"
    if args.track:
        import wandb

        # Store full config in wandb for reproducibility
        wandb.init(
            project=args.wandb_project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key, q_key = jr.split(jr.key(args.seed))

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )
    obs, _ = envs.reset(seed=args.seed)

    dqn = DQN(jr.key(args.seed + 1234), envs, obs)

    for global_step in (progress_bar := tqdm(range(args.total_timesteps))):
        wandb.log({})  # commit to wandb

        actions = np.array(dqn.action(global_step, envs, obs))
        print(actions)
        breakpoint()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        dqn.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        dqn.report(global_step, infos, wandb=wandb, tqdm_bar=progress_bar)

        obs = next_obs
        if global_step > args.learning_starts:
            dqn.step(global_step)

    if args.save_model:
        raise NotImplementedError()

    envs.close()


if __name__ == "__main__":
    main()
