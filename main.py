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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert Hydra config to dynamically created dataclass
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
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n)
    q_network.apply = jax.jit(q_network.apply)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adamw(learning_rate=args.learning_rate),  # optimizer of choice: adamw
    )

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(
            q_state.target_params, next_observations
        )  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[
                jnp.arange(args.batch_size), actions.squeeze()
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    last_mean_rs = 0  # the average reward, reporting purposes

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    for global_step in (progress_bar := tqdm(range(args.total_timesteps))):
        wandb.log({})  # commit to wandb
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # record rewards for reporting purposes
        rs, ls = [], []
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    rs.append(info["episode"]["r"])
                    ls.append(info["episode"]["l"])

            current_mean = jnp.array(rs).mean()
            if last_mean_rs == 0:
                last_mean_rs = current_mean
            last_mean_rs = current_mean * 0.05 + 0.95 * last_mean_rs

            if args.track:  # post updates into wandb
                wandb.log(
                    {
                        "charts/episodic_return": jnp.array(rs).mean(),
                        "charts/episodic_length": jnp.array(ls).mean(),
                    },
                    commit=False,
                )

        if global_step % 200 == 0:  # print pretty updates into tty
            progress_bar.set_description_str(f"Reward: {float(last_mean_rs):.2f}")

        # save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # CRUCIAL step easy to overlook, moving to the new observations
        obs = next_obs

        # training process
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations,
                    data.actions,
                    data.next_observations,
                    data.rewards.flatten(),
                    data.dones.flatten(),
                )

                if global_step % 1000 == 0:  # more reporting
                    progress_bar.set_postfix(
                        loss=f"{float(jax.device_get(loss).squeeze()):.3f}",
                        epsilon=f"{epsilon:.2f}",
                    )

                    if args.track:
                        wandb.log(
                            {
                                "losses/td_loss": jax.device_get(loss),
                                "losses/q_values": jax.device_get(old_val).mean(),
                            },
                            commit=False,
                        )

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(
                        q_state.params, q_state.target_params, args.tau
                    )
                )

    if args.save_model:  # even more reporting
        model_path = f"runs/{run_name}/{args.exp_name}.nanodqn_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        progress_bar.write(f"Model saved to {model_path}")

        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        wandb.log({"eval/episodic_return_mean": episodic_returns.mean()})

    envs.close()


if __name__ == "__main__":
    main()
