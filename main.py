import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from tqdm.auto import tqdm


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases; on by default"""
    wandb_project_name: str = "nanodqn"
    """the wandb's project name"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            # We'll manually control when videos are recorded
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: False,  # Disable automatic recording
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
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


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}_{hex(int(time.time()) % 65536)}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)  # for python default library
    np.random.seed(args.seed)  # for numpy
    key = jr.key(args.seed)  # jax: sets up the key
    key, q_key = jr.split(key)  # splits the key in 2

    # env setup
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
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adamw(learning_rate=args.learning_rate),  # optimizer of choice: adamw
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, 1)
    )

    # this is the only dependency on stable baselines; i think we should find an alternative
    rb = sb3.common.buffers.ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
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
                jnp.arange(q_pred.shape[0]), actions.squeeze()
            ]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            q_state.params
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()  # start timer for SPS (steps-per-second) computation
    last_mean_rs = 0  # the average reward, reporting purposes

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    progress_bar = tqdm(range(args.total_timesteps))

    # For recording videos at 10% intervals
    video_interval = args.total_timesteps // 10
    video_checkpoints = set(
        i * video_interval for i in range(1, 11)
    )  # 10%, 20%, ..., 100%
    last_video_path = None

    for global_step in progress_bar:
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
            last_mean_rs = current_mean * 0.01 + 0.99 * last_mean_rs

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

        # Record and upload video at 10% intervals if video capture is enabled
        if args.capture_video and args.track and global_step in video_checkpoints:
            # Trigger video recording for the next episode
            progress_bar.write(
                f"Recording video at {global_step/args.total_timesteps*100:.0f}% of training"
            )
            env_vec = envs.envs[
                0
            ]  # Get the first environment (we're only recording one)

            env_vec.start_video_recorder()

            # Run one episode to record
            episode_done = False
            episode_obs, _ = env_vec.reset()
            while not episode_done:
                episode_action = (
                    envs.single_action_space.sample()
                    if random.random() < 0.1
                    else q_network.apply(q_state.params, episode_obs[None]).argmax(
                        axis=-1
                    )[0]
                )
                episode_obs, _, episode_term, episode_trunc, _ = env_vec.step(
                    jax.device_get(episode_action)
                )
                episode_done = episode_term or episode_trunc
            env_vec.reset()

            # Stop recording and get the video path
            video_path = env_vec.video_recorder.path

            # Wait for the file to be fully written
            time.sleep(0.3)

            # Upload the video to wandb
            if os.path.exists(video_path):
                progress_bar.write(
                    f"Uploaded video at {global_step/args.total_timesteps*100:.0f}% of training"
                )
            else:
                progress_bar.write(
                    f"Failed to save the video at {global_step/args.total_timesteps*100:.0f}% of training"
                )

        # training process
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 1000 == 0:  # more reporting
                    sps = int(global_step / (time.time() - start_time))
                    progress_bar.set_postfix(
                        SPS=sps,
                        loss=f"{float(jax.device_get(loss).squeeze()):.3f}",
                        epsilon=f"{epsilon:.2f}",
                    )

                    if args.track:
                        wandb.log(
                            {
                                "losses/td_loss": jax.device_get(loss),
                                "losses/q_values": jax.device_get(old_val).mean(),
                                "charts/SPS": sps,
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
