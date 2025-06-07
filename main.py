import equinox as eqx
import numpy as np
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
import gymnax
from replay_buffer import ReplayBuffer
from tqdm import tqdm as tqdm
import optax
import wandb
import os
import pandas as pd
import yaml
import dataclasses
import hashlib
import json
from pathlib import Path
from models import Model, ModelWithPrior, DQN, Bootstrapped, EpsilonGreedy, AMCDQN, filtered_cond

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

max_trainings_in_parallel = 25


class Config(eqx.Module):
    batch_size: int = 128
    lr: float = 1e-4
    num_envs: int = 16
    ensemble_size: int = 4
    prior_scale: float = 3
    env_name: str = "DeepSea-bsuite"
    kind: str = "dqn"
    hardness: int = 12
    randomize_actions: bool = True
    num_episodes: int = 10_000
    dqn_episodes_to_reset: int = 0

    def autoseed(self):
        cfg_dict = dataclasses.asdict(self)
        cfg_str = json.dumps(cfg_dict, sort_keys=True)
        h = hashlib.sha256(cfg_str.encode()).hexdigest()
        return int(h[:6], 16)

    def __str__(self):
        return f"{self.kind}({self.ensemble_size})_{self.env_name}({self.hardness})_beta{self.prior_scale}"

    def short_str(self):
        return str(self)


@eqx.filter_jit
def main(key=None, cfg=Config(), debug=False):
    if debug:
        jax.debug.print(
            "Main is running with: kind={}, ensemble_size={}, hardness={}",
            cfg.kind,
            cfg.ensemble_size,
            cfg.hardness,
        )
    key, model_key, target_model_key, reset_key, loop_key, ikey = jr.split(key, 6)

    env, env_params = gymnax.make(
        cfg.env_name, size=cfg.hardness, randomize_actions=cfg.randomize_actions
    )
    obs, state = jax.vmap(lambda k: env.reset(k, env_params))(jr.split(reset_key, cfg.num_envs))
    action_space = env.action_space(env_params)  # type: ignore

    assert "n" in action_space.__dict__, (
        "The environment is not discrete, or maybe incorrectly initialized"
    )
    act_size = action_space.__dict__.get("n", 2)
    single_obs = obs[0]
    obs_size = single_obs.size

    rb_mask_size = cfg.ensemble_size if ("boot" in cfg.kind or "amc" in cfg.kind) else 1

    rb = ReplayBuffer.make(
        10_000,
        single_obs,
        action_space.sample(key),
        mask_size=rb_mask_size,
    )
    key, _ = jr.split(key, 2)

    if cfg.kind == "boot":
        model = Bootstrapped(
            action_space=action_space,
            model_factory=lambda k: Model(obs_size, act_size, key=k),
            ensemble_size=cfg.ensemble_size,
            key=model_key,
        )
    elif cfg.kind == "bootrp":
        model = Bootstrapped(
            action_space=action_space,
            model_factory=lambda k: ModelWithPrior(
                obs_size, act_size, scale=cfg.prior_scale, key=k
            ),
            ensemble_size=cfg.ensemble_size,
            key=model_key,
        )
    elif cfg.kind == "eps":
        model = EpsilonGreedy(
            action_space=action_space,
            model_factory=lambda k: Model(obs_size, act_size, key=k),
            key=model_key,
        )
        raise RuntimeError(
            "Btw, epsilon greedy is probably broken because progress handling depends on a nonlocal value captured by closure which is probably bad idk"
        )
    elif cfg.kind == "dqn":
        model = DQN(
            action_space=action_space,
            model_factory=lambda k: Model(obs_size, act_size, key=k),
            key=model_key,
        )
    elif cfg.kind == "amc":
        model = AMCDQN(
            action_space=action_space,
            model_factory=lambda k: Model(obs_size, act_size, key=k),
            ensemble_size=cfg.ensemble_size,
            key=model_key,
        )
    elif cfg.kind == "amcrp":
        model = AMCDQN(
            action_space=action_space,
            model_factory=lambda k: ModelWithPrior(
                obs_size, act_size, scale=cfg.prior_scale, key=k
            ),
            ensemble_size=cfg.ensemble_size,
            key=model_key,
        )
    else:
        raise TypeError(
            f"{cfg.kind} of the model is undefined, only allowed ['eps', 'boot', 'bootrp', 'dqn', 'amc']"
        )

    key, _ = jr.split(key, 2)
    optim = optax.adamw(cfg.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit(donate="all-except-first")
    def train(rb, model, opt_state, key):
        key, fkey = jr.split(key)

        def loss_wrap(model):
            keys = jr.split(fkey, cfg.batch_size)
            samples = rb.sample(key, cfg.batch_size)
            loss, aux = eqx.filter_vmap(model.loss)(keys, samples)
            return loss.mean(), aux

        (loss, info), grads = eqx.filter_value_and_grad(loss_wrap, has_aux=True)(model)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, (*info, loss)

    @eqx.filter_jit
    def step(model, rb, obs_state, key, progress, model_indices):
        obs, state = obs_state
        step_key, mask_key, key = jr.split(key, 3)

        eps = jnp.clip(1 - progress * 2, 0.05, 1.0)

        def unbatched_step(key, state, obs, index):
            act_key, step_key = jr.split(key)
            action = model.action(obs, act_key, epsilon=eps, index=index)
            return *env.step(step_key, state, action, env_params), action

        n_obs, state, reward, done, _, action = eqx.filter_vmap(unbatched_step)(
            jr.split(step_key, cfg.num_envs), state, obs, model_indices
        )

        # NOTE: The mask does not do anything as of now: this is fine, follows p=1
        rb = rb.add(
            obs,
            n_obs,
            action,
            reward,
            done,
            mask=jr.bernoulli(mask_key, p=0.5, shape=(*reward.shape, rb_mask_size)),
        )
        obs = n_obs

        return model, rb, (obs, state), (done, reward)

    if debug:
        jax.debug.print("Starting to fill the replay buffer...")

    # collect stuff for the replay buffer
    def rb_scan_fn(carry_dyn, key, carry_st):
        carry = eqx.combine(carry_dyn, carry_st)
        model, rb, (obs, state) = carry
        key, subkey = jr.split(key)
        model, rb, (obs, state), _ = step(
            model,
            rb,
            (obs, state),
            subkey,
            progress=0,
            model_indices=jnp.arange(cfg.num_envs),
        )
        carry = (model, rb, (obs, state))
        carry_dyn, _ = eqx.partition(carry, eqx.is_array)
        return carry_dyn, None

    carry = (model, rb, (obs, state))
    carry_dyn, carry_st = eqx.partition(carry, eqx.is_array)
    n_iters = rb.buffer_size // cfg.num_envs
    carry_dyn, _ = jax.lax.scan(
        eqx.Partial(rb_scan_fn, carry_st=carry_st), init=carry_dyn, xs=jr.split(key, n_iters)
    )
    carry = eqx.combine(carry_dyn, carry_st)
    model, rb, (obs, state) = carry

    if debug:
        jax.debug.print("Filled out the replay buffer. Starting training.")

    @eqx.filter_jit(donate="all")
    def inner_loop(model, rb, obs, state, model_indices, opt_state, key, rews, i):
        progress = jnp.clip(i / num_steps, 0.0, 1.0)
        key, subkey, train_key, remake_key = jr.split(key, 4)
        d_key = jr.split(subkey, cfg.num_envs)

        # a single step in all the repeats of the environments
        model, rb, (obs, state), (dones, c_rewards) = step(
            model, rb, (obs, state), subkey, progress, model_indices
        )

        model, opt_state, train_info = train(rb, model, opt_state, train_key)
        rews += c_rewards

        _log = {
            "qval": train_info[0].mean(),
            "qtarget": train_info[1].mean(),
            "tdloss": train_info[2].mean(),
            "train_reward": jnp.where(dones, rews, jnp.nan),
            "model_indices": model_indices,
        }

        def episode_done(rews, model_indices, j):
            # in case an episode is completed we set the reward to an observed one,
            # and choose a new member to be ran
            model_indices = model_indices.at[j].set(jr.randint(d_key[j], (), 0, cfg.ensemble_size))
            rews = rews.at[j].set(0)
            return rews, model_indices

        # the following chooses new models and records rewards (for logging purposes)
        # when episode is completed
        (rews, model_indices), _ = jax.lax.scan(
            lambda carry, j: (
                jax.lax.cond(
                    dones[j],
                    lambda: episode_done(*carry, j),
                    lambda: carry,
                ),
                None,
            ),
            init=(rews, model_indices),
            xs=jnp.arange(len(dones)),
        )

        # synchronize the target model once in a while
        if cfg.kind == "amc" or cfg.kind == "amcrp":
            do_resample = i % (1000 // cfg.num_envs) == (1000 // cfg.num_envs - 1)

            model = filtered_cond(
                do_resample,
                lambda: model.vi_update(remake_key),
                lambda: model,
            )
            """
            jax.lax.cond(
                do_resample,
                lambda: jax.debug.callback(
                    lambda do, v: print(v) if do else None,
                    do_resample,
                    model.vi_logvar.layers[1].weight[3, 4],
                ),
                lambda: None,
            )
            """
        # Regular target update for other methods
        model = eqx.tree_at(
            lambda _m: _m.target_model,
            model,
            jax.lax.cond(
                i % (500 // cfg.num_envs) == (500 // cfg.num_envs - 1),
                lambda: model.model,
                lambda: model.target_model,
            ),
        )

        key, _ = jr.split(key, 2)
        return model, rb, obs, state, model_indices, opt_state, key, rews, _log

    rews = np.zeros((cfg.num_envs,), dtype=np.float32)
    # NOTE: I want to have exactly "Y" episodes, not exactly "Z" steps!!
    # NOTE: each episode in deepsea takes exactly 'hardness' steps
    assert "DeepSea" in cfg.env_name, "you schedule episodes based on hardness"
    num_steps = cfg.num_episodes * cfg.hardness // cfg.num_envs

    model_indices = jr.randint(ikey, (cfg.num_envs,), 0, cfg.ensemble_size)

    def inner_loop_wrap(carry_dyn, i, carry_st):
        carry = eqx.combine(carry_dyn, carry_st)
        *new_carry, _log = inner_loop(*carry, i)
        new_carry = eqx.filter(new_carry, eqx.is_array)
        return tuple(new_carry), _log

    carry = (model, rb, obs, state, model_indices, opt_state, key, rews)
    carry_dyn, carry_st = eqx.partition(carry, eqx.is_array)
    partial_fn = eqx.Partial(inner_loop_wrap, carry_st=carry_st)

    carry_dyn, logs = jax.lax.scan(
        partial_fn,
        init=carry_dyn,
        xs=jnp.arange(num_steps),
    )
    model, rb, obs, state, model_indices, opt_state, key, rews = eqx.combine(carry_dyn, carry_st)

    return model, logs


def schedule_runs(
    N: int, cfg: Config, output_root: str, concurrent: int = max_trainings_in_parallel
):
    # This function just reports results in a nice format
    VERSION = 1
    run_name_base = f"{cfg.short_str()}"
    run_name = run_name_base
    folder_path = Path(output_root) / run_name

    if folder_path.exists():
        print(f"{folder_path} exists hence skipping")
        return

    results = []
    thresh = 0.95

    tqdm.write(f"Starting the run scheduler with {cfg.short_str()}, N={N}")

    starting_seed = cfg.autoseed()
    keys = jr.split(jr.key(starting_seed), N)
    tqdm.write(f"Scheduler using seed: {starting_seed}")

    concurrent = min(N, concurrent)
    pbar = range(0, N, concurrent)
    if N >= concurrent * 5:
        pbar = tqdm(pbar)
    for _i in pbar:
        # next line does the actual training with given seeds
        _, logs = eqx.filter_vmap(eqx.Partial(main, cfg=cfg))(keys[_i : _i + concurrent])
        for tr, m_indices in zip(logs["train_reward"], logs["model_indices"]):
            tr = np.array(tr)
            mask = np.logical_not(np.isnan(tr))
            tr, m_indices = tr[mask], m_indices[mask]

            weak_convergence = tr.max() >= thresh
            time_to_weak = (tr >= thresh).argmax() if weak_convergence else len(tr)

            strong_convergence = True
            time_to_strong = 0
            for _m_id in range(cfg.ensemble_size):
                member_convergence = np.logical_and(tr >= thresh, _m_id == m_indices)
                strong_convergence &= member_convergence.any()
                time_to_strong = max(
                    member_convergence.argmax() if member_convergence.any() else len(tr),
                    time_to_strong,
                )

            results.append(
                {
                    "weak_convergence": weak_convergence,
                    "time_to_weak": time_to_weak,
                    "strong_convergence": strong_convergence,
                    "time_to_strong": time_to_strong,
                }
            )
            tqdm.write(f"{results[-1]}")

        if not folder_path.exists():
            folder_path.mkdir(parents=True)
            with open(folder_path / "config.yaml", "w") as f:
                yaml.safe_dump(dataclasses.asdict(cfg), f)
            with open(folder_path / ".version", "w") as f:
                f.write(f"{VERSION}")
        pd.DataFrame(results).to_csv(folder_path / "results.csv", index=False)

    return results


if __name__ == "__main__":
    experiment = "14"
    N = 50
    for kind in ["boot", "bootrp"]:
        for hardness in [6, 8, 10, 12, 14, 16]:
            for ens_size in [2, 3, 5, 7, 10, 12, 16]:
                schedule_runs(
                    N,
                    cfg=Config(
                        kind=kind, num_episodes=50_000, ensemble_size=ens_size, hardness=hardness
                    ),
                    output_root=f"results/{experiment}",
                )
