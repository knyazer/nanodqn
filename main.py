from os import wait
import equinox as eqx
import numpy as np
from typing import List, Protocol, TypeVar, Any
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
import gymnax
from replay_buffer import ReplayBuffer, ReplayBufferSample
from tqdm import tqdm as tqdm
import optax
import wandb
import os
import pandas as pd

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


DOUBLE_DQN = False

T = TypeVar("T")


class ModuleT(Protocol[T]):
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...


class Model(eqx.Module):
    layers: List[ModuleT]

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        /,
        key: PRNGKeyArray,
        layer_sizes=None,
    ):
        if layer_sizes is None:
            layer_sizes = [24, 24]
        layer_sizes = [observation_size, *layer_sizes, action_size]

        layers = []
        for inpsize, outsize in zip(layer_sizes[:-1], layer_sizes[1:]):
            subkey, key = jr.split(key)
            layers.append(eqx.nn.Linear(inpsize, outsize, key=subkey))

        self.layers = layers

    def __call__(self, x: Float[Array, "*"]) -> Float[Array, "act_size"]:  # noqa
        x = x.ravel()  # flatten
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return x


class ModelWithPrior(Model):
    prior: Model
    scale: Float[Array, ""]

    def __init__(self, observation_size, action_size, /, scale, key, layer_sizes=None):
        k1, k2 = jr.split(key)
        super().__init__(observation_size, action_size, key=k1, layer_sizes=layer_sizes)

        self.prior = Model(observation_size, action_size, key=k2, layer_sizes=layer_sizes)
        self.scale = scale
        # we interpret "scaling MLP by beta" by "scaling each parameter of MLP by beta"
        # self.prior = jax.tree.map(
        #    lambda node: node * self.scale if eqx.is_inexact_array(node) else node, self.prior
        # )

    def __call__(self, x):
        return super().__call__(x) + jax.lax.stop_gradient(self.prior(x) * self.scale)


gamma = 0.99


class DQN(eqx.Module):
    action_space: Any
    model: Model
    target_model: Model

    def __init__(self, /, model_factory, action_space, key: PRNGKeyArray):
        mkey, tkey = jr.split(key, 2)
        self.model = model_factory(mkey)
        self.target_model = model_factory(tkey)
        self.action_space = action_space

    def loss(self, key: PRNGKeyArray, sample):
        rkey, key = jr.split(key)

        target_model = self.target_model
        model = self.model

        # compute the 'target' q value (next step)
        if DOUBLE_DQN:  # double dqn
            best_action = model(sample.next_observations.squeeze()).argmax()
            next_target_q_value = (target_model(sample.next_observations.squeeze()))[best_action]
        else:
            next_target_q_value = jnp.max(target_model(sample.next_observations.squeeze()))

        next_q_value = jax.lax.stop_gradient(
            sample.rewards + (1 - sample.dones) * gamma * next_target_q_value
        ).squeeze()

        # compute our model's prediction of the next value
        q_pred = model(sample.observations.squeeze())
        q_pred = q_pred[sample.actions]
        return ((q_pred - next_q_value) ** 2).mean(), (q_pred, next_q_value)

    def action(self, observation: Any, *args, **kwargs):
        return self.model(observation).argmax()


class AMCDQN(eqx.Module):
    """
    This class implements Adaptive MCMC DQN: sample a bunch of networks
    from a hypernetwork, backprop them for a bit (and fill the buffer with
    new transitions), and then do a single step on the VI model updating to the
    new posterior. As of now, the VI model is a normal.


    So, e.g. once every 20-ish steps of gradient descent, we do a single step on
    the VI model, resample the self.model to get the new ones, update the self.target_model
    to be the current self.model ones (the just-resampled ones) and then train for a bit again
    """

    ...


class EpsilonGreedy(DQN):
    def __init__(self, /, model_factory, action_space, key: PRNGKeyArray):
        mkey, tkey = jr.split(key, 2)
        self.model = model_factory(mkey)
        self.target_model = model_factory(tkey)
        self.action_space = action_space

    def action(self, observation: Any, key: PRNGKeyArray, epsilon: Float[Array, ""], **kw):
        choice_key, act_key = jr.split(key)
        return jax.lax.cond(
            jr.uniform(choice_key, ()) < epsilon,
            lambda: jax.lax.stop_gradient(self.action_space.sample(act_key)),
            lambda: self.model(observation).argmax(),
        )


class Bootstrapped(DQN):
    action_space: Any
    ensemble_size: int
    model: Model  # this model is stacked node-wise
    target_model: Model  # this too, each target is node-wise

    def __init__(self, /, model_factory, ensemble_size, action_space, key: PRNGKeyArray):
        tkey, mkey = jr.split(key)
        keys = jr.split(mkey, ensemble_size)
        models = [model_factory(mkey) for mkey in keys]

        keys = jr.split(tkey, ensemble_size)
        tmodels = [model_factory(tkey) for tkey in keys]

        self.model = jax.tree.map(lambda *nodes: jnp.stack(nodes), *models, is_leaf=eqx.is_array)
        self.target_model = jax.tree.map(
            lambda *nodes: jnp.stack(nodes), *tmodels, is_leaf=eqx.is_array
        )
        self.ensemble_size = ensemble_size
        self.action_space = action_space

    def __getitem__(self, idx):
        return jax.tree.map(lambda node: node[idx] if eqx.is_array(node) else node, self)

    def __len__(self):
        return self.ensemble_size

    def loss(self, key: PRNGKeyArray, sample):
        rkey, subkey, key = jr.split(key, 3)

        # sample data from the buffer
        model_index = jr.randint(subkey, (), 0, self.ensemble_size)
        self = self[model_index]

        target_model = self.target_model
        model = self.model

        # compute the 'target' q value (next step)
        if DOUBLE_DQN:  # double dqn
            best_action = model(sample.next_observations.squeeze()).argmax()
            next_target_q_value = (target_model(sample.next_observations.squeeze()))[best_action]
        else:
            next_target_q_value = jnp.max(target_model(sample.next_observations.squeeze()))

        next_q_value = jax.lax.stop_gradient(
            sample.rewards + (1 - sample.dones) * gamma * next_target_q_value
        ).squeeze()

        # compute our model's prediction of the next value
        q_pred = model(sample.observations.squeeze())
        q_pred = q_pred[sample.actions]
        return ((q_pred - next_q_value) ** 2).mean(), (q_pred, next_q_value)

    def action(self, observation: Float[Array, "obs_size"], *_, index=None, **kws):
        assert index is not None
        single_model = jax.tree.map(
            lambda node: node[index] if eqx.is_array(node) else node, self.model
        )
        return single_model(observation).argmax()


class Config(eqx.Module):
    batch_size: int = 128
    lr: float = 1e-4
    num_envs: int = 16
    ensemble_size: int = 4
    prior_scale: float = 3
    env_name: str = "DeepSea-bsuite"
    kind: str = "dqn"
    prefix: str = "j4"
    hardness: int = 24

    def __str__(self):
        return f"{self.prefix} {self.kind}({self.ensemble_size}) {self.env_name}({self.hardness})"


@eqx.filter_jit
def main(seed=0, cfg=Config(), debug=True):
    if debug:
        jax.debug.print(
            "Main is running with: kind={}, ensemble_size={}, hardness={}",
            cfg.kind,
            cfg.ensemble_size,
            cfg.hardness,
        )
    key = jr.key(seed)
    key, model_key, target_model_key, reset_key, loop_key, ikey = jr.split(key, 6)

    env, env_params = gymnax.make(cfg.env_name, size=cfg.hardness)
    obs, state = jax.vmap(lambda k: env.reset(k, env_params))(jr.split(reset_key, cfg.num_envs))
    action_space = env.action_space(env_params)  # type: ignore

    assert "n" in action_space.__dict__, (
        "The environment is not discrete, or maybe incorrectly initialized"
    )
    act_size = action_space.__dict__.get("n", 2)
    single_obs = obs[0]
    obs_size = single_obs.size

    rb_mask_size = cfg.ensemble_size if "boot" in cfg.kind else 1

    rb = ReplayBuffer.make(
        10_000,
        single_obs,
        action_space.sample(key),
        mask_size=rb_mask_size,
    )

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
    elif cfg.kind == "dqn":
        model = DQN(
            action_space=action_space,
            model_factory=lambda k: Model(obs_size, act_size, key=k),
            key=model_key,
        )
    else:
        raise TypeError(
            f"{cfg.kind} of the model is undefined, only allowed ['eps', 'boot', 'bootrp', 'dqn']"
        )

    optim = optax.adamw(cfg.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def eval_model(model, key):
        @eqx.filter_jit
        def eval_run(model, key):
            key, reset_key, ikey = jr.split(key, 3)
            obs, state = env.reset(reset_key, env_params)
            init_carry = (key, obs, state, False, jnp.zeros(()))
            index = jr.randint(ikey, (), 0, cfg.ensemble_size)

            def _step(carry, _):
                key, obs, state, done, ret = carry
                key, act_key, step_key = jr.split(key, 3)

                def env_step():
                    action = model.action(obs, act_key, epsilon=0.0, index=index)
                    obs2, state2, reward, done2, _ = env.step(step_key, state, action, env_params)
                    return (key, obs2, state2, done2, ret + reward)

                return jax.lax.cond(done, lambda: (key, obs, state, done, ret), env_step), None

            final_carry, _ = jax.lax.scan(_step, init_carry, xs=None, length=hardness)
            _, _, _, _, episode_return = final_carry

            return episode_return

        key, eval_key = jr.split(key)
        eval_rewards = eqx.filter_vmap(lambda k: eval_run(model, k))(
            jr.split(eval_key, cfg.batch_size * 4)
        )
        return eval_rewards.mean()

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
        # Note that the following does not follow the RP nor BS paper, but
        # instead is more intuitive: we add the transitions only to our "own"
        # buffers

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
        key, subkey, train_key = jr.split(key, 3)
        d_key = jr.split(subkey, cfg.num_envs)
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
            model_indices = model_indices.at[j].set(jr.randint(d_key[j], (), 0, cfg.ensemble_size))
            rews = rews.at[j].set(0)
            return rews, model_indices

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

        model = eqx.tree_at(
            lambda _m: _m.target_model,
            model,
            jax.lax.cond(
                i % (500 // cfg.num_envs) == (500 // cfg.num_envs - 1),
                lambda: model.model,
                lambda: model.target_model,
            ),
        )

        return model, rb, obs, state, model_indices, opt_state, key, rews, _log

    rews = np.zeros((cfg.num_envs,), dtype=np.float32)
    num_steps = 100_000 // cfg.num_envs

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


def make_wandb_run_from_logs(cfg, logs):
    wandb.init(name=str(cfg), group=str(cfg), save_code=True)
    for j in range(len(logs[list(logs.keys())[0]])):
        _log = {}
        for k in logs:
            _log[k] = logs[k][j]
        if _log["train_reward"] == jnp.nan:
            del _log["train_reward"]
        wandb.log({**_log})

    wandb.finish()


max_trainings_in_parallel = 20


def schedule_runs(kind, N, ensemble_size=1):
    cfg = Config(kind=kind, ensemble_size=ensemble_size)
    print(f"Starting the run scheduler with {kind}{ensemble_size}, N={N}")
    results = []
    thresh = 0.95
    for i in tqdm(range(0, N, max_trainings_in_parallel)):
        _, logs = eqx.filter_vmap(eqx.Partial(main, cfg=cfg))(
            jnp.arange(i, i + max_trainings_in_parallel)
        )
        for tr, m_indices in zip(logs["train_reward"], logs["model_indices"]):
            tr = np.array(tr)
            mask = np.logical_not(np.isnan(tr))
            tr, m_indices = tr[mask], m_indices[mask]

            # weak convergence is "at least one model solves the problem"
            weak_convergence = tr.max() >= thresh
            time_to_weak = (tr >= thresh).argmax() if weak_convergence else len(tr)

            # strong convergence is "all models solve the problem"
            # thus for normal dqn they are the same, but for boot - not
            strong_convergence = True
            time_to_strong = 0
            for _m_id in range(ensemble_size):
                # note that strong convergence is reported in "steps of all members"
                # not "steps of this particular member" for consistency
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

    pd.DataFrame(results).to_csv(f"results/N={N}_{kind}{ensemble_size}.csv")
    return results


if __name__ == "__main__":
    N = 100
    schedule_runs("boot", N=N, ensemble_size=2)
    schedule_runs("boot", N=N, ensemble_size=4)
    schedule_runs("dqn", N=N)

    schedule_runs("boot", N=N, ensemble_size=1)
    schedule_runs("boot", N=N, ensemble_size=3)
    schedule_runs("boot", N=N, ensemble_size=5)
    """
    for ens_size in range(2, 10):
        schedule_runs("boot", N=100, ensemble_size=ens_size)
    """
