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

    def __init__(self, observation_size, action_size, /, key, layer_sizes=None):
        k1, k2 = jr.split(key)
        super().__init__(observation_size, action_size, key=k1, layer_sizes=layer_sizes)

        self.prior = Model(observation_size, action_size, key=k2, layer_sizes=layer_sizes)

    def __call__(self, x):
        return super().__call__(x) + jax.lax.stop_gradient(self.prior(x))


gamma = 0.99


class DQN(eqx.Module):
    action_space: Any
    model: Model
    target_model: Model
    replay_buffer: ReplayBuffer

    def __init__(self, /, model_factory, action_space, rb, key: PRNGKeyArray):
        mkey, tkey = jr.split(key, 2)
        self.model = model_factory(mkey)
        self.target_model = model_factory(tkey)
        self.action_space = action_space
        self.replay_buffer = rb

    def loss(self, key: PRNGKeyArray):
        rkey, key = jr.split(key)

        sample = self.replay_buffer.sample(rkey, 1)
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

    def add_to_buffer(self, *data):
        return eqx.tree_at(lambda s: s.replay_buffer, self, self.replay_buffer.add(*data))

    def sync_target(self):
        self = eqx.tree_at(lambda m: m.target_model, self, self.model)
        return self


class EpsilonGreedy(DQN):
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
    replay_buffer: ReplayBuffer

    def __init__(self, /, model_factory, ensemble_size, action_space, rb, key: PRNGKeyArray):
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
        self.replay_buffer = jax.tree.map(
            lambda *nodes: jnp.stack(nodes), *([rb] * ensemble_size), is_leaf=eqx.is_array
        )

    def __getitem__(self, idx):
        return jax.tree.map(lambda node: node[idx] if eqx.is_array(node) else node, self)

    def __len__(self):
        return self.ensemble_size

    def add_to_buffer(self, index, data):
        new_rb = self[index].replay_buffer.add(*data)
        new_stacked_rb = jax.tree.map(
            lambda node, newnode: node.at[index].set(newnode), self.replay_buffer, new_rb
        )
        return eqx.tree_at(lambda s: s.replay_buffer, self, new_stacked_rb)

    def loss(self, key: PRNGKeyArray):
        rkey, subkey, key = jr.split(key, 3)

        # sample data from the buffer
        model_index = jr.randint(subkey, (), 0, self.ensemble_size)
        self = self[model_index]

        sample = self.replay_buffer.sample(rkey, 1)
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


batch_size = 64
lr = 1e-4
num_envs = 40
ensemble_size = 4
env_name = "DeepSea-bsuite"
w_group = "boot4 sweep"


def main(seed=0):
    key = jr.key(seed)
    key, model_key, target_model_key, reset_key, loop_key, ikey = jr.split(key, 6)

    env, env_params = gymnax.make(env_name, size=50)
    obs, state = jax.vmap(lambda k: env.reset(k, env_params))(jr.split(reset_key, num_envs))
    action_space = env.action_space(env_params)  # type: ignore

    assert "n" in action_space.__dict__, (
        "The environment is not discrete, or maybe incorrectly initialized"
    )
    act_size = action_space.__dict__.get("n", 2)
    single_obs = obs[0]
    obs_size = single_obs.size

    model = Bootstrapped(
        action_space=action_space,
        model_factory=lambda k: Model(obs_size, act_size, key=k),
        ensemble_size=ensemble_size,
        rb=ReplayBuffer(10_000, single_obs, action_space.sample(key)),
        key=model_key,
    )

    """
    model = EpsilonGreedy(
        action_space=action_space,
        model_factory=lambda k: Model(obs_size, act_size, key=k),
        rb=ReplayBuffer(10_000, single_obs, action_space.sample(key)),
        key=model_key,
    )
    """

    optim = optax.adamw(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train(model, opt_state, key):
        key, fkey = jr.split(key)

        def loss_wrap(model):
            keys = jr.split(fkey, batch_size)
            loss, aux = eqx.filter_vmap(model.loss)(keys)
            return loss.mean(), aux

        (loss, info), grads = eqx.filter_value_and_grad(loss_wrap, has_aux=True)(model)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, (*info, loss)

    @eqx.filter_jit
    def eval_run(model, key):
        key, reset_key, ikey = jr.split(key, 3)
        obs, state = env.reset(reset_key, env_params)
        init_carry = (key, obs, state, False, jnp.zeros(()))
        index = jr.randint(ikey, (), 0, ensemble_size)

        def _step(carry, _):
            key, obs, state, done, ret = carry
            key, act_key, step_key = jr.split(key, 3)

            def env_step():
                action = model.action(obs, act_key, epsilon=0.0, index=index)
                obs2, state2, reward, done2, _ = env.step(step_key, state, action, env_params)
                return (key, obs2, state2, done2, ret + reward)

            return jax.lax.cond(done, lambda: (key, obs, state, done, ret), env_step), None

        final_carry, _ = jax.lax.scan(_step, init_carry, xs=None, length=500)
        _, _, _, _, episode_return = final_carry

        return episode_return

    @eqx.filter_jit
    def step(model, obs_state, key, progress, model_indices):
        obs, state = obs_state
        step_key, key = jr.split(key)

        eps = jnp.clip(1 - progress * 2, 0.05, 1.0)

        def unbatched_step(key, state, obs, index):
            act_key, step_key = jr.split(key)
            action = model.action(obs, act_key, epsilon=eps, index=index)
            return *env.step(step_key, state, action, env_params), action

        n_obs, state, reward, done, _, action = eqx.filter_vmap(unbatched_step)(
            jr.split(step_key, num_envs), state, obs, model_indices
        )
        # Note that the following does not follow the RP nor BS paper, but
        # instead is more intuitive: we add the transitions only to our "own"
        # buffers
        if isinstance(model, Bootstrapped):
            for _ in range(2):
                random_indices = jr.randint(key, model_indices.shape, 0, ensemble_size)
                for i, model_index in zip(range(len(n_obs)), random_indices):
                    model = model.add_to_buffer(
                        model_index, (obs[i], n_obs[i], action[i], reward[i], done[i])
                    )
        else:
            for i in range(len(n_obs)):
                model = model.add_to_buffer(obs[i], n_obs[i], action[i], reward[i], done[i])

        obs = n_obs

        return model, (obs, state), (done, reward)

    jax.debug.print("Starting to fill the replay buffer...")

    # collect stuff for the replay buffer
    for i in range(10_000 // num_envs):
        key, subkey = jr.split(key)
        for ind in range(ensemble_size):
            model, (obs, state), info = step(
                model, (obs, state), subkey, progress=0, model_indices=jnp.array([ind] * num_envs)
            )

    jax.debug.print("Filled out the replay buffer. Starting training.")

    def init_wandb():
        wandb.init(name=f"{type(model).__name__}-{env_name}//({ensemble_size})", group=w_group)
        wandb.run.log_code(".")  # type: ignore

    jax.debug.callback(init_wandb)

    @eqx.filter_jit
    def inner_loop(model, obs, state, model_indices, opt_state, key, rews, i):
        progress = jnp.clip(i / num_steps, 0.0, 1.0)
        key, subkey, train_key = jr.split(key, 3)
        d_key = jr.split(subkey, num_envs)
        model, (obs, state), (dones, c_rewards) = step(
            model, (obs, state), subkey, progress, model_indices
        )

        model, opt_state, train_info = train(model, opt_state, train_key)
        rews += c_rewards

        _log = {
            "qval": train_info[0].mean(),
            "qtarget": train_info[1].mean(),
            "tdloss": train_info[2].mean(),
            "train_reward": jax.lax.cond(
                jnp.count_nonzero(dones) == 0, lambda: jnp.nan, lambda: (rews * dones).max()
            ),
        }

        def episode_done(rews, model_indices, j):
            model_indices = model_indices.at[j].set(jr.randint(d_key[j], (), 0, ensemble_size))
            rews = rews.at[j].set(0)
            return rews, model_indices

        for j in range(len(dones)):
            rews, model_indices = jax.lax.cond(
                dones[j],
                lambda: episode_done(rews, model_indices, j),
                lambda: (rews, model_indices),
            )

        model = eqx.tree_at(
            lambda _m: _m.target_model,
            model,
            jax.lax.cond(
                i % (500 // num_envs) == (500 // num_envs - 1),
                lambda: model.model,
                lambda: model.target_model,
            ),
        )

        """
        def fast_eval(model, key):
            key, eval_key = jr.split(key)
            eval_rewards = eqx.filter_vmap(lambda k: eval_run(model, k))(
                jr.split(eval_key, batch_size * 8)
            )
            _log({"eval_reward": eval_rewards.mean()})
            jax.debug.print("Eval reward: {}", eval_rewards.mean())
            return key

        key = jax.lax.cond(
            i % (num_steps // 20) == (num_steps // 20) - 1,
            lambda: fast_eval(model, key),
            lambda: key,
        )
        """

        return model, obs, state, model_indices, opt_state, key, rews, _log

    rews = np.zeros((num_envs,), dtype=np.float32)
    num_steps = 50_000 // num_envs

    model_indices = jr.randint(ikey, (num_envs,), 0, ensemble_size)

    def inner_loop_wrap(carry_dyn, i, carry_st):
        carry = eqx.combine(carry_dyn, carry_st)
        *new_carry, _log = inner_loop(*carry, i)
        new_carry = eqx.filter(new_carry, eqx.is_array)
        return tuple(new_carry), _log

    @eqx.filter_jit
    def fast_eval(model, key):
        key, eval_key = jr.split(key)
        eval_rewards = eqx.filter_vmap(lambda k: eval_run(model, k))(
            jr.split(eval_key, batch_size * 4)
        )
        return eval_rewards.mean()

    partial_fn = None
    for i in range(0, num_steps, 50):
        carry = (model, obs, state, model_indices, opt_state, key, rews)
        carry_dyn, carry_st = eqx.partition(carry, eqx.is_array)
        if partial_fn is None:
            partial_fn = eqx.Partial(inner_loop_wrap, carry_st=carry_st)

        carry_dyn, logs = jax.lax.scan(
            partial_fn,
            init=carry_dyn,
            xs=jnp.arange(i, i + 50),
        )
        model, obs, state, model_indices, opt_state, key, rews = eqx.combine(carry_dyn, carry_st)

        for j in range(len(logs[list(logs.keys())[0]])):
            _log = {}
            for k in logs:
                _log[k] = logs[k][j]
            if _log["train_reward"] == jnp.nan:
                del _log["train_reward"]
            wandb.log({**_log})
        wandb.log({"eval_reward": fast_eval(model, key)})

    wandb.finish()
    return model


if __name__ == "__main__":
    for seed in range(12):
        main(seed)
