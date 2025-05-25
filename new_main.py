import equinox as eqx
import numpy as np
from typing import List, Protocol, TypeVar, Any
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
import gymnax
from gymnax.visualize import Visualizer
from replay_buffer import ReplayBuffer, ReplayBufferSample
from tqdm import tqdm as tqdm
import optax

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
            layer_sizes = [120, 84]
        layer_sizes = [observation_size, *layer_sizes, action_size]

        layers = []
        for inpsize, outsize in zip(layer_sizes[:-1], layer_sizes[1:]):
            subkey, key = jr.split(key)
            layers.append(eqx.nn.Linear(inpsize, outsize, key=subkey))

        self.layers = layers

    def __call__(self, x: Float[Array, "obs_size"]) -> Float[Array, "act_size"]:  # noqa
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return x


gamma = 0.98


class DQN(eqx.Module):
    action_space: Any
    model: Model
    target_model: Model

    def loss(self, sample: ReplayBufferSample):
        assert len(sample.observations.shape) == 1
        next_target_q_value = jnp.max(self.target_model(sample.next_observations), axis=-1)
        next_q_value = jax.lax.stop_gradient(
            sample.rewards + (1 - sample.dones) * gamma * next_target_q_value
        )

        q_pred = self.model(sample.observations)
        q_pred = q_pred[sample.actions]
        return ((q_pred - next_q_value) ** 2).mean()


class EpsilonGreedy(DQN):
    def action(self, observation: Any, key: PRNGKeyArray, epsilon: Float[Array, ""]):  # type: ignore
        choice_key, act_key = jr.split(key)
        return jax.lax.cond(
            jr.uniform(choice_key, ()) < epsilon,
            lambda: jax.lax.stop_gradient(self.action_space.sample(act_key)),
            lambda: self.model(observation).argmax(),
        )


class Bootstrapped(DQN):
    ensemble_size: int
    model: Model  # this model is stacked node-wise

    def __init__(self, model_factory, ensemble_size, key: PRNGKeyArray):
        keys = jr.split(key, ensemble_size)
        models = [model_factory(mkey) for mkey in keys]
        self.model = jax.tree.map(lambda *nodes: jnp.stack(nodes), models, is_leaf=eqx.is_array)
        self.ensemble_size = ensemble_size

    def action(self, observation: Float[Array, "obs_size"], key: PRNGKeyArray):
        key, subkey = jr.split(key)
        # choose a random network
        model_index = jr.randint(subkey, (), 0, self.ensemble_size)
        single_model = jax.tree.map(
            lambda node: node[model_index] if eqx.is_array(node) else node, self.model
        )
        return single_model(observation).argmax()


batch_size = 128

if __name__ == "__main__":
    key = jr.key(0)
    key, model_key, target_model_key, reset_key, loop_key = jr.split(key, 5)

    env, env_params = gymnax.make("CartPole-v1")
    obs, state = env.reset(reset_key, env_params)
    action_space = env.action_space(env_params)  # type: ignore

    assert "n" in action_space.__dict__, (
        "The environment is not discrete, or maybe incorrectly initialized"
    )
    act_size = action_space.__dict__.get("n", 2)

    model = EpsilonGreedy(
        action_space=action_space,
        model=Model(obs.size, act_size, key=model_key),
        target_model=Model(obs.size, act_size, key=target_model_key),
    )

    optim = optax.chain(
        optax.adamw(3e-4),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    replay_buffer = ReplayBuffer(10_000, obs, action_space.sample(key))

    def train(model, opt_state, rb, key):
        samples = eqx.filter_vmap(rb.sample)(jr.split(key, batch_size))
        jax.debug.print("{}", samples.observations)
        info, grads = eqx.filter_value_and_grad(lambda m: eqx.filter_vmap(m.loss)(samples).mean())(
            model
        )
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, info

    def eval_run(model, key):
        reset_key, key = jr.split(key)
        obs, state = env.reset(reset_key, env_params)
        init_carry = (key, obs, state, False, jnp.zeros(()))

        def _step(carry, _):
            key, obs, state, done, ret = carry
            key, act_key, step_key = jr.split(key, 3)

            def env_step():
                action = model.action(obs, act_key, epsilon=0.0)
                obs2, state2, reward, done2, _ = env.step(step_key, state, action, env_params)
                return (key, obs2, state2, done2, ret + reward)

            return jax.lax.cond(done, lambda: (key, obs, state, done, ret), env_step), state

        final_carry, state_arr = jax.lax.scan(_step, init_carry, xs=None, length=500)
        _, _, _, _, episode_return = final_carry

        return episode_return, state_arr

    @eqx.filter_jit
    def step(model, opt_state, obs_state, replay_buffer, key, progress, *, only_data=False):
        obs, state = obs_state
        act_key, step_key, train_key, key = jr.split(key, 4)

        eps = jnp.clip(1 - progress, 0.05, 1.0)
        action = model.action(obs, act_key, eps)
        n_obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        replay_buffer = replay_buffer.add(obs, n_obs, action, reward, done)
        obs = n_obs

        info = []
        if not only_data:
            model, opt_state, train_info = train(model, opt_state, replay_buffer, train_key)
            info += [train_info]

        return model, opt_state, (obs, state), replay_buffer, (done, info)

    # collect stuff for the replay buffer
    for i in tqdm(range(10_000)):
        key, subkey = jr.split(key)
        model, opt_state, (obs, state), replay_buffer, info = step(
            model, opt_state, (obs, state), replay_buffer, subkey, jnp.array(0.0), only_data=True
        )

    # the actual training
    deltas = []
    delta = 0
    for i in (pbar := tqdm(range(200_000))):
        progress = jnp.clip(i / 100_000, 0.0, 1.0)
        key, subkey = jr.split(key)
        model, opt_state, (obs, state), replay_buffer, info = step(
            model, opt_state, (obs, state), replay_buffer, subkey, progress, only_data=(i % 10 != 0)
        )

        delta += 1
        if info[0]:
            deltas.append(delta)
            delta = 0

        if i % 500 == 499:
            model = eqx.tree_at(lambda m: m.target_model, model, model.model)

        if i % 50 == 0:
            rewards = np.array(deltas)
            timedisc = 0.95 ** (len(rewards) - np.arange(len(rewards)))
            reward = (rewards * timedisc).sum() / timedisc.sum()
            pbar.set_description(f"Last train length-ish: {reward:.2f}")

        if i % 5000 == 0:
            key, eval_key = jr.split(key)
            eval_rewards, state_seq = eqx.filter_vmap(lambda k: eval_run(model, k))(
                jr.split(eval_key, 16)
            )
            print(f"{eval_rewards.mean():.2f}+-{eval_rewards.std():.2f}")
