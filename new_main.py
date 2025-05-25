import equinox as eqx
from typing import List, Protocol, TypeVar, Any
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
import gymnax
from replay_buffer import ReplayBuffer, ReplayBufferSample
from tqdm import tqdm as tqdm
import optax

breakpoint()

T = TypeVar("T")


class ModuleT(Protocol[T]):
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...


class ActionSpace(eqx.Module):
    def size(self):
        return 0

    def sample(self, key: PRNGKeyArray):
        return 0


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

    def __call__(self, x: Float[Array, "action_size"]):  # noqa
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        return x


gamma = 0.99


class DQN(eqx.Module):
    action_space: ActionSpace
    model: Model
    target_model: Model

    def action(self, observation: Any, key: PRNGKeyArray):
        return self.model(observation).argmax()

    def loss(self, sample: ReplayBufferSample):
        next_target_q_value = jnp.max(self.target_model(sample.next_observations))
        next_q_value = sample.rewards + (1 - sample.dones) * gamma * next_target_q_value

        q_pred = self.model(sample.observations)
        q_pred = q_pred[sample.actions]
        return ((q_pred - next_q_value) ** 2).mean()


class EpsilonGreedy(DQN):
    def action(self, observation: Any, key: PRNGKeyArray, /, progress: float):  # type: ignore
        epsilon = 1 - progress
        epsilon = eqx.error_if(epsilon, (epsilon < 0) or (epsilon > 1), f"Got eps={epsilon}")

        return jax.lax.cond(
            jr.uniform(key, ()) < epsilon,
            lambda: self.action_space.sample(key),
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

    def action(self, observation: Any, key: PRNGKeyArray):
        key, subkey = jr.split(key)
        # choose a random network
        model_index = jr.randint(subkey, (), 0, self.ensemble_size)
        single_model = jax.tree.map(
            lambda node: node[model_index] if eqx.is_array(node) else node, self.model
        )
        return single_model(observation)


batch_size = 64

if __name__ == "__main__":
    key = jr.key(0)
    key, model_key, target_model_key, reset_key = jr.split(key, 4)

    env, env_params = gymnax.make("CartPole-v1")
    obs, state = env.reset(reset_key, env_params)
    action_space = env.action_space(env_params)

    assert "n" in action_space.__dict__, (
        "The environment is not discrete, or maybe incorrectly initialized"
    )

    ex_obs = obs
    ex_act = action_space.sample(key)

    network = Model(ex_obs.size, ex_act.size, key=model_key)
    model = EpsilonGreedy(
        action_space=action_space,
        model=network,
        target_model=network,
    )

    optim = optax.adamw(1e-4)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    replay_buffer = ReplayBuffer(10_000, ex_obs, ex_act)

    def train(model, opt_state, rb, key):
        samples = eqx.filter_vmap(rb.sample)(jr.split(key, batch_size))
        loss, grads = eqx.filter_value_and_grad(lambda m: eqx.filter_vmap(m.loss)(samples).mean())(
            model
        )
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def step(model, opt_state, obs_state, replay_buffer, key, *, only_data=False):
        obs, state = obs_state
        act_key, step_key, train_key, key = jr.split(key, 4)

        action = model.action(obs, act_key, progress=0.1)
        n_obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
        replay_buffer = replay_buffer.add(obs, n_obs, action, reward, done)

        if not only_data:
            model, opt_state, loss = train(model, opt_state, replay_buffer, train_key)

        obs = n_obs
        return model, opt_state, (obs, state), replay_buffer, key

    inp = (model, opt_state, (obs, state), replay_buffer, key)
    for i in tqdm(range(10_000)):
        inp = step(*inp, only_data=True)
    breakpoint()
