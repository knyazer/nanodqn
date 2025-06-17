import equinox as eqx
from typing import List, Protocol, TypeVar, Any
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
from tqdm import tqdm as tqdm

DOUBLE_DQN = False

T = TypeVar("T")


def _loss(model, target_model, sample):
    if DOUBLE_DQN:
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


def keys_like(pytree, key):
    leaves, treedef = jax.tree.flatten(pytree)
    subkeys = jr.split(key, len(leaves))
    return jax.tree.unflatten(treedef, subkeys)


def filtered_cond(cond, btrue, bfalse, *args):
    dyntrue, st = eqx.partition(btrue(*args), eqx.is_array)
    dynfalse = eqx.filter(bfalse(*args), eqx.is_array)
    dynfin = jax.lax.cond(cond, lambda: dyntrue, lambda: dynfalse)
    return eqx.combine(st, dynfin)


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
    scale: float = eqx.field(static=True)

    def __init__(self, observation_size, action_size, /, scale, key, layer_sizes=None):
        k1, k2 = jr.split(key)
        super().__init__(observation_size, action_size, key=k1, layer_sizes=layer_sizes)

        self.prior = Model(observation_size, action_size, key=k2, layer_sizes=layer_sizes)
        self.scale = float(scale)

    def __call__(self, x):
        x = x.ravel()
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
        return _loss(self.model, self.target_model, sample)

    def action(self, observation: Any, *args, **kwargs):
        return self.model(observation).argmax()


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
    """Any bootstrapped model"""

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

    def loss_all(self, samples):
        def loss_single(sample, model_index):
            s = self[model_index]
            return eqx.filter_vmap(lambda smp: _loss(s.model, s.target_model, smp))(sample)

        return eqx.filter_vmap(loss_single)(samples, jnp.arange(self.ensemble_size))

    def loss(self, key: PRNGKeyArray, sample):
        model_index = jr.randint(key, (), 0, self.ensemble_size)
        s = self[model_index]
        return _loss(s.model, s.target_model, sample)

    def q_pred_all(self, sample):
        def wrp(m_index):
            m = self[m_index]
            return eqx.filter_vmap(m.model)(sample.next_observations.squeeze())

        return eqx.filter_vmap(wrp)(jnp.arange(self.ensemble_size))

    def action(self, observation: Float[Array, "obs_size"], *_, index=None, **kws):
        assert index is not None
        single_model = jax.tree.map(
            lambda node: node[index] if eqx.is_array(node) else node, self.model
        )
        return single_model(observation).argmax()
