import equinox as eqx
from typing import List, Protocol, TypeVar, Any
from jax import random as jr
from jax import numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Float, Array
from tqdm import tqdm as tqdm

DOUBLE_DQN = False

T = TypeVar("T")


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
    scale: Float[Array, ""]

    def __init__(self, observation_size, action_size, /, scale, key, layer_sizes=None):
        k1, k2 = jr.split(key)
        super().__init__(observation_size, action_size, key=k1, layer_sizes=layer_sizes)

        self.prior = Model(observation_size, action_size, key=k2, layer_sizes=layer_sizes)
        self.scale = scale

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

    action_space: Any
    ensemble_size: int
    model: Model  # ensemble of models (like Bootstrapped)
    target_model: Model  # ensemble of target models
    vi_mean: Model  # variational mean parameters
    vi_logvar: Model  # variational log variance parameters

    def __init__(self, /, model_factory, ensemble_size, action_space, key: PRNGKeyArray):
        self.action_space = action_space
        self.ensemble_size = ensemble_size

        # Initialize variational parameters
        vi_key, sample_key = jr.split(key)
        self.vi_mean = model_factory(vi_key)
        self.vi_logvar = jax.tree.map(lambda x: jnp.full_like(x, -2.0), self.vi_mean)

        # Sample initial ensemble from VI distribution
        sample_keys = jr.split(sample_key, ensemble_size * 2)
        model_keys, target_keys = sample_keys[:ensemble_size], sample_keys[ensemble_size:]

        models = [self._sample_from_vi(k) for k in model_keys]
        target_models = [self._sample_from_vi(k) for k in target_keys]

        # Stack like in Bootstrapped
        self.model = jax.tree.map(lambda *nodes: jnp.stack(nodes), *models, is_leaf=eqx.is_array)
        self.target_model = jax.tree.map(
            lambda *nodes: jnp.stack(nodes), *target_models, is_leaf=eqx.is_array
        )

    def _sample_from_vi(self, key: PRNGKeyArray):
        """Sample a single model from variational distribution"""
        keys = jr.split(key, len(jax.tree_leaves(self.vi_mean)))
        key_iter = iter(keys)

        def sample_param(mean, logvar):
            if eqx.is_inexact_array(mean):
                std = jnp.exp(0.5 * logvar)
                return mean + std * jr.normal(next(key_iter), mean.shape)
            else:
                return mean

        return jax.tree.map(sample_param, self.vi_mean, self.vi_logvar)

    def __getitem__(self, idx):
        return jax.tree.map(lambda node: node[idx] if eqx.is_array(node) else node, self)

    def __len__(self):
        return self.ensemble_size

    def loss(self, key: PRNGKeyArray, sample):
        """Loss function - same as Bootstrapped"""
        rkey, subkey, key = jr.split(key, 3)

        model_index = jr.randint(subkey, (), 0, self.ensemble_size)
        self = self[model_index]

        target_model = self.target_model
        model = self.model

        if DOUBLE_DQN:
            best_action = model(sample.next_observations.squeeze()).argmax()
            next_target_q_value = (target_model(sample.next_observations.squeeze()))[best_action]
        else:
            next_target_q_value = jnp.max(target_model(sample.next_observations.squeeze()))

        next_q_value = jax.lax.stop_gradient(
            sample.rewards + (1 - sample.dones) * gamma * next_target_q_value
        ).squeeze()

        q_pred = model(sample.observations.squeeze())
        q_pred = q_pred[sample.actions]
        return ((q_pred - next_q_value) ** 2).mean(), (q_pred, next_q_value)

    def action(self, observation: Float[Array, "obs_size"], *_, index=None, **kws):
        assert index is not None
        single_model = jax.tree.map(
            lambda node: node[index] if eqx.is_array(node) else node, self.model
        )
        return single_model(observation).argmax()

    def vi_update(self, key: PRNGKeyArray):
        model_key, target_key, key = jr.split(key, 3)

        def compute_mle_params(ensemble_models):
            def compute_stats(node):
                # Compute empirical mean and variance across ensemble
                if eqx.is_inexact_array(node):
                    mean = jnp.mean(node, axis=0)
                    var = jnp.var(node, axis=0) + 1e-6  # small regularization
                    logvar = jnp.log(var)
                    return mean, logvar
                else:
                    return node, node

            tupled = jax.tree.map(compute_stats, ensemble_models, is_leaf=eqx.is_inexact_array)
            return (
                jax.tree.map(lambda t: t[0], tupled, is_leaf=lambda x: isinstance(x, tuple)),
                jax.tree.map(lambda t: t[1], tupled, is_leaf=lambda x: isinstance(x, tuple)),
            )

        def sample_new_model(key, mean, logvar):
            def sample_array(mean_arr, logvar_arr, key):
                if eqx.is_inexact_array(mean_arr):
                    return (
                        jnp.sqrt(jnp.exp(logvar_arr)) * jr.normal(key, logvar_arr.shape) + mean_arr
                    )
                else:
                    return mean_arr

            return jax.tree.map(sample_array, mean, logvar, keys_like(mean, key))

        new_vi_mean, new_vi_logvar = compute_mle_params(self.model)
        new_vi_logvar = jax.tree.map(
            lambda x: jnp.clip(x, -10, 0) if eqx.is_inexact_array(x) else x,
            new_vi_logvar,
            is_leaf=eqx.is_inexact_array,
        )
        jax.debug.print(
            "{}+-{}",
            new_vi_mean.layers[1].weight[0, 3:4],
            jnp.exp(new_vi_logvar.layers[1].weight[0, 3:4]),
        )

        self = eqx.tree_at(lambda s: s.vi_mean, self, new_vi_mean)
        self = eqx.tree_at(lambda s: s.vi_logvar, self, new_vi_logvar)

        partial = eqx.filter_vmap(
            eqx.Partial(sample_new_model, mean=new_vi_mean, logvar=new_vi_logvar)
        )
        new_models = partial(jr.split(model_key, self.ensemble_size))
        new_target_models = partial(jr.split(target_key, self.ensemble_size))

        self = eqx.tree_at(lambda s: s.model, self, new_models)
        self = eqx.tree_at(lambda s: s.target_model, self, self.model)

        return self


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
