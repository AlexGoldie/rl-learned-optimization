"""Full OPEN optimizer, incorporating all input features and learnable stochasticity"""

from typing import Any, Optional

import flax
import flax.linen as nn
import gin

import jax
from jax import lax
import jax.numpy as jnp
import ipdb
import sys
from rl_optimizer import base as opt_base

sys.path.insert(0, "rl_optimizer/learned_optimization")
from learned_optimization import tree_utils
from learned_optimization.learned_optimizers import (
    common,
)

sys.path.remove("rl_optimizer/learned_optimization")
from optax import adam
import optax


PRNGKey = jnp.ndarray


def _second_moment_normalizer(x, axis, eps=1e-5):
    return x * lax.rsqrt(eps + jnp.mean(jnp.square(x), axis=axis, keepdims=True))


def iter_proportion(iterations, total_its=100000):
    f32 = jnp.float32

    return iterations / f32(total_its)


@flax.struct.dataclass
class GRUOptState:
    params: Any
    rolling_features: common.MomAccumulator
    iteration: jnp.ndarray
    state: Any
    carry: Any


@gin.configurable
class GRU_Opt(opt_base.LearnedOptimizer):
    def __init__(
        self,
        exp_mult=0.001,
        step_mult=0.001,
        hidden_size=16,
        gru_features=8,
    ):

        super().__init__()
        self._step_mult = step_mult
        self._exp_mult = exp_mult

        self.gru_features = gru_features

        self._gru = nn.GRUCell(features=self.gru_features)

        self._mod = nn.Sequential(
            [
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(hidden_size),
                nn.LayerNorm(),
                nn.relu,
                nn.Dense(3),
            ]
        )

    def init(self, key: PRNGKey) -> opt_base.MetaParams:
        # There are 19 features used as input. For now, hard code this.
        key = jax.random.split(key, 5)

        proxy_carry = self._gru.initialize_carry(key[4], (1,))

        return {
            "params": self._mod.init(key[0], jnp.zeros([self.gru_features])),
            "gru_params": self._gru.init(key[2], proxy_carry, jnp.zeros([19])),
        }

    def opt_fn(
        self, theta: opt_base.MetaParams, is_training: bool = False
    ) -> opt_base.Optimizer:
        # ALL MOMENTUM TIMESCALES
        decays = jnp.asarray([0.1, 0.5, 0.9, 0.99, 0.999, 0.9999])

        mod = self._mod
        gru = self._gru
        exp_mult = self._exp_mult
        step_mult = self._step_mult

        theta_mlp = theta["params"]
        theta_gru = theta["gru_params"]

        class _Opt(opt_base.Optimizer):
            def init(
                self,
                params: opt_base.Params,
                model_state: Any = None,
                num_steps: Optional[int] = None,
                key: Optional[PRNGKey] = None,
            ) -> GRUOptState:
                """Initialize opt state."""

                param_tree = jax.tree_util.tree_structure(params)

                keys = jax.random.split(key, param_tree.num_leaves)
                keys = jax.tree_util.tree_unflatten(param_tree, keys)

                carry = jax.tree_util.tree_map(
                    lambda p, k: gru.initialize_carry(k, jnp.expand_dims(p, -1).shape),
                    params,
                    keys,
                )

                return GRUOptState(
                    params=params,
                    state=model_state,
                    rolling_features=common.vec_rolling_mom(decays).init(params),
                    iteration=jnp.asarray(0, dtype=jnp.int32),
                    carry=carry,
                )

            def update(
                self,
                opt_state_actor: GRUOptState,
                crit_opt_state: GRUOptState,
                grad: Any,
                activations: float,
                key: Optional[PRNGKey] = None,
                training_prop=0,
                batch_prop=0,
                config=None,
                layer_props=None,
                model_state: Any = None,
                mask=None,
            ) -> GRUOptState:

                next_rolling_features_actor = common.vec_rolling_mom(decays).update(
                    opt_state_actor.rolling_features, grad["actor"]
                )

                next_rolling_features_critic = common.vec_rolling_mom(decays).update(
                    crit_opt_state.rolling_features, grad["critic"]
                )

                rolling_features = {
                    "actor": next_rolling_features_actor.m,
                    "critic": next_rolling_features_critic.m,
                }

                training_step_feature = training_prop
                batch_feature = batch_prop
                eps1 = 1e-13
                eps2 = 1e-8
                t = opt_state_actor.iteration + 1

                def _update_tensor(p, g, mom, k, dorm, carry, layer_prop, mask):

                    # this doesn't work with scalar parameters, so let's reshape.
                    if not p.shape:
                        p = jnp.expand_dims(p, 0)

                        # use gradient conditioning (Optim4RL)
                        gsign = jnp.expand_dims(jnp.sign(g), 0)
                        glog = jnp.expand_dims(jnp.log(jnp.abs(g) + eps1), 0)

                        mom = jnp.expand_dims(mom, 0)
                        did_reshape = True
                    else:
                        gsign = jnp.sign(g)
                        glog = jnp.log(jnp.abs(g) + eps1)
                        did_reshape = False

                    inps = []
                    inp_g = []

                    batch_gsign = jnp.expand_dims(gsign, axis=-1)
                    batch_glog = jnp.expand_dims(glog, axis=-1)

                    # feature consisting of raw parameter values
                    batch_p = jnp.expand_dims(p, axis=-1)
                    inps.append(batch_p)

                    # feature consisting of all momentum values
                    momsign = jnp.sign(mom)
                    momlog = jnp.log(jnp.abs(mom) + eps1)
                    inps.append(momsign)
                    inps.append(momlog)

                    inp_stack = jnp.concatenate(inps, axis=-1)

                    axis = list(range(len(p.shape)))

                    inp_stack_g = jnp.concatenate(
                        [inp_stack, batch_gsign, batch_glog], axis=-1
                    )

                    inp_stack_g = _second_moment_normalizer(inp_stack_g, axis=axis)

                    # once normalized, add features that are constant across tensor.
                    # namly the training proportion, batch proportion, parameter value and dormancy
                    def stack_tensors(feature, input):

                        stacked = jnp.reshape(
                            feature, [1] * len(axis) + list(feature.shape[-1:])
                        )
                        stacked = jnp.tile(stacked, list(p.shape) + [1])
                        return jnp.concatenate([input, stacked], axis=-1)

                    inp = jnp.tile(
                        jnp.reshape(
                            training_step_feature,
                            [1] * len(axis) + list(training_step_feature.shape[-1:]),
                        ),
                        list(p.shape) + [1],
                    )

                    stacked_batch_prop = jnp.tile(
                        jnp.reshape(
                            batch_feature,
                            [1] * len(axis) + list(batch_feature.shape[-1:]),
                        ),
                        list(p.shape) + [1],
                    )

                    layer_prop = jnp.expand_dims(layer_prop, 0)

                    stacked_layer_prop = jnp.tile(
                        jnp.reshape(
                            layer_prop, [1] * len(axis) + list(layer_prop.shape[-1:])
                        ),
                        list(p.shape) + [1],
                    )

                    inp = jnp.concatenate([inp, stacked_layer_prop], axis=-1)

                    inp = jnp.concatenate([inp, stacked_batch_prop], axis=-1)

                    batch_dorm = jnp.expand_dims(dorm, axis=-1)

                    if p.shape != dorm.shape:
                        batch_dorm = jnp.tile(
                            batch_dorm, [p.shape[0]] + len(axis) * [1]
                        )

                    inp = jnp.concatenate([inp, batch_dorm], axis=-1)

                    inp_g = jnp.concatenate([inp_stack_g, inp], axis=-1)

                    gru_new_carry, gru_out = gru.apply(theta_gru, carry, inp_g)

                    # apply the per parameter MLP.
                    output = mod.apply(theta_mlp, gru_out)

                    update_ = (
                        output[..., 0] * step_mult * jnp.exp(output[..., 1] * exp_mult)
                    )

                    # Add the stochasticity *only* to the actor (using the mask)
                    update = (
                        update_
                        + output[..., 2]
                        * mask
                        * jax.random.normal(k, shape=update_.shape)
                        * step_mult
                    )

                    update = update.reshape(p.shape)

                    return (update, gru_new_carry)

                full_params = {
                    "actor": opt_state_actor.params,
                    "critic": crit_opt_state.params,
                }
                param_tree = jax.tree_util.tree_structure(full_params)

                keys = jax.random.split(key, param_tree.num_leaves)
                keys = jax.tree_util.tree_unflatten(param_tree, keys)

                activations = jax.tree_util.tree_flatten(activations)[0]

                activations = jax.tree_util.tree_unflatten(param_tree, activations)

                def calc_dormancy(tensor_activations):
                    tensor_activations = tensor_activations + 1e-11
                    total_activations = jnp.abs(tensor_activations).sum(axis=-1)
                    total_activations = jnp.tile(
                        jnp.expand_dims(total_activations, -1),
                        tensor_activations.shape[-1],
                    )
                    dormancy = (
                        tensor_activations
                        / total_activations
                        * tensor_activations.shape[-1]
                    )
                    return dormancy

                dormancies = jax.tree_util.tree_map(calc_dormancy, activations)

                full_carry = {
                    "actor": opt_state_actor.carry,
                    "critic": crit_opt_state.carry,
                }

                updates_carry = jax.tree_util.tree_map(
                    _update_tensor,
                    full_params,
                    grad,
                    rolling_features,
                    keys,
                    dormancies,
                    full_carry,
                    layer_props,
                    mask,
                )

                updates_carry_leaves = jax.tree_util.tree_leaves(updates_carry)
                updates = [
                    updates_carry_leaves[i]
                    for i in range(0, len(updates_carry_leaves), 2)
                ]
                new_carry = [
                    updates_carry_leaves[i + 1]
                    for i in range(0, len(updates_carry_leaves), 2)
                ]

                updates = jax.tree_util.tree_unflatten(param_tree, updates)
                new_carry = jax.tree_util.tree_unflatten(param_tree, new_carry)

                # Make update globally 0
                updates_flat = jax.flatten_util.ravel_pytree(updates)[0]
                update_mean = updates_flat.mean()
                update_mean = jax.tree_util.tree_unflatten(
                    param_tree, jnp.tile(update_mean, param_tree.num_leaves)
                )

                updates = jax.tree_util.tree_map(
                    lambda x, mu: x - mu, updates, update_mean
                )

                def param_update(p, update):

                    new_param = p - update

                    return new_param

                next_params = jax.tree_util.tree_map(param_update, full_params, updates)

                # For simplicity, maitain different opt states between the actor and the critic
                next_opt_state_actor = GRUOptState(
                    params=tree_utils.match_type(
                        next_params["actor"], opt_state_actor.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_actor, opt_state_actor.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                    carry=new_carry["actor"],
                )

                next_opt_state_critic = GRUOptState(
                    params=tree_utils.match_type(
                        next_params["critic"], crit_opt_state.params
                    ),
                    rolling_features=tree_utils.match_type(
                        next_rolling_features_critic, crit_opt_state.rolling_features
                    ),
                    iteration=opt_state_actor.iteration + 1,
                    state=model_state,
                    carry=new_carry["critic"],
                )

                param_flat_actor = jax.flatten_util.ravel_pytree(next_params["actor"])[
                    0
                ]
                if config["VISUALISE"]:
                    param_mean_actor = jnp.mean(jnp.array(param_flat_actor))
                param_abs_mean_actor = jnp.mean(jnp.abs(jnp.array(param_flat_actor)))

                param_flat_critic = jax.flatten_util.ravel_pytree(
                    next_params["critic"]
                )[0]
                if config["VISUALISE"]:
                    param_mean_critic = jnp.mean(jnp.array(param_flat_critic))
                param_abs_mean_critic = jnp.mean(jnp.abs(jnp.array(param_flat_critic)))

                if config["VISUALISE"]:
                    return (
                        next_opt_state_actor,
                        next_opt_state_critic,
                        (param_mean_actor, param_abs_mean_actor),
                        (param_mean_critic, param_abs_mean_critic),
                    )
                else:
                    return (
                        next_opt_state_actor,
                        next_opt_state_critic,
                        (param_abs_mean_actor),
                        (param_abs_mean_critic),
                    )

        return _Opt()
