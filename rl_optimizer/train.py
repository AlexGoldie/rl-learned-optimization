"""training loop for OPEN on non-gridworld environments"""

import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import flax
import distrax
import gymnax
from rl_optimizer.utils import (
    GymnaxGymWrapper,
    GymnaxLogWrapper,
    FlatWrapper,
    BraxGymnaxWrapper,
    ClipAction,
    TransformObservation,
    NormalizeObservation,
    NormalizeReward,
    VecEnv,
)

from brax.envs.wrappers.gym import GymWrapper
from brax import envs
from evosax import OpenES, ParameterReshaper, FitnessShaper, CMA_ES, SNES, SimpleGA
from evosax.utils import ESLog

import os
import os.path as osp
from datetime import datetime
from rl_optimizer.network import GRU_Opt
from gymnax.environments import spaces

from tqdm import tqdm
import wandb
import time
from optax import adam
from functools import partial

import argparse
from rl_optimizer.configs import all_configs

from rl_optimizer.eval import make_plot

sys.path.insert(0, "rl_optimizer/groove")

import environments.gridworld.gridworld as grid
import environments.gridworld.configs as grid_conf
from environments.gridworld.configs import ENV_MODE_KWARGS

sys.path.remove("rl_optimizer/groove")


class Actor(nn.Module):
    action_dim: Sequence[int]
    config: dict

    @nn.compact
    def __call__(self, x):
        hsize = self.config["HSIZE"]
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean_activation_1 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }

        actor_mean = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean_activation_2 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean_activation_3 = {
            "kernel": jnp.mean(jnp.abs(actor_mean), axis=0),
            "bias": jnp.mean(jnp.abs(actor_mean), axis=0),
        }

        if self.config["CONTINUOUS"]:
            actor_logtstd = self.param(
                "log_std", nn.initializers.zeros, (self.action_dim,)
            )
            actor_mean_activation_4 = jnp.expand_dims(
                jnp.mean(jnp.exp(actor_logtstd), axis=0), axis=0
            )
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            pi = distrax.Categorical(logits=actor_mean)

        if self.config["CONTINUOUS"]:
            activations = (
                actor_mean_activation_1,
                actor_mean_activation_2,
                actor_mean_activation_3,
                actor_mean_activation_4,
            )
        else:
            activations = (
                actor_mean_activation_1,
                actor_mean_activation_2,
                actor_mean_activation_3,
            )

        return pi, activations


class Critic(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x):
        hsize = self.config["HSIZE"]
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic_mean_activation_1 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }
        critic = nn.Dense(
            hsize, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic_mean_activation_2 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        critic_mean_activation_3 = {
            "kernel": jnp.mean(jnp.abs(critic), axis=0),
            "bias": jnp.mean(jnp.abs(critic), axis=0),
        }

        activations = (
            critic_mean_activation_1,
            critic_mean_activation_2,
            critic_mean_activation_3,
        )
        return jnp.squeeze(critic, axis=-1), activations


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def symlog(x):
    return jnp.sign(x) * jnp.log(jnp.abs(x) + 1)


gridtypes = [
    "sixteen_rooms",
    "labyrinth",
    "rand_sparse",
    "rand_dense",
    "rand_long",
    "standard_maze",
    "rand_all",
]


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["TOTAL_UPDATES"] = (
        config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    )

    @partial(jax.jit, static_argnames=["grid_type"])
    def train(meta_params, rng, grid_type=6):

        if "Brax-" in config["ENV_NAME"]:
            name = config["ENV_NAME"].split("Brax-")[1]
            env, env_params = BraxGymnaxWrapper(env_name=name), None
            if config.get("CLIP_ACTION"):
                env = ClipAction(env)
            env = GymnaxLogWrapper(env)
            if config.get("SYMLOG_OBS"):
                env = TransformObservation(env, transform_obs=symlog)

            env = VecEnv(env)
            if config.get("NORMALIZE"):
                env = NormalizeObservation(env)
                env = NormalizeReward(env, config["GAMMA"])
            actor = Actor(env.action_space(env_params).shape[0], config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space(env_params).shape)
        else:
            # INIT ENV
            if config["ENV_NAME"] == "gridworld":
                env = grid.GridWorld(**ENV_MODE_KWARGS[gridtypes[grid_type]])
                rng, rng_ = jax.random.split(rng)
                env_params = grid_conf.reset_env_params(rng_, gridtypes[grid_type])
            else:
                env, env_params = gymnax.make(config["ENV_NAME"])
            env = GymnaxGymWrapper(env, env_params, config)
            env = FlatWrapper(env)

            env = GymnaxLogWrapper(env)
            env = VecEnv(env)
            actor = Actor(env.action_space, config=config)
            critic = Critic(config=config)
            init_x = jnp.zeros(env.observation_space)

        # INIT NETWORK

        rng, _rng = jax.random.split(rng)
        actor_params = actor.init(_rng, init_x)
        critic_params = critic.init(_rng, init_x)

        if config["larger"]:
            meta_opt = GRU_Opt(hidden_size=32, gru_features=16)
        else:
            meta_opt = GRU_Opt(hidden_size=16, gru_features=8)
        opt = meta_opt.opt_fn(meta_params)
        clip_opt = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])

        rng, rng_act, rng_crit = jax.random.split(rng, 3)
        train_state_actor = opt.init(actor_params, key=rng_act)
        train_state_critic = opt.init(critic_params, key=rng_crit)

        act_param_tree = jax.tree_util.tree_structure(train_state_actor.params)
        act_layer_props = []
        num_act_layers = len(train_state_actor.params["params"])
        for i, layer in enumerate(train_state_actor.params["params"]):
            layer_prop = i / (num_act_layers - 1)
            if type(train_state_actor.params["params"][layer]) == dict:
                act_layer_props.extend(
                    [layer_prop] * len(train_state_actor.params["params"][layer])
                )
            else:
                act_layer_props.extend([layer_prop])

        act_layer_props = jax.tree_util.tree_unflatten(act_param_tree, act_layer_props)

        crit_param_tree = jax.tree_util.tree_structure(train_state_critic.params)
        crit_layer_props = []
        num_crit_layers = len(train_state_critic.params["params"])
        for i, layer in enumerate(train_state_critic.params["params"]):
            layer_prop = i / (num_crit_layers - 1)
            if type(train_state_critic.params["params"][layer]) == dict:
                crit_layer_props.extend(
                    [layer_prop] * len(train_state_critic.params["params"][layer])
                )
            else:
                crit_layer_props.extend([layer_prop])

        crit_layer_props = jax.tree_util.tree_unflatten(
            crit_param_tree, crit_layer_props
        )

        # INIT ENV
        all_rng = jax.random.split(_rng, config["NUM_ENVS"] + 1)
        rng, _rng = all_rng[0], all_rng[1:]
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state_actor,
                    train_state_critic,
                    env_state,
                    last_obs,
                    last_done,
                    rng,
                    last_param_abs,
                ) = runner_state
                rng, _rng = jax.random.split(rng)
                # SELECT ACTION
                pi, _ = actor.apply(train_state_actor.params, last_obs)
                value, _ = critic.apply(train_state_critic.params, last_obs)
                action = pi.sample(seed=_rng)

                log_prob = pi.log_prob(action)
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state_actor,
                    train_state_critic,
                    env_state,
                    obsv,
                    done,
                    rng,
                    0.0,
                )

                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state_actor,
                train_state_critic,
                env_state,
                last_obs,
                last_done,
                rng,
                last_param_abs,
            ) = runner_state
            last_val, _ = critic.apply(train_state_critic.params, last_obs)
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_key, batch_info):
                    train_state_actor, train_state_critic, key = train_state_key
                    key, key_ = jax.random.split(key)

                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(actor_params, critic_params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, actor_activations = actor.apply(
                            actor_params, traj_batch.obs
                        )
                        value, critic_activations = critic.apply(
                            critic_params, traj_batch.obs
                        )

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )

                        return total_loss, (actor_activations, critic_activations)

                    training_prop = (
                        train_state_actor.iteration
                        // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                    ) / (config["NUM_UPDATES"] - 1)
                    batch_prop = (
                        (train_state_actor.iteration // config["NUM_MINIBATCHES"])
                        % config["UPDATE_EPOCHS"]
                    ) / (config["UPDATE_EPOCHS"] - 1)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True, argnums=[0, 1])
                    (total_loss, (actor_activations, critic_activations)), (
                        actor_grads,
                        critic_grads,
                    ) = grad_fn(
                        train_state_actor.params,
                        train_state_critic.params,
                        traj_batch,
                        advantages,
                        targets,
                    )

                    key_actor, key_critic = jax.random.split(key_)
                    actor_grads, _ = clip_opt.update(actor_grads, None)
                    critic_grads, _ = clip_opt.update(critic_grads, None)
                    actor_mask = {"kernel": 1, "bias": 1}
                    critic_mask = {"kernel": 0, "bias": 0}

                    # FOR NOW, HARD CODE MASK
                    if config["CONTINUOUS"]:
                        mask = {
                            "actor": {
                                "params": {
                                    "Dense_0": actor_mask,
                                    "Dense_1": actor_mask,
                                    "Dense_2": actor_mask,
                                    "log_std": 1,
                                }
                            },
                            "critic": {
                                "params": {
                                    "Dense_0": critic_mask,
                                    "Dense_1": critic_mask,
                                    "Dense_2": critic_mask,
                                }
                            },
                        }

                    else:

                        mask = {
                            "actor": {
                                "params": {
                                    "Dense_0": actor_mask,
                                    "Dense_1": actor_mask,
                                    "Dense_2": actor_mask,
                                }
                            },
                            "critic": {
                                "params": {
                                    "Dense_0": critic_mask,
                                    "Dense_1": critic_mask,
                                    "Dense_2": critic_mask,
                                }
                            },
                        }

                    activations = {
                        "actor": actor_activations,
                        "critic": critic_activations,
                    }
                    grads = {"actor": actor_grads, "critic": critic_grads}
                    layer_props = {"actor": act_layer_props, "critic": crit_layer_props}

                    # APPLY OPTIMIZER
                    (
                        train_state_actor,
                        train_state_critic,
                        actor_updates,
                        critic_updates,
                    ) = opt.update(
                        train_state_actor,
                        train_state_critic,
                        grads,
                        activations,
                        key=key_actor,
                        training_prop=training_prop,
                        config=config,
                        batch_prop=batch_prop,
                        layer_props=layer_props,
                        mask=mask,
                    )

                    train_state_key_ = (train_state_actor, train_state_critic, key)
                    return train_state_key_, (
                        total_loss,
                        actor_updates,
                        critic_updates,
                        actor_activations,
                        critic_activations,
                    )

                (
                    train_state_actor,
                    train_state_critic,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state_key = (train_state_actor, train_state_critic, rng)
                train_state_key, total_loss_updates = jax.lax.scan(
                    _update_minbatch, train_state_key, minibatches
                )
                train_state_actor, train_state_critic, rng = train_state_key
                update_state = (
                    train_state_actor,
                    train_state_critic,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )

                return update_state, total_loss_updates

            update_state = (
                train_state_actor,
                train_state_critic,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_update = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state_actor = update_state[0]
            train_state_critic = update_state[1]

            (
                loss_info,
                actor_updates,
                critic_updates,
                actor_activations,
                critic_activations,
            ) = loss_update
            if config["VISUALISE"]:

                metric = traj_batch.info

            else:
                metric = dict()

                metric.update(
                    {
                        "returned_episode_returns": traj_batch.info[
                            "returned_episode_returns"
                        ][-1].mean()
                    }
                )

            rng = update_state[-1]

            actor_size = len(jax.flatten_util.ravel_pytree(runner_state[0].params)[0])
            critic_size = len(jax.flatten_util.ravel_pytree(runner_state[1].params)[0])
            if config["VISUALISE"]:
                actor_abs = jax.flatten_util.ravel_pytree(actor_updates[1])[0]
                critic_abs = jax.flatten_util.ravel_pytree(actor_updates[1])[0]
            else:
                actor_abs = jax.flatten_util.ravel_pytree(actor_updates)[0][-1]
                critic_abs = jax.flatten_util.ravel_pytree(actor_updates)[0][-1]
            abs_param_mean = (actor_size * actor_abs + critic_size * critic_abs) / (
                actor_size + critic_size
            )
            runner_state = (
                train_state_actor,
                train_state_critic,
                env_state,
                last_obs,
                last_done,
                rng,
                abs_param_mean.mean(),
            )
            if config["VISUALISE"]:

                # CALCULATE DORMANCY FOR TRACKING
                def calc_dormancy(tensor_activations, tau=1e-7):
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
                    tau_dormancy = dormancy < tau
                    tau_dormancy = tau_dormancy.sum(axis=(-1))
                    return tau_dormancy

                full_activations = (actor_activations, critic_activations)
                dormancies = jax.tree_util.tree_map(calc_dormancy, full_activations)
                dormancies = jax.tree_util.tree_flatten(dormancies)[0]

                prop_dormancies = jnp.stack(dormancies).sum(axis=0)
                prop_dormancies = (
                    prop_dormancies
                    / len(jax.flatten_util.ravel_pytree(full_activations)[0])
                    * prop_dormancies.shape[0]
                    * prop_dormancies.shape[1]
                )
                param_mean = (
                    actor_size * jax.flatten_util.ravel_pytree(actor_updates[0])[0]
                    + critic_size * jax.flatten_util.ravel_pytree(critic_updates[0])[0]
                ) / (actor_size + critic_size)
                log_updates = {
                    "param_mean": param_mean,
                    "param_abs_mean": abs_param_mean,
                }
                metric.update({"dormancy": prop_dormancies})
                metric.update(log_updates)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_actor,
            train_state_critic,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            _rng,
            0.0,
        )

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return runner_state, metric

    return train


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-generations", type=int, default=512)
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--popsize", type=int, default=64)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--save-every-k", type=int, default=24)
    parser.add_argument("--noise-level", type=float, default=0.03)
    parser.add_argument("--pmap", default=False, action="store_true")
    parser.add_argument("--no-pmap", dest="pmap", action="store_false")
    parser.add_argument("--wandb-name", type=str, default="OPEN")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--larger", default=False, action="store_true")

    args = parser.parse_args()

    evo_config = {
        "ENV_NAME": args.envs,
        "POPULATION_SIZE": args.popsize,
        "NUM_GENERATIONS": args.num_generations,
        "NUM_ROLLOUTS": args.num_rollouts,
        "SAVE_EVERY_K": args.save_every_k,
        "NOISE_LEVEL": args.noise_level,
        "PMAP": args.pmap,
        "LR": args.lr,
        "num_GPUs": jax.local_device_count(),
    }

    all_configs = {k: all_configs[k] for k in evo_config["ENV_NAME"]}

    save_loc = "training"
    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    save_dir = f"{save_loc}/{str(datetime.now()).replace(' ', '_')}_optimizer"
    os.mkdir(f"{save_dir}")

    popsize = args.popsize
    num_generations = args.num_generations
    num_rollouts = args.num_rollouts
    save_every_k_gens = args.save_every_k

    wandb.init(
        project="OPEN",
        config=evo_config,
        name=args.wandb_name,
        entity=args.wandb_entity,
    )

    if args.larger:
        meta_opt = GRU_Opt(hidden_size=32, gru_features=16)
    else:
        meta_opt = GRU_Opt(hidden_size=16, gru_features=8)
    params = meta_opt.init(jax.random.PRNGKey(0))
    param_reshaper = ParameterReshaper(params)

    def make_rollout(train_fn):
        def single_rollout(rng_input, meta_params):
            params, metrics = train_fn(meta_params, rng_input)

            fitness = metrics["returned_episode_returns"][-1]
            return (fitness, metrics["returned_episode_returns"][-1])

        vmap_rollout = jax.vmap(single_rollout, in_axes=(0, None))
        rollout = jax.jit(jax.vmap(vmap_rollout, in_axes=(0, param_reshaper.vmap_dict)))

        if evo_config["PMAP"]:
            rollout = jax.pmap(rollout)

        return rollout

    for k in all_configs.keys():
        all_configs[k]["NUM_UPDATES"] = (
            all_configs[k]["TOTAL_TIMESTEPS"]
            // all_configs[k]["NUM_STEPS"]
            // all_configs[k]["NUM_ENVS"]
        )
        all_configs[k]["larger"] = args.larger

    rollouts = {k: make_rollout(make_train(v)) for k, v in all_configs.items()}

    rng = jax.random.PRNGKey(42)
    strategy = OpenES(
        popsize=popsize,
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        lrate_init=evo_config["LR"],
        sigma_init=evo_config["NOISE_LEVEL"],
        sigma_decay=args.sigma_decay,
        lrate_decay=args.lr_decay,
    )
    es_params = strategy.default_params

    es_logging = ESLog(
        pholder_params=params, num_generations=num_generations, top_k=5, maximize=True
    )
    log = es_logging.initialize()
    fit_shaper = FitnessShaper(
        centered_rank=True, z_score=False, w_decay=0.0, maximize=True
    )
    fit_shaper_multi = FitnessShaper(
        centered_rank=True, z_score=False, w_decay=0.0, maximize=False
    )

    state = strategy.initialize(rng, es_params)

    most_neg = {env: 0 for env in evo_config["ENV_NAME"]}

    fit_history = []
    for gen in tqdm(range(num_generations)):

        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, state = jax.jit(strategy.ask)(rng_ask, state, es_params)

        # Set up antithetic task sampling by repeating each optimizer.
        if args.envs == ["gridworld"]:
            x, state = jax.jit(strategy.ask)(rng_ask, state, es_params)
            new_orders = [
                [i, int(args.popsize / 2) + i] for i in range(int(args.popsize / 2))
            ]
            new_orders = jnp.array([x for y in new_orders for x in y])
            x = x[new_orders]

        reshaped_params = param_reshaper.reshape(x)
        fit_info = {}

        fit_info[f"meta learning rate"] = state.opt_state.lrate
        fit_info[f"meta noise sigma"] = state.sigma

        all_fitness = []

        for env in args.envs:

            rng, rng_eval = jax.random.split(rng)

            all_configs[env]["VISUALISE"] = False
            rollout = rollouts[env]

            # Antithetic task sampling for gridworld - antithetic perturbatiosn are evaluated on the same rng
            if args.envs == ["gridworld"]:
                batch_rng = jax.random.split(rng_eval, args.popsize / 2)
                batch_rng = jnp.repeat(batch_rng, 2, axis=0)

            else:
                batch_rng = jax.random.split(rng_eval, num_rollouts)
                batch_rng = jnp.tile(batch_rng, (args.popsize, 1, 1))

            if args.pmap:
                batch_rng_pmap = jnp.reshape(
                    batch_rng, (jax.local_device_count(), -1, num_rollouts, 2)
                )
                fitness, unreg_fitness = rollout(batch_rng_pmap, reshaped_params)

                fitness = fitness.reshape(-1, evo_config["NUM_ROLLOUTS"]).mean(axis=1)
                unreg_fitness = unreg_fitness.reshape(
                    -1, evo_config["NUM_ROLLOUTS"]
                ).mean(axis=1)
            else:
                batch_rng = jnp.reshape(batch_rng, (-1, num_rollouts, 2))
                fitness, unreg_fitness = rollout(batch_rng, reshaped_params)
                fitness = fitness.mean(axis=1)
                unreg_fitness = unreg_fitness.mean(axis=1)

            print(f"fitness:       {fitness}")

            fit_info[f"{env}/fitness_notnorm_{env}"] = jnp.mean(fitness)
            fit_info[f"{env}/best_fitness_{env}"] = jnp.max(fitness)
            fit_info[f"{env}/worst_fitness{env}"] = jnp.min(fitness)

            fitness = jnp.nan_to_num(fitness, nan=-100000)
            print(f"mean fitness_{env}  =   {jnp.mean(fitness):.3f}")

            fit_re = fit_shaper.apply(x, fitness)

            print(f"fitness_spread at gen {gen} is {fitness.max()-fitness.min()}")
            log = es_logging.update(log, x, fitness)
            print(
                f"Generation: {gen}, Best: {log['log_top_1'][gen]}, Fitness: {fitness.mean()}"
            )
            fit_history.append(fitness.mean())

            fitness_var = jnp.var(fitness)
            dispersion = fitness_var / jnp.abs(fitness.mean())

            param_sum = state.mean.sum()
            param_abs_sum = jnp.abs(state.mean).sum()
            param_abs_mean = jnp.abs(state.mean).mean()
            fitness_spread = fitness.max() - fitness.min()

            fit_norm = fitness / all_configs[env]["PPO_TEMP"]

            mean_norm = fit_norm.mean()

            wandb.log(
                {
                    f"{env}/avg_fitness": fitness.mean(),
                    f"{env}/fitness_histo_{env}": wandb.Histogram(fitness, num_bins=16),
                    f"{env}/fitness_spread_{env}": fitness_spread,
                    f"{env}/fitness_variance": fitness_var,
                    f"{env}_norm_fit": mean_norm,
                    f"{env}_norm_histo": wandb.Histogram(fit_norm, num_bins=16),
                    "dispersion_coeff": dispersion,
                    **fit_info,
                }
            )

            all_fitness.append(fit_norm)

        fitnesses = jnp.stack(all_fitness, axis=0)

        fitnesses_mean = jnp.mean(fitnesses, axis=0)

        wandb.log(
            {
                "average normalised fit": fitnesses_mean.mean(),
                "average_histo": wandb.Histogram(fitnesses_mean, num_bins=16),
            }
        )

        # normalization for antithetic task sampling
        if args.envs == ["gridworld"]:
            first_greater = jnp.greater(fitness[::2], fitness[1::2])
            rank_fitness = jnp.zeros_like(fitness)
            rank_fitness = rank_fitness.at[::2].set(-1 * first_greater.astype(float))
            fitness_rerank = rank_fitness.at[1::2].set(
                first_greater.astype(float) - 1.0
            )
        else:
            fitness_rerank = fit_shaper.apply(x, fitnesses_mean)

        state = jax.jit(strategy.tell)(x, fitness_rerank, state, es_params)

        if gen % save_every_k_gens == 0:
            print("SAVING!")
            jnp.save(osp.join(save_dir, f"curr_param_{gen}.npy"), state.mean)
            np.save(osp.join(save_dir, f"fit_history.npy"), np.array(fit_history))

            wandb.save(
                osp.join(save_dir, f"curr_param_{gen}.npy"),
                base_path=save_dir,
            )
            if args.envs == ["gridworld"]:
                plot_envs = gridtypes
                grids = True
            else:
                plot_envs = args.envs
                grids = False

            time.sleep(1)
            make_plot(
                exp_names=wandb.run.id,
                exp_nums=gen,
                envs=plot_envs,
                num_runs=8,
                pmap=args.pmap,
                larger=args.larger,
                training=True,
                grids=grids,
            )
