import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces, environment
from typing import NamedTuple, Optional, Tuple, Union
from brax import envs
from flax import struct
from functools import partial
from gymnax.wrappers.purerl import GymnaxWrapper
import chex
from jax import vjp, flatten_util
from jax.tree_util import tree_flatten


class GymnaxGymWrapper:
    def __init__(self, env, env_params, config):
        self.env = env
        self.env_params = env_params
        if config["CONTINUOUS"]:
            self.action_space = env.action_space(self.env_params).shape[0]
        else:
            self.action_space = env.action_space(self.env_params).n

        self.observation_space = env.observation_space(self.env_params).shape

    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        obs, env_state = self.env.reset(_rng, self.env_params)
        state = (env_state, rng)
        return obs, state

    def step(self, key, state, action, params=None):
        env_state, rng = state
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = self.env.step(
            _rng, env_state, action, self.env_params
        )
        state = (env_state, rng)
        return obs, state, reward, done, info


class MetaGymnaxGymWrapper:
    def __init__(self, env, env_param_generator):
        self.env = env
        self.env_param_generator = env_param_generator
        self.observation_space = env.observation_space(self.env_params).shape
        self.action_space = env.action_space(self.env_params).n

    def reset(self, rng):
        rng, _rng = jax.random.split(rng)
        env_params = self.env_param_generator(_rng)
        rng, _rng = jax.random.split(rng)
        obs, env_state = self.env.reset(_rng, env_params)
        state = (env_state, env_params, rng)
        return obs, state

    def step(self, state, action):
        env_state, env_params, rng = state
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = self.env.step(
            _rng, env_state, action, env_params
        )
        state = (env_state, env_params, rng)
        return obs, state, reward, done, info


class FlatWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = np.prod(env.observation_space)
        self.action_space = env.action_space

    def reset(self, rng, params=None):
        obs, env_state = self.env.reset(rng, params)
        obs = jnp.reshape(obs, (self.observation_space,))
        return obs, env_state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = jnp.reshape(obs, (self.observation_space,))
        return obs, state, reward, done, info


class EpisodeStats(NamedTuple):
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class GymnaxLogWrapper(GymnaxWrapper):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, rng, params=None):
        obs, env_state = self.env.reset(rng, params)
        state = (env_state, EpisodeStats(0, 0, 0, 0))
        return obs, state

    def step(self, key, state, action, params=None):
        # def step(self, state, action):
        env_state, episode_stats = state
        obs, env_state, reward, done, info = self.env.step(
            key, env_state, action, params
        )
        new_episode_return = episode_stats.episode_returns + reward
        new_episode_length = episode_stats.episode_lengths + 1
        new_episode_stats = EpisodeStats(
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=episode_stats.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=episode_stats.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        state = (env_state, new_episode_stats)
        info = {}
        info["returned_episode_returns"] = new_episode_stats.returned_episode_returns
        info["returned_episode_lengths"] = new_episode_stats.returned_episode_lengths
        return obs, state, reward, done, info


class EvalStats(NamedTuple):
    first_returned_episode_returns: float
    ever_done: bool


class GymnaxLogEvalWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, rng, params=None):
        obs, env_state = self.env.reset(rng, params)
        state = (env_state, EvalStats(0, False))
        return obs, state

    def step(self, key, state, action, params=None):
        env_state, episode_stats = state
        obs, env_state, reward, done, info = self.env.step(
            key, env_state, action, params
        )
        ever_done = jnp.logical_or(episode_stats.ever_done, done)
        episode_return = episode_stats.first_returned_episode_returns + reward * (
            1 - ever_done
        )
        new_episode_stats = EvalStats(
            first_returned_episode_returns=episode_return,
            ever_done=ever_done,
        )
        state = (env_state, new_episode_stats)
        info = {}
        info["first_returned_episode_returns"] = (
            new_episode_stats.first_returned_episode_returns
        )
        info["ever_done"] = new_episode_stats.ever_done
        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        # def __init__(self, env_name, backend="generalized"):

        # ****** BACKEND CURRENTLY NOT IMPLEMENTED

        env = envs.get_environment(env_name=env_name, backend=backend)
        env = envs.wrappers.training.EpisodeWrapper(
            env, episode_length=1000, action_repeat=1
        )
        env = envs.wrappers.training.AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        # print(f'bye: {key}')
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class ClipAction(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.action_lim = config["ACTION_LIM"]

    def step(self, key, state, action, params=None):
        """TODO: FIX"""
        # old_action = action
        action = jnp.clip(action, -1, 1)
        # jax.debug.print('{old} -> {new}', old=old_action, new=action)
        # action = jnp.clip(action, -1.0, 1.0)
        # jax.debug.print('{action}', action=action)
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


# Use parameter Reshaper from old gymnax
def ravel_pytree(pytree):
    leaves, _ = tree_flatten(pytree)
    flat, _ = vjp(ravel_list, *leaves)
    return flat


def ravel_list(*lst):
    return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])


class ParameterReshaper(object):
    def __init__(
        self,
        placeholder_params: Union[chex.ArrayTree, chex.Array],
        n_devices: Optional[int] = None,
        verbose: bool = True,
    ):
        """Reshape flat parameters vectors into generation eval shape."""
        # Get network shape to reshape
        self.placeholder_params = placeholder_params

        # Set total parameters depending on type of placeholder params
        flat, self.unravel_pytree = flatten_util.ravel_pytree(placeholder_params)
        self.total_params = flat.shape[0]
        self.reshape_single = jax.jit(self.unravel_pytree)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1 and verbose:
            print(
                f"ParameterReshaper: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

        if verbose:
            print(
                f"ParameterReshaper: {self.total_params} parameters detected"
                " for optimization."
            )

    def reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Perform reshaping for a 2D matrix (pop_members, params)."""
        vmap_shape = jax.vmap(self.reshape_single)
        if self.n_devices > 1:
            x = self.split_params_for_pmap(x)
            map_shape = jax.pmap(vmap_shape)
        else:
            map_shape = vmap_shape
        return map_shape(x)

    def multi_reshape(self, x: chex.Array) -> chex.ArrayTree:
        """Reshape parameters lying already on different devices."""
        # No reshaping required!
        vmap_shape = jax.vmap(self.reshape_single)
        return jax.pmap(vmap_shape)(x)

    def flatten(self, x: chex.ArrayTree) -> chex.Array:
        """Reshaping pytree parameters into flat array."""
        vmap_flat = jax.vmap(ravel_pytree)
        if self.n_devices > 1:
            # Flattening of pmap paramater trees to apply vmap flattening
            def map_flat(x):
                x_re = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), x)
                return vmap_flat(x_re)

        else:
            map_flat = vmap_flat
        flat = map_flat(x)
        return flat

    def multi_flatten(self, x: chex.Array) -> chex.ArrayTree:
        """Flatten parameters lying remaining on different devices."""
        # No reshaping required!
        vmap_flat = jax.vmap(ravel_pytree)
        return jax.pmap(vmap_flat)(x)

    def split_params_for_pmap(self, param: chex.Array) -> chex.Array:
        """Helper reshapes param (bs, #params) into (#dev, bs/#dev, #params)."""
        return jnp.stack(jnp.split(param, self.n_devices))

    @property
    def vmap_dict(self) -> chex.ArrayTree:
        """Get a dictionary specifying axes to vmap over."""
        vmap_dict = jax.tree_util.tree_map(lambda x: 0, self.placeholder_params)
        return vmap_dict
