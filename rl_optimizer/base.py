# A MISHMASH OF BASE OBJECTS FROM LEARNED_OPTIMIZERS
import abc
import collections
from typing import Any, Callable, Sequence, Optional, Tuple

import chex
import flax

import jax.numpy as jnp

MetaParamOpt = collections.namedtuple("MetaParamOpt", ["init", "opt_fn"])

PRNGKey = jnp.ndarray
Params = Any
MetaParams = Any


class LearnedOptimizer(abc.ABC):
    """Base class for learned optimizers."""

    @abc.abstractmethod
    def init(self, key: PRNGKey) -> MetaParams:
        raise NotImplementedError()

    @abc.abstractmethod
    def opt_fn(self, theta: MetaParams, is_training: bool = False):
        raise NotImplementedError()

    @property
    def name(self):
        return None


ModelState = Any
Params = Any
Gradient = Params
OptState = Any


@flax.struct.dataclass
class StatelessState:
    params: chex.ArrayTree
    state: chex.ArrayTree


class Optimizer(abc.ABC):
    """Baseclass for the Optimizer interface."""

    def get_params(self, state: OptState) -> Params:
        return state.params

    def get_state(self, state: OptState) -> ModelState:
        return state.state

    def get_params_state(self, state: OptState) -> Tuple[Params, ModelState]:
        return self.get_params(state), self.get_state(state)

    def init(
        self,
        params: Params,
        state: Optional[ModelState] = None,
        num_steps: Optional[int] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ) -> OptState:
        raise NotImplementedError

    def set_params(self, state: OptState, params: Params) -> OptState:
        return state.replace(params=params)

    def update(
        self,
        opt_state: OptState,
        grad: Gradient,
        model_state: Optional[ModelState] = None,
        key: Optional[chex.PRNGKey] = None,
        **kwargs,
    ) -> OptState:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        """Name of optimizer.
        This property is used when serializing results / baselines. This should
        lead with the class name, and follow with all parameters used to create
        the object. For example: "<ClassName>_<param1><value>_<param2><value>"
        """
        return "UnnamedOptimizer"
