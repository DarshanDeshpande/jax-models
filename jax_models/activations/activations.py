import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Iterable, Callable


def relu6(input):
    return jnp.minimum(jnp.maximum(input, 0), 6)


def hardswish(input):
    return input * (relu6(input + 3) / 6)


def mish(input):
    return input * jnp.tanh(nn.softplus(input))


class PReLU(nn.Module):
    alpha_init: Callable = nn.zeros
    shared_axis: Union[Iterable, int] = None

    @nn.compact
    def __call__(self, inputs):
        param_shape = list(inputs.shape[1:])
        if self.shared_axis is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        alpha = self.param("alpha", self.alpha_init, tuple(param_shape))
        pos = nn.relu(inputs)
        neg = -alpha * nn.relu(-inputs)
        return pos + neg
