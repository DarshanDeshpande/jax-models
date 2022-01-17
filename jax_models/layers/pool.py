import jax.numpy as jnp
import flax.linen as nn

from typing import Union, Iterable


class AdaptiveAveragePool1D(nn.Module):
    output_size: Union[int, Iterable[int]]

    @nn.compact
    def __call__(self, inputs):
        output_size = (
            (self.output_size,)
            if isinstance(self.output_size, int)
            else self.output_size
        )
        split = jnp.split(inputs, output_size[0], axis=1)
        stack = jnp.stack(split, axis=1)
        return jnp.mean(stack, axis=2)
