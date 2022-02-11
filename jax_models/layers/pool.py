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


class AdaptiveAveragePool2D(nn.Module):
    output_size: Union[Iterable, int]

    @nn.compact
    def __call__(self, inputs):
        if isinstance(self.output_size, (list, tuple)):
            h_bins = self.output_size[0]
            w_bins = self.output_size[1]
        else:
            h_bins = w_bins = self.output_size

        split_cols = jnp.split(inputs, h_bins, axis=1)
        split_cols = jnp.stack(split_cols, axis=1)
        split_rows = jnp.split(split_cols, w_bins, axis=3)
        split_rows = jnp.stack(split_rows, axis=3)
        return jnp.mean(split_rows, axis=[2, 4])
