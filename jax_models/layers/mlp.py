import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D
from typing import Optional, Callable


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    use_dwconv: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)

        if self.use_dwconv:
            batch, n, channels = x.shape
            width = height = int(n ** 0.5)
            x = jnp.reshape(x, (batch, height, width, channels))
            x = DepthwiseConv2D((3, 3))(x)
            x = jnp.reshape(x, (batch, -1, channels))

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x
