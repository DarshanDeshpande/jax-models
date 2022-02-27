import jax.numpy as jnp
import flax.linen as nn

from ..layers import DepthwiseConv2D
from typing import Optional, Callable


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.0
    use_dwconv: bool = False
    linear: bool = False
    dense_kernel_init: Callable = nn.initializers.xavier_uniform()
    conv_kernel_init: Optional[Callable] = nn.initializers.xavier_normal()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Dense(self.dim, kernel_init=self.dense_kernel_init, name="fc1")(inputs)

        if self.linear:
            x = nn.relu(x)

        if self.use_dwconv:
            batch, n, channels = x.shape
            width = height = int(n ** 0.5)
            x = jnp.reshape(x, (batch, height, width, channels))
            x = DepthwiseConv2D(
                (3, 3), name="dwconv", weights_init=self.conv_kernel_init
            )(x)
            x = jnp.reshape(x, (batch, -1, channels))

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Dense(self.out_dim, kernel_init=self.dense_kernel_init, name="fc2")(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x
