import jax.numpy as jnp
import flax.linen as nn


class ExtractPatches(nn.Module):
    @nn.compact
    def __call__(self, inputs, patch_size):
        batch, height, width, channels = inputs.shape
        height, width = height // patch_size, width // patch_size
        x = jnp.reshape(
            inputs, (batch, height, patch_size, width, patch_size, channels)
        )
        x = jnp.reshape(x, (batch, height * width, patch_size ** 2 * channels))
        return x


class MergePatches(nn.Module):
    @nn.compact
    def __call__(self, inputs, patch_size):
        batch, length, _ = inputs.shape
        height = width = int(length ** 0.5)
        x = jnp.reshape(inputs, (batch, height, patch_size, width, patch_size, -1))
        x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
        return x
