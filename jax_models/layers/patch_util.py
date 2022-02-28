from typing import Callable
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
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(x, (batch, height * width, patch_size ** 2 * channels))
        return x


class MergePatches(nn.Module):
    @nn.compact
    def __call__(self, inputs, patch_size):
        batch, length, _ = inputs.shape
        height = width = int(length ** 0.5)
        x = jnp.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
        x = jnp.swapaxes(x, 2, 3)
        x = jnp.reshape(x, (batch, height * patch_size, width * patch_size, -1))
        return x


class PatchEmbed(nn.Module):
    patch_size: int = 16
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        batch, height, width, channels = inputs.shape
        x = nn.Conv(
            self.emb_dim,
            (self.patch_size, self.patch_size),
            self.patch_size,
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (batch, -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class OverlapPatchEmbed(nn.Module):
    emb_dim: int = 768
    patch_size: int = 16
    stride: int = 4
    kernel_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, inputs):
        conv = nn.Conv(
            self.emb_dim,
            (self.patch_size, self.patch_size),
            self.stride,
            padding=(
                (self.patch_size // 2, self.patch_size // 2),
                (self.patch_size // 2, self.patch_size // 2),
            ),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        flat = jnp.reshape(conv, (conv.shape[0], -1, conv.shape[-1]))
        norm = nn.LayerNorm(name="norm")(flat)
        return norm
