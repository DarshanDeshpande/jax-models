import jax.numpy as jnp
import jax.random as random

import flax.linen as nn


class Mask(nn.Module):
    def __call__(self, x, mask_ratio):
        batch, length, dim = x.shape
        keep_len = int(length * (1 - mask_ratio))

        rng = self.make_rng("noise")
        noise = random.uniform(rng, (batch, length))
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :keep_len]
        expanded = jnp.expand_dims(ids_keep, -1)

        repeat = jnp.tile(expanded, (1, 1, dim))
        masked = jnp.take_along_axis(x, repeat, axis=1)

        mask = jnp.ones([batch, length])
        mask.at[:, :keep_len].set(0)
        mask = jnp.take_along_axis(mask, ids_restore, axis=1)

        return masked, mask, ids_restore
