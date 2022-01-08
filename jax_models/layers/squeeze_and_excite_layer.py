import jax.numpy as jnp

import flax.linen as nn


class SqueezeAndExcitation(nn.Module):
    reduction: int = 16

    @nn.compact
    def __call__(self, inputs):
        batch, _, _, channels = inputs.shape
        global_avg_pool = jnp.mean(inputs, axis=[1, 2], keepdims=False)
        dense1 = nn.Dense(channels // self.reduction, use_bias=False)(global_avg_pool)
        dense1 = nn.relu(dense1)
        dense2 = nn.Dense(channels, use_bias=False)(dense1)
        dense2 = nn.sigmoid(dense2)
        expand_dims = jnp.reshape(dense2, (batch, 1, 1, channels))
        shape_broadcasted = jnp.broadcast_to(expand_dims, inputs.shape)
        return inputs * shape_broadcasted
