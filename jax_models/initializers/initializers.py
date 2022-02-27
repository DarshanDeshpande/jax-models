import jax.random as random
import jax.numpy as jnp


def trunc_norm_init(key, shape, dtype=jnp.float32, std=0.02, mean=0.0):
    return std * random.truncated_normal(key, -2, 2, shape) + mean
