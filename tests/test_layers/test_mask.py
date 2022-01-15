import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.layers import Mask


class TestMask(unittest.TestCase):
    def test_random_mask(self):
        rng1, rng2 = random.split(random.PRNGKey(0), 2)
        x = jnp.zeros([1, 16, 256])
        mask_ratio = 0.5

        mask = Mask()
        params = mask.init({"params": rng1, "noise": rng2}, x, mask_ratio)
        out, mask, restore = mask.apply(
            {"params": params}, x, mask_ratio, rngs={"noise": rng2}
        )

        # Output -> (batch, input.shape[1]//mask_ratio, input.shape[-1])
        self.assertEqual(out.shape, (1, 8, 256))
        # For zeros, output should be completely masked
        self.assertTrue((out == 0).all())
        # For zeros, mask should be all ones (fully masked)
        self.assertTrue((mask == 1).all())
