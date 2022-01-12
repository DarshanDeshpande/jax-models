import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.models.convnext import ConvNeXt


class TestConvNeXt(unittest.TestCase):
    def test_output_shape(self):
        key, drop = random.split(random.PRNGKey(0), 2)
        model = ConvNeXt(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init({"params": key, "drop_path": drop}, x, False)["params"]
        out = model.apply({"params": params}, x, False, rngs={"drop_path": drop})
        self.assertEqual(out.shape, (1, 7, 7, 768))
