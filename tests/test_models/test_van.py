import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.models.van import *


class TestVAN(unittest.TestCase):
    def test_output_shape(self):
        model = VAN()

        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": random.PRNGKey(0)}, x, True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out = model.apply({"params": params, "batch_stats": batch_stats}, x, True)

        self.assertEqual(out.shape, (1, 1000))

    def test_pretrained(self):
        x = jnp.zeros([1, 224, 224, 3])

        model, params, batch_stats = van_tiny(
            pretrained=True, download_dir="weights/van"
        )
        model.apply({"params": params, "batch_stats": batch_stats}, x, True)

        model, params, batch_stats = van_small(
            pretrained=True, download_dir="weights/van"
        )
        model.apply({"params": params, "batch_stats": batch_stats}, x, True)

        model, params, batch_stats = van_base(
            pretrained=True, download_dir="weights/van"
        )
        model.apply({"params": params, "batch_stats": batch_stats}, x, True)

        model, params, batch_stats = van_large(
            pretrained=True, download_dir="weights/van"
        )
        model.apply({"params": params, "batch_stats": batch_stats}, x, True)
