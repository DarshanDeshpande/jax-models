import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.models.conv_mixer import ConvMixer


class TestConvMixer(unittest.TestCase):
    def test_output_shape(self):
        key = random.PRNGKey(0)
        model = ConvMixer(
            features=512,
            patch_size=7,
            num_mixer_layers=12,
            filter_size=8,
            attach_head=False,
        )
        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": key}, x, False)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            False,
            mutable=["batch_stats"],
        )[0]

        self.assertEqual(out.shape, (1, 32, 32, 512))
