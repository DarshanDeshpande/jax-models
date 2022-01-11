import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.models.mlp_mixer import MLPMixer


class TestMLPMixer(unittest.TestCase):
    def test_output_shape(self):
        key, drop = random.split(random.PRNGKey(0), 2)
        model = MLPMixer(
            patch_size=32,
            num_mixers_layers=2,
            hidden_size=768,
            channels_dim=256,
            tokens_dim=256,
            dropout=0.2,
            attach_head=False,
            num_classes=1000,
            deterministic=None,
        )
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init({"params": key, "dropout": drop}, x, False)["params"]
        out = model.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(out.shape, (1, 49, 768))
