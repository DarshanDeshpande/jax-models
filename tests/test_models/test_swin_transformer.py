import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.models.swin_transformer import SwinTransformer


class TestSwinTransformer(unittest.TestCase):
    def test_headless_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        model = SwinTransformer(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        out = model.apply(
            {"params": params},
            x,
            False,
            rngs={"dropout": rng2, "drop_path": rng3},
            mutable=["attention_mask", "rel_pos_index"],
        )
        self.assertEqual(out[0].shape, (1, 768))

    def test_head_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        model = SwinTransformer(attach_head=True)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        out = model.apply(
            {"params": params},
            x,
            False,
            rngs={"dropout": rng2, "drop_path": rng3},
            mutable=["attention_mask", "rel_pos_index"],
        )
        self.assertEqual(out[0].shape, (1, 1000))
