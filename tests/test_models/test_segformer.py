import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.models.segformer import SegFormer


class TestSegFormer(unittest.TestCase):
    def test_output_shape(self):
        key, drop, drop_path = random.split(random.PRNGKey(0), 3)
        model = SegFormer(
            4,
            (32, 64, 160, 256),
            (1, 2, 5, 8),
            0.0,
            0.0,
            0.1,
            (2, 2, 2, 2),
            (8, 4, 2, 1),
            attach_head=True,
            num_classes=19,
            decoder_emb=256,
            axis_name=None,
            axis_index_groups=None,
        )

        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init(
            {"params": key, "dropout": drop, "drop_path": drop_path}, x, False
        )
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            False,
            rngs={"dropout": drop, "drop_path": drop_path},
            mutable=["batch_stats"],
        )

        self.assertEqual(out.shape, (1, 56, 56, 19))
