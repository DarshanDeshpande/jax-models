import unittest
import jax.random as random
import jax.numpy as jnp

from jax_models.models.masked_autoencoder import MaskedAutoencoderViT


class TestMAE(unittest.TestCase):
    def test_mae_shape(self):
        rng1, rng2, rng3, rng4 = random.split(random.PRNGKey(0), 4)
        mask = MaskedAutoencoderViT(
            emb_dim=256,
            dec_emb_dim=128,
            depth=2,
            dec_depth=1,
            mask_ratio=0.75,
        )
        x = jnp.zeros([1, 64, 64, 3])
        params = mask.init(
            {"params": rng1, "noise": rng2, "drop_path": rng3, "dropout": rng4},
            x,
            False,
        )["params"]
        out = mask.apply(
            {"params": params},
            x,
            False,
            rngs={"noise": rng2, "drop_path": rng3, "dropout": rng4},
            mutable=["pos_emb"],
        )
        self.assertEqual(out[0][0].shape, (1, 16, 768))
        self.assertEqual(out[0][1].shape, (1, 16))
