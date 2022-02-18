import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.models.cait import *


class TestmodelTransformer(unittest.TestCase):
    def test_headless_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        model = CaiT(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        out = model.apply(
            {"params": params},
            x,
            False,
            rngs={"dropout": rng2, "drop_path": rng3},
        )
        self.assertEqual(out.shape, (1, 768))

    def test_head_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        model = CaiT(attach_head=True)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        out = model.apply(
            {"params": params},
            x,
            False,
            rngs={"dropout": rng2, "drop_path": rng3},
        )
        self.assertEqual(out.shape, (1, 1000))

    def test_pretrained_weights(self):
        x = jnp.zeros([1, 224, 224, 3])
        y = jnp.zeros([1, 384, 384, 3])
        z = jnp.zeros([1, 448, 448, 3])

        model, params = XXS24_224(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            x,
            True,
        )

        model, params = XXS24_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = XXS36_224(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            x,
            True,
        )

        model, params = XXS36_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = XS24_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = S24_224(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            x,
            True,
        )

        model, params = S24_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = S36_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = M36_384(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            y,
            True,
        )

        model, params = M48_448(pretrained=True, download_dir="weights/cait_weights/")
        model.apply(
            {"params": params},
            z,
            True,
        )
