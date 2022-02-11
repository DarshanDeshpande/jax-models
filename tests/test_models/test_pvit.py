import unittest
import jax.numpy as jnp
import jax.random as random

from jax_models.models.pvit import *


class TestSegFormer(unittest.TestCase):
    def test_output_shape(self):
        model = PyramidViT(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            use_att_bias=True,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            attach_head=True,
            num_classes=1000,
        )

        x = jnp.zeros([1, 224, 224, 3])
        params = model.init({"params": random.PRNGKey(0)}, x, True)["params"]
        out = model.apply({"params": params}, x, True)

        self.assertEqual(out.shape, (1, 1000))

    def test_pretrained(self):
        x = jnp.zeros([1, 224, 224, 3])

        model, params = PViT_B0(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B1(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B2(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B3(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B4(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B5(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)

        model, params = PViT_B2_Linear(pretrained=True, download_dir="weights/pvit")
        model.apply({"params": params}, x, True)
