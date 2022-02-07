import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.models.convnext import *


class TestConvNeXt(unittest.TestCase):
    def test_output_shape(self):
        key, drop = random.split(random.PRNGKey(0), 2)
        model = ConvNeXt(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init({"params": key, "drop_path": drop}, x, False)["params"]
        out = model.apply({"params": params}, x, False, rngs={"drop_path": drop})
        self.assertEqual(out.shape, (1, 7, 7, 768))

    def test_pretrained_weights(self):
        x = jnp.zeros([1, 224, 224, 3])
        y = jnp.zeros([1, 384, 384, 3])

        convnext, params = ConvNeXt_Tiny(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Small(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Base_224_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Base_224_22K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Base_224_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Base_384_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            y,
            True,
        )

        convnext, params = ConvNeXt_Base_384_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            y,
            True,
        )

        convnext, params = ConvNeXt_Large_224_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Large_224_22K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Large_224_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_Large_384_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            y,
            True,
        )

        convnext, params = ConvNeXt_Large_384_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            y,
            True,
        )

        convnext, params = ConvNeXt_XLarge_224_22K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_XLarge_224_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            x,
            True,
        )

        convnext, params = ConvNeXt_XLarge_384_22K_1K(
            pretrained=True, download_dir="weights/convnext"
        )
        convnext.apply(
            {"params": params},
            y,
            True,
        )
