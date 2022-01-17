import unittest
from jax import random
import jax.numpy as jnp

from jax_models.models.patchconvnet import (
    ConvolutionalStem,
    TrunkBlock,
    AttentionPoolingBlock,
    PatchConvNet,
)


class LayerTests(unittest.TestCase):
    def test_conv_stem(self):
        sample_image = jnp.zeros([1, 224, 224, 3])
        convstem = ConvolutionalStem(emb_dim=768)
        params = convstem.init({"params": random.PRNGKey(0)}, sample_image)["params"]
        output_shape = convstem.apply({"params": params}, sample_image).shape
        self.assertEqual(output_shape, (1, 196, 768))

    def test_trunk(self):
        arr = jnp.zeros([1, 196, 768])
        trunk = TrunkBlock(768)
        params = trunk.init({"params": random.PRNGKey(0)}, arr)["params"]
        output_shape = trunk.apply({"params": params}, arr).shape
        self.assertEqual(output_shape, (1, 196, 768))

    def test_attention_pooling_block(self):
        params, key = random.split(random.PRNGKey(0), 2)
        attblock = AttentionPoolingBlock(768, 4)
        arr = jnp.zeros([1, 196, 768])
        cls_token = jnp.zeros([1, 1, 768])
        params = attblock.init(
            {"params": params, "dropout": key}, [arr, cls_token], True
        )["params"]
        output_shape = attblock.apply(
            {"params": params}, [arr, cls_token], True, rngs={"dropout": key}
        ).shape

        self.assertEqual(output_shape, (1, 1, 768))


class PatchConvNetTest(unittest.TestCase):
    def test_output_shape(self):
        rng1, rng2, rng3 = random.split(random.PRNGKey(0), 3)
        num_classes = 100
        model = PatchConvNet(
            attach_head=True, num_classes=num_classes, depth=1, dim=384, dropout=0.3
        )
        x = jnp.zeros([1, 224, 224, 3])
        params = model.init(
            {"params": rng1, "dropout": rng2, "drop_path": rng3}, x, False
        )["params"]
        logits_shape = model.apply(
            {"params": params}, x, False, rngs={"dropout": rng2, "drop_path": rng3}
        ).shape
        self.assertEqual(logits_shape, (1, num_classes))
