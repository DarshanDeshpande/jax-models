import unittest

import jax.numpy as jnp
import jax.random as random

from jax_models.models.mpvit import (
    ConvolutionalStem,
    MultiScalePatchEmbedding,
    ConvolutionalLocalFeature,
    FactorizedAttention,
    MultiPathTransformerBlock,
)
from jax_models.models.mpvit import *


class Layer_Tests(unittest.TestCase):
    def test_conv_stem(self):
        key = random.PRNGKey(0)
        stem = ConvolutionalStem(emb_dim=64)
        x = jnp.zeros([1, 224, 224, 3])
        variables = stem.init({"params": key}, x, False)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = stem.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            False,
            mutable=["batch_stats"],
        )
        self.assertEqual(out.shape, (1, 3136, 64))

    def test_ms_patch_emb(self):
        key = random.PRNGKey(0)
        mspe = MultiScalePatchEmbedding(
            features=64, kernel_size=(3, 3), strides=1, padding="SAME"
        )
        x = jnp.zeros([1, 256, 64])
        variables = mspe.init({"params": key}, x, False)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = mspe.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            False,
            mutable=["batch_stats"],
        )
        self.assertTrue(out[0].shape == out[1].shape == out[2].shape)
        self.assertEqual(out[0].shape, x.shape)

    def test_conv_local_feature(self):
        key = random.PRNGKey(0)
        clf = ConvolutionalLocalFeature(features=64)
        x = jnp.zeros([1, 256, 64])
        variables = clf.init({"params": key}, x, False)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = clf.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            False,
            mutable=["batch_stats"],
        )
        self.assertEqual(out.shape, x.shape)

    def test_factorized_attention(self):
        key, drop = random.split(random.PRNGKey(0), 2)
        att = FactorizedAttention(256, 8)
        x = jnp.zeros([1, 197, 256])
        variables = att.init({"params": key, "dropout": drop}, x, False)
        params = variables["params"]
        out = att.apply({"params": params}, x, False, rngs={"dropout": drop})
        self.assertEqual(out.shape, x.shape)

    def test_mp_transformer(self):
        key, drop = random.split(random.PRNGKey(0), 2)
        mptb = MultiPathTransformerBlock(
            features=256,
            dim=256,
            num_heads=8,
            att_drop=0.1,
            proj_drop=0.1,
            mlp_ratio=2,
            num_encoder_layers=1,
        )
        x = jnp.zeros([1, 196, 256])
        variables = mptb.init({"params": key, "dropout": drop}, (x, x, x), False)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = mptb.apply(
            {"params": params, "batch_stats": batch_stats},
            (x, x, x),
            False,
            rngs={"dropout": drop},
            mutable=["batch_stats"],
        )
        self.assertEqual(out.shape, x.shape)


class MPViT_Tests(unittest.TestCase):
    def test_tiny(self):
        drop, key = random.split(random.PRNGKey(0), 2)
        model = mpvit_tiny(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": key, "dropout": drop}, x, True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            True,
            mutable=["batch_stats"],
            rngs={"dropout": drop},
        )
        self.assertEqual(out.shape, (1, 16, 216))

    def test_xsmall(self):
        drop, key = random.split(random.PRNGKey(0), 2)
        model = mpvit_xsmall(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": key, "dropout": drop}, x, True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            True,
            mutable=["batch_stats"],
            rngs={"dropout": drop},
        )
        self.assertEqual(out.shape, (1, 16, 256))

    def test_small(self):
        drop, key = random.split(random.PRNGKey(0), 2)
        model = mpvit_small(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": key, "dropout": drop}, x, True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            True,
            mutable=["batch_stats"],
            rngs={"dropout": drop},
        )
        self.assertEqual(out.shape, (1, 16, 288))

    def test_base(self):
        drop, key = random.split(random.PRNGKey(0), 2)
        model = mpvit_base(attach_head=False)
        x = jnp.zeros([1, 224, 224, 3])
        variables = model.init({"params": key, "dropout": drop}, x, True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        out, batch_stats = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            True,
            mutable=["batch_stats"],
            rngs={"dropout": drop},
        )
        self.assertEqual(out.shape, (1, 16, 480))
