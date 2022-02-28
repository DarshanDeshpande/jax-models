import unittest
import jax.numpy as jnp
import jax.random as random
import numpy as np

from jax_models.layers import (
    ExtractPatches,
    MergePatches,
    PatchEmbed,
    OverlapPatchEmbed,
)


class TestPatchLayers(unittest.TestCase):
    def test_patch_creation(self):
        img = jnp.arange(36).reshape(1, 6, 6, 1)
        ep = ExtractPatches()
        ep.init({"params": random.PRNGKey(0)}, img, 3)
        patches = ep.apply({"params": {}}, img, 3)
        self.assertEqual(patches.shape, (1, 4, 9))

        expected = jnp.asarray(
            [
                [
                    [0, 1, 2, 6, 7, 8, 12, 13, 14],
                    [3, 4, 5, 9, 10, 11, 15, 16, 17],
                    [18, 19, 20, 24, 25, 26, 30, 31, 32],
                    [21, 22, 23, 27, 28, 29, 33, 34, 35],
                ]
            ],
            dtype=jnp.int32,
        )

        np.testing.assert_array_equal(patches, expected)

    def test_patch_merging(self):
        patches = jnp.asarray(
            [
                [
                    [0, 1, 2, 6, 7, 8, 12, 13, 14],
                    [3, 4, 5, 9, 10, 11, 15, 16, 17],
                    [18, 19, 20, 24, 25, 26, 30, 31, 32],
                    [21, 22, 23, 27, 28, 29, 33, 34, 35],
                ]
            ],
            dtype=jnp.int32,
        )
        mp = MergePatches()
        mp.init({"params": random.PRNGKey(0)}, patches, 3)
        recovered = mp.apply({"params": {}}, patches, 3)

        self.assertEqual(recovered.shape, (1, 6, 6, 1))
        np.testing.assert_array_equal(recovered, jnp.arange(36).reshape(1, 6, 6, 1))

    def test_patch_embed(self):
        x = jnp.zeros([1, 32, 32, 3])
        pe = PatchEmbed(patch_size=4, emb_dim=128)
        params = pe.init({"params": random.PRNGKey(0)}, x)["params"]
        out = pe.apply({"params": params}, x)
        self.assertEqual(out.shape, (1, 64, 128))

    def test_overlap_patch_embed(self):
        x = jnp.zeros([1, 32, 32, 3])
        pe = OverlapPatchEmbed(emb_dim=128, patch_size=4, stride=8)
        params = pe.init({"params": random.PRNGKey(0)}, x)["params"]
        out = pe.apply({"params": params}, x)
        self.assertEqual(out.shape, (1, 25, 128))
