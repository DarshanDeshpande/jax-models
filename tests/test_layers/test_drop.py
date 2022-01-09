import unittest
from jax import random
import numpy as np

from jax_models.layers import DropPath


class TestDropPath(unittest.TestCase):
    def test_drop_path(self):
        droppath = DropPath(0.3)
        params_key, drop_path_key, key = random.split(random.PRNGKey(0), 3)

        arr = random.uniform(key, [3, 2, 2, 2])
        params = droppath.init(
            {"params": params_key, "drop_path": drop_path_key}, arr, False
        )
        output = droppath.apply(
            {"params": params}, arr, False, rngs={"drop_path": drop_path_key}
        )

        test_arr = [
            [
                [[0.07269315, 0.9130809], [1.2821915, 0.98061377]],
                [[0.45805728, 0.3266367], [1.4167114, 0.10965841]],
            ],
            [
                [[0.6914027, 1.1517124], [1.3579687, 0.5189223]],
                [[0.7914342, 0.3071068], [0.3616304, 0.057035]],
            ],
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        ]

        assert np.testing.assert_allclose(output, test_arr) is None
