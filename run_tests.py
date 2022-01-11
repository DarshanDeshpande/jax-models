import unittest
import os

loader = unittest.TestLoader()
start_dir = os.path.dirname(os.path.abspath(__file__))
suite = loader.discover(os.path.join(start_dir, "tests/"))

runner = unittest.TextTestRunner()
runner.run(suite)
