from setuptools import setup
from codecs import open
from os import path

current_path = path.abspath(path.dirname(__file__))

with open(path.join(current_path, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jax_models",
    version="0.0.5",
    description="Unofficial JAX implementations of deep learning research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darshandeshpande/jax-models",
    author="Darshan Deshpande",
    author_email="darshan.g.deshpande@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="jax flax models computer-vision deep-learning transformers",
    packages=["jax_models"],
    package_data={"jax_models": ["*/*"]},
    include_package_data=True,
    install_requires=["flax >= 0.3.6", "tqdm"],
    python_requires=">=3.6",
)
