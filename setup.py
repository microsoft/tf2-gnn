#!/usr/bin/env python
"""Make the module pip installable."""

import setuptools

setuptools.setup(
    name="tf2_gnn",
    version="0.9.21",
    description="TensorFlow 2.0 implementation of Graph Neural Networks.",
    python_requires=">=3.6",
    install_requires=["numpy", "docopt", "dpu-utils>=0.2.7", "h5py"],
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},
    package_data={"": ["default_hypers/*.json"]},
    entry_points={
        "console_scripts": [
            "tf2_gnn_train = tf2_gnn.cli.train:run",
            "tf2_gnn_test = tf2_gnn.cli.test:run",
        ]
    },
)
