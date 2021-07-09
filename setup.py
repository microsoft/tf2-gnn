#!/usr/bin/env python
"""Make the module pip installable."""

import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="tf2_gnn",
    version="2.13.0",
    license="MIT",
    author="Marc Brockschmidt",
    author_email="mabrocks@microsoft.com",
    description="TensorFlow 2.0 implementation of Graph Neural Networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/tf2-gnn/",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "tensorflow>=2.0.0",
        "dpu-utils>=0.2.7",
        "h5py",
    ],
    packages=setuptools.find_packages(where="."),
    package_dir={"": "."},
    package_data={"": ["default_hypers/*.json"]},
    entry_points={
        "console_scripts": [
            "tf2_gnn_train = tf2_gnn.cli.train:run",
            "tf2_gnn_test = tf2_gnn.cli.test:run",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
