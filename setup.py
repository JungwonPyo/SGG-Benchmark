#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

setup(
    name="sgg_benchmark",
    version="2.0.0",
    author="maelicneau",
    author_email="teoneau@gmail.com",
    description="Real-Time Scene Graph Generation Benchmark",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements,
)
