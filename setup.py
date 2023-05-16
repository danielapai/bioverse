#!/usr/bin/env python

import os
from setuptools import setup

setup(
    name = "bioverse",
    version = "1.1.1",
    author = "Alex Bixel",
    author_email = "d.alex.bixel@gmail.com",
    description = ("A simulation framework to assess the statistical power of future biosignature surveys"),
    url = "https://github.com/danielapai/bioverse",
    packages=['bioverse'],
    include_package_data=True
)
