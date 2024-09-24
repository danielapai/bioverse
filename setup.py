#!/usr/bin/env python

import os
from setuptools import setup


def requirements_list():
    reqs_list = []

    with open("requirements.txt", mode="r") as reqs_file:
        for line in reqs_file:
            reqs_list.append(line)

    return reqs_list


setup(
    name = "bioverse",
    version = "1.1.6",
    author = "Alex Bixel",
    author_email = "d.alex.bixel@gmail.com",
    description = ("A simulation framework to assess the statistical power of future biosignature surveys"),
    url = "https://github.com/danielapai/bioverse",
    packages=['bioverse'],
    install_requires=requirements_list(),
    include_package_data=True
)
