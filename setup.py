#!/usr/bin/env python

import os
from setuptools import setup


def requirements_list():
    here = os.path.abspath(os.path.dirname(__file__))
    reqs_path = os.path.join(here, "requirements.txt")
    
    reqs_list = []

    with open(reqs_path, mode="r") as reqs_file:
        for line in reqs_file:
            reqs_list.append(line)

    return reqs_list


setup(
    name = "bioverse",
    version="1.1.8",
    author = "Alex Bixel",
    author_email = "bioverse-dev@list.arizona.edu",
    description = ("A simulation framework to assess the statistical power of future biosignature surveys"),
    url = "https://github.com/danielapai/bioverse",
    packages=['bioverse'],
    install_requires=requirements_list(),
    include_package_data=True
)
