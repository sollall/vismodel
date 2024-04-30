#! /usr/bin/env python

from setuptools import find_packages, setup

import vismodel

DESCRIPTION = "visualizer for pytorch model"
NAME = "vismodel"
AUTHOR = "sollall, nak1133"
AUTHOR_EMAIL = ""
URL = "https://github.com/sollall/vismodel"
LICENSE = "Undefined"
DOWNLOAD_URL = "https://github.com/sollall/vismodel"
VERSION = vismodel.__version__
PYTHON_REQUIRES = ">=3.10"


INSTALL_REQUIRES = ["numpy == 1.24.0","matplotlib == 3.8.4"]

EXTRAS_REQUIRE = {
    "dev": ["ruff == 0.4.2", "pytest == 8.1.2"],
}

all_groups = set(EXTRAS_REQUIRE.keys())

PACKAGES = ["vismodel"]

CLASSIFIERS = []

with open("README.md", "r") as fp:
    readme = fp.read()
long_description = readme

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    include_package_data=True,
)
