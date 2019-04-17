#!/usr/bin/env python
# -*- coding: utf-8 -*-

# For a fully annotated version of this file and what it does, see
# https://github.com/pypa/sampleproject/blob/master/setup.py

# To upload this file to PyPI you must build it then upload it:
# python setup.py sdist bdist_wheel  # build in 'dist' folder
# python-m twine upload dist/*  # 'twine' must be installed: 'pip install twine'


import ast
import io
import re
import os
import sys

from setuptools import find_packages, setup

# Package meta-data.
NAME = "pytorch_toolbelt"
DESCRIPTION = "PyTorch extensions for fast R&D prototyping and Kaggle farming"
URL = "https://github.com/BloodAxe/pytorch-toolbelt"
EMAIL = "ekhvedchenya@gmail.com"
AUTHOR = "Eugene Khvedchenya"
REQUIRES_PYTHON = ">=3.6.0"

DEPENDENCIES = ["torch>=0.4.1", "torchvision>0.2", "opencv-python>=3.0"]
EXCLUDE_FROM_PACKAGES = ["contrib", "docs", "tests", "examples"]
CURDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    main_file = os.path.join(CURDIR, "pytorch_toolbelt", "__init__.py")
    _version_re = re.compile(r"__version__\s+=\s+(?P<version>.*)")
    with open(main_file, "r", encoding="utf8") as f:
        match = _version_re.search(f.read())
        version = match.group("version") if match is not None else '"unknown"'
    return str(ast.literal_eval(version))


def load_readme():
    readme_path = os.path.join(CURDIR, "README.md")
    with io.open(readme_path, encoding="utf-8") as f:
        return "\n" + f.read()


def get_test_requirements():
    requirements = ['pytest']
    if sys.version_info < (3, 3):
        requirements.append('mock')
    return requirements


setup(
    name=NAME,
    version=get_version(),
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    install_requires=DEPENDENCIES,
    python_requires=REQUIRES_PYTHON,
    extras_require={'tests': get_test_requirements()},
    include_package_data=True,
    keywords=["PyTorch", "Kaggle", "Deep Learning", "Machine Learning", "ResNet", "VGG", "ResNext", "Unet", "Focal"],
    scripts=[],
    license="License :: OSI Approved :: MIT License",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        "Topic :: Software Development :: Libraries :: Application Frameworks"
        # "Private :: Do Not Upload"
    ],
)
