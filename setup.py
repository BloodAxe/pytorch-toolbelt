import ast
import io
import re
import os
import sys

from setuptools import find_packages, setup
from distutils.version import LooseVersion


def is_docker() -> bool:
    """
    Check whether setup is running in Docker environment.
    """
    # Note: You have to set the environment variable AM_I_IN_A_DOCKER_CONTAINER manually
    # in your Dockerfile .
    if os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False):
        return True

    path = "/proc/self/cgroup"
    if not os.path.isfile(path):
        return False

    with open(path) as f:
        for line in f:
            if re.match("\\d+:[\\w=]+:/docker(-[ce]e)?/\\w+", line):
                return True

    return False


def is_kaggle() -> bool:
    """
    Check whether setup is running in Kaggle environment.
    This is not 100% bulletproff solution to detect whether we are in Kaggle Notebooks,
    but it should be enough unless Kaggle change their environment variables.
    """
    return (
        ("KAGGLE_CONTAINER_NAME" in os.environ)
        or ("KAGGLE_URL_BASE" in os.environ)
        or ("KAGGLE_DOCKER_IMAGE" in os.environ)
    )


def is_colab() -> bool:
    """
    Check whether setup is running in Google Colab.
    This is not 100% bulletproff solution to detect whether we are in Colab,
    but it should be enough unless Google change their environment variables.
    """
    return (
        ("COLAB_GPU" in os.environ)
        or ("GCE_METADATA_TIMEOUT" in os.environ)
        or ("GCS_READ_CACHE_BLOCK_SIZE_MB" in os.environ)
    )


def get_opencv_requirement():
    """
    Return the OpenCV requirement string.
    Since opencv library is distributed in several independent packages,
    we first check whether any form of opencv is already installed. If not,
    we choose between opencv-python vs opencv-python-headless version based
    on the environment.
    For headless environment (Docker, Colab & Kaggle notebooks), we install
    opencv-python-headless; otherwise - default to opencv-python.
    """
    try:
        import cv2

        return []
    except ImportError:
        default_requirement = "opencv-python>=4.1"
        headless_requirement = "opencv-python-headless>=4.1"

        if is_docker() or is_kaggle() or is_colab():
            return [headless_requirement]
        else:
            return [default_requirement]


# Package meta-data.
NAME = "pytorch_toolbelt"
DESCRIPTION = "PyTorch extensions for fast R&D prototyping and Kaggle farming"
URL = "https://github.com/BloodAxe/pytorch-toolbelt"
EMAIL = "ekhvedchenya@gmail.com"
AUTHOR = "Eugene Khvedchenya"
REQUIRES_PYTHON = ">=3.8.0"

DEPENDENCIES = [
    # We rely on particular activation functions that were added in 1.8.1
    "torch>=1.11.0" if LooseVersion(sys.version) >= LooseVersion("3.7") else "torch>=1.10.0",
    # We use some pretrained models from torchvision
    "torchvision",
    # Library uses scipy for linear_sum_assignment for match_bboxes.
    # 1.4.0 is the first release where `maximize` argument gets introduced to this function
    "scipy>=1.4.0",
] + get_opencv_requirement()

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
    requirements = ["pytest", "black==23.3.0", "timm==0.6.7", "matplotlib"]
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
    extras_require={"tests": get_test_requirements()},
    include_package_data=True,
    keywords=[
        "PyTorch",
        "Kaggle",
        "Deep Learning",
        "Machine Learning",
        "ResNet",
        "VGG",
        "ResNext",
        "Unet",
        "Focal",
        "FPN",
        "EfficientNet",
        "Test-Time Augmentation",
        "Model Ensembling",
        "Model Surgery",
    ],
    scripts=[],
    license="License :: OSI Approved :: MIT License",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks"
        # "Private :: Do Not Upload"
    ],
)
