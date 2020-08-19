"""Filesystem utilities

"""

import glob
import os
from typing import Union, List

import cv2
import numpy as np

__all__ = [
    "auto_file",
    "change_extension",
    "find_images_in_dir",
    "find_in_dir",
    "find_in_dir_glob",
    "has_ext",
    "has_image_ext",
    "id_from_fname",
    "read_image_as_is",
    "read_rgb_image",
]

COMMON_IMAGE_EXTENSIONS = [".bmp", ".png", ".jpeg", ".jpg", ".tiff", ".tif"]


def has_ext(fname: str, extensions: Union[str, List[str]]) -> bool:
    if not isinstance(extensions, (str, list)):
        raise ValueError("Argument extensions must be either string or list of strings")
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = set(map(str.lower, extensions))

    name, ext = os.path.splitext(fname)
    return ext.lower() in extensions


def has_image_ext(fname: str) -> bool:
    return has_ext(fname, COMMON_IMAGE_EXTENSIONS)


def find_in_dir(dirname: str) -> List[str]:
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]


def find_in_dir_with_ext(dirname: str, extensions: Union[str, List[str]]) -> List[str]:
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname)) if has_ext(fname, extensions)]


def find_images_in_dir(dirname: str) -> List[str]:
    return [fname for fname in find_in_dir(dirname) if has_image_ext(fname)]


def find_in_dir_glob(dirname: str, recursive=False):
    files = list(glob.iglob(dirname, recursive=recursive))
    return list(sorted(files))


def id_from_fname(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def change_extension(fname: str, new_ext: str) -> str:
    return os.path.splitext(fname)[0] + new_ext


def auto_file(filename: str, where: str = ".") -> str:
    """Get a full path to file using it's name.
    This function recisively search for matching filename in @where and returns single match.
    :param where:
    :param filename:
    :return:
    """
    if os.path.isabs(filename):
        return filename

    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return prob

    files = list(glob.iglob(os.path.join(where, "**", filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError("Given file could not be found with recursive search:" + filename)

    if len(files) > 1:
        raise FileNotFoundError(
            "More than one file matches given filename. Please specify it explicitly:\n" + "\n".join(files)
        )

    return files[0]


def read_rgb_image(fname: str) -> np.ndarray:
    """
    Read RGB image from filesystem in RGB color order.
    Note: By default, OpenCV loads images in BGR memory order format.
    :param fname: Image file path
    :return: A numpy array with a loaded image in RGB format
    """
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image "{fname}"')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def read_image_as_is(fname: str) -> np.ndarray:
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image "{fname}"')
    return image
