"""Filesystem utilities

"""

import glob
import os
from pathlib import Path
from typing import Union, List, Iterable

import cv2
import numpy as np

__all__ = [
    "auto_file",
    "change_extension",
    "find_images_in_dir",
    "find_images_in_dir_recursive",
    "find_in_dir",
    "find_in_dir_glob",
    "find_in_dir_with_ext",
    "find_subdirectories_in_dir",
    "has_ext",
    "has_image_ext",
    "id_from_fname",
    "read_image_as_is",
    "read_rgb_image",
    "zipdir",
]

COMMON_IMAGE_EXTENSIONS = (".bmp", ".png", ".jpeg", ".jpg", ".tiff", ".tif", ".jp2")


def has_ext(fname: str, extensions: Union[str, Iterable[str]]) -> bool:
    if not isinstance(extensions, (str, list, tuple)):
        raise ValueError("Argument extensions must be either string or list of strings")
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = set(map(str.lower, extensions))

    name, ext = os.path.splitext(fname)
    return ext.lower() in extensions


def has_image_ext(fname: str) -> bool:
    return has_ext(fname, COMMON_IMAGE_EXTENSIONS)


def find_in_dir(dirname: str) -> List[str]:
    if not os.path.isdir(dirname):
        raise FileNotFoundError(f"Directory {dirname} not found")

    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]


def find_subdirectories_in_dir(dirname: str) -> List[str]:
    """
    Retrieve list of subdirectories (non-recursive) in the given directory.
    Args:
        dirname: Target directory name

    Returns:
        Sorted list of absolute paths to directories
    """
    all_entries = find_in_dir(dirname)
    return [entry for entry in all_entries if os.path.isdir(entry)]


def find_in_dir_with_ext(dirname: str, extensions: Union[str, List[str]]) -> List[str]:
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname)) if has_ext(fname, extensions)]


def find_images_in_dir(dirname: str) -> List[str]:
    return [fname for fname in find_in_dir(dirname) if has_image_ext(fname) and os.path.isfile(fname)]


def find_images_in_dir_recursive(dirname: str) -> List[str]:
    return [
        fname
        for fname in glob.glob(os.path.join(dirname, "**"), recursive=True)
        if has_image_ext(fname) and os.path.isfile(fname)
    ]


def find_in_dir_glob(dirname: str, recursive=False):
    files = list(glob.iglob(dirname, recursive=recursive))
    return list(sorted(files))


def id_from_fname(fname: str) -> str:
    return os.path.splitext(os.path.basename(fname))[0]


def change_extension(fname: Union[str, Path], new_ext: str) -> Union[str, Path]:
    if isinstance(fname, str):
        return os.path.splitext(fname)[0] + new_ext
    elif isinstance(fname, Path):
        if new_ext[0] != ".":
            new_ext = "." + new_ext
        return fname.with_suffix(new_ext)
    else:
        raise RuntimeError(
            f"Received input argument `fname` for unsupported type {type(fname)}. Argument must be string or Path."
        )


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


def read_rgb_image(fname: Union[str, Path]) -> np.ndarray:
    """
    Read RGB image from filesystem in RGB color order.
    Note: By default, OpenCV loads images in BGR memory order format.
    :param fname: Image file path
    :return: A numpy array with a loaded image in RGB format
    """
    if type(fname) != str:
        fname = str(fname)

    image = cv2.imread(fname, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f'Cannot read image "{fname}"')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
    return image


def read_image_as_is(fname: Union[str, Path]) -> np.ndarray:
    if type(fname) != str:
        fname = str(fname)
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image "{fname}"')
    return image


def zipdir(path, output_filename):
    """Create a zip file from a directory."""
    import zipfile

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(
                    os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, ".."))
                )
