"""Filesystem utilities

"""

import glob
import os
import cv2
import numpy as np


def has_image_ext(fname: str):
    name, ext = os.path.splitext(fname)
    return ext.lower() in {'.bmp', '.png', '.jpeg', '.jpg', '.tiff'}


def find_in_dir(dirname: str):
    return [os.path.join(dirname, fname) for fname in sorted(os.listdir(dirname))]


def find_images_in_dir(dirname: str):
    return [fname for fname in find_in_dir(dirname) if has_image_ext(fname)]


def find_in_dir_glob(dirname: str, recursive=False):
    files = list(glob.iglob(dirname, recursive=recursive))
    return list(sorted(files))


def id_from_fname(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]


def change_extension(fname: str, new_ext: str):
    return os.path.splitext(fname)[0] + new_ext


def auto_file(filename: str, where: str = '.') -> str:
    """
    Helper function to find a unique filename in subdirectory without specifying fill path to it
    :param where:
    :param filename:
    :return:
    """

    if os.path.isabs(filename):
        return filename

    prob = os.path.join(where, filename)
    if os.path.exists(prob) and os.path.isfile(prob):
        return prob

    files = list(glob.iglob(os.path.join(where, '**', filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError('Given file could not be found with recursive search:' + filename)

    if len(files) > 1:
        raise FileNotFoundError('More than one file matches given filename. Please specify it explicitly:' + filename)

    return files[0]


def read_rgb_image(fname: str) -> np.ndarray:
    import PIL
    image = np.asarray(PIL.Image.open(fname))
    return image

    # image = cv2.imread(fname, cv2.IMREAD_COLOR)
    # if image is None:
    #     raise IOError(f'Cannot read image \"{fname}\"')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return image


def read_image_as_is(fname: str) -> np.ndarray:
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image \"{fname}\"')
    return image
