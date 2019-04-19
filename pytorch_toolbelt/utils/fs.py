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

    files = list(glob.iglob(os.path.join(where, '**', filename), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError('Given file could not be found with recursive search:' + filename)

    if len(files) > 1:
        raise FileNotFoundError('More than one file matches given filename. Please specify it explicitly:' + filename)

    return files[0]


def read_rgb_image(fname: str) -> np.ndarray:
    """Read RGB image from filesystem. This function uses PIL to load image since PIL respects EXIF image orientation flag.
    :param fname: Image file path
    :return: A numpy array with a loaded image in RGB format
    """
    from PIL import Image
    im = Image.open(fname)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    image = np.asarray(im)
    return image


def read_image_as_is(fname: str) -> np.ndarray:
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f'Cannot read image \"{fname}\"')
    return image
