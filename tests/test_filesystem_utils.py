import os.path
from pathlib import Path

from pytorch_toolbelt.utils.fs import read_rgb_image, read_image_as_is, change_extension

lena_str_name = os.path.join(os.path.dirname(__file__), "lena.png")
lena_path = Path(os.path.dirname(__file__)) / "lena.png"
lena_path_jpeg = Path(os.path.dirname(__file__)) / "lena.jpeg"


def test_read_rgb_image_str():
    img = read_rgb_image(lena_str_name)
    assert img.shape == (220, 220, 3)


def test_read_rgb_image_pathlib():
    img = read_rgb_image(lena_path)
    assert img.shape == (220, 220, 3)


def test_read_image_as_is_str():
    img = read_image_as_is(lena_str_name)
    assert img.shape == (220, 220, 3)


def test_read_image_as_is_pathlib():
    img = read_image_as_is(lena_path)
    assert img.shape == (220, 220, 3)


def test_change_extension_str():
    assert change_extension("lena.png", ".jpeg") == "lena.jpeg"


def test_change_extension_pathlib():
    assert change_extension(lena_path, "jpeg") == lena_path_jpeg
    assert change_extension(lena_path, ".jpeg") == lena_path_jpeg
