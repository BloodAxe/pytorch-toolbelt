from __future__ import absolute_import

import itertools
import math
import warnings
from typing import List

import cv2
import numpy as np

from .torch_utils import image_to_tensor

__all__ = [
    "plot_confusion_matrix",
    "render_figure_to_tensor",
    "hstack_autopad",
    "vstack_autopad",
    "vstack_header",
    "grid_stack",
]


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    figsize=(16, 16),
    fontsize=12,
    normalize=False,
    title="Confusion matrix",
    cmap=None,
    fname=None,
    noshow=False,
    backend="Agg",
):
    """Render the confusion matrix and return matplotlib's figure with it.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.cm.Oranges

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".3f" if normalize else "d"
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if np.isfinite(cm[i, j]):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                fontsize=fontsize,
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label\nAccuracy={:0.4f}; Misclass={:0.4f}".format(accuracy, misclass))
    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname=fname, dpi=200)

    if not noshow:
        plt.show()

    return f


def render_figure_to_tensor(figure):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure.canvas.draw()

    # string = figure.canvas.tostring_argb()

    image = np.array(figure.canvas.renderer._renderer)
    plt.close(figure)
    del figure

    image = image_to_tensor(image)
    return image


def hstack_autopad(images: List[np.ndarray], pad_value=0) -> np.ndarray:
    """
    Stack images horizontally with automatic padding

    Args:
        images: List of images to stack

    Returns:
        image
    """
    max_height = 0
    for img in images:
        max_height = max(max_height, img.shape[0])

    padded_images = []
    for img in images:
        height = img.shape[0]
        pad_top = 0
        pad_bottom = max_height - height
        pad_left = 0
        pad_right = 0
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
        (rows, cols) = img.shape[0:2]
        padded_images.append(img)

    return np.hstack(padded_images)


def vstack_autopad(images: List[np.ndarray], pad_value=0) -> np.ndarray:
    """
    Stack images vertically with automatic padding

    Args:
        images: List of images to stack

    Returns:
        image
    """
    max_width = 0
    for img in images:
        max_width = max(max_width, img.shape[1])

    padded_images = []
    for img in images:
        width = img.shape[1]
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = max_width - width
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
        padded_images.append(img)

    return np.vstack(padded_images)


def vstack_header(
    image: np.ndarray,
    title: str,
    bg_color=(35, 41, 40),
    text_color=(242, 248, 248),
    text_thickness: int = 2,
    text_scale=1.5,
) -> np.ndarray:
    (rows, cols) = image.shape[:2]

    title_image = np.zeros((30, cols, 3), dtype=np.uint8)
    title_image[:] = bg_color
    cv2.putText(
        title_image,
        title,
        (10, 24),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=text_scale,
        color=text_color,
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )

    return vstack_autopad([title_image, image])


def grid_stack(images: List[np.ndarray], rows: int = None, cols: int = None) -> np.ndarray:
    if rows is None and cols is None:
        rows = int(math.ceil(math.sqrt(len(images))))
        cols = int(math.ceil(len(images) / rows))
    elif rows is None:
        rows = math.ceil(len(images) / cols)
    elif cols is None:
        cols = math.ceil(len(images) / rows)
    else:
        if len(images) > rows * cols:
            raise ValueError("Number of rows * cols must be greater than number of images")

    image_rows = []
    for r in range(rows):
        image_rows.append(hstack_autopad(images[r * cols : (r + 1) * cols]))

    return vstack_autopad(image_rows)
