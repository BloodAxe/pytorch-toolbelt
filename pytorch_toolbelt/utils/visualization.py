from __future__ import absolute_import

import itertools
import math
import warnings
from typing import List, Iterable, Tuple, Optional

import cv2
import numpy as np

from .torch_utils import image_to_tensor

__all__ = [
    "plot_confusion_matrix",
    "plot_compressed_confusion_matrix",
    "render_figure_to_tensor",
    "hstack_autopad",
    "vstack_autopad",
    "vstack_header",
    "grid_stack",
    "plot_heatmap",
]


def plot_heatmap(
    cm: np.ndarray,
    title: str,
    x_label=None,
    y_label=None,
    x_ticks: List[str] = None,
    y_ticks: List[str] = None,
    format_string=None,
    show_scores=True,
    fontsize=12,
    figsize: Tuple[int, int] = (16, 16),
    fname=None,
    noshow: bool = False,
    cmap=None,
    backend="Agg",
):
    if len(cm.shape) != 2:
        raise ValueError("Heatmap must be a 2-D array")
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.Oranges

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    if x_ticks is not None:
        plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation=45, ha="right")

    if y_ticks is not None:
        plt.yticks(np.arange(len(y_ticks)), y_ticks)

    if format_string is None:
        format_string = ".2f" if np.issubdtype(cm.dtype, np.floating) else "d"

    if show_scores:
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            text = format(cm[i, j], format_string) if np.isfinite(cm[i, j]) else "N/A"
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(
                j,
                i,
                text,
                horizontalalignment="center",
                verticalalignment="center_baseline",
                fontsize=fontsize,
                color=color,
            )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname=fname, dpi=200)

    if not noshow:
        plt.show()

    return f


def plot_compressed_confusion_matrix(
    cm: np.ndarray,
    figsize: Tuple[int, int] = (16, 16),
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap=None,
    fname=None,
    noshow: bool = False,
):
    from matplotlib import pyplot as plt

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
    else:
        accuracy = np.trace(cm) / (float(np.sum(cm)) + 1e-8)
        misclass = 1 - accuracy

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)

    if normalize:
        # We don't show Accuracy & Misclassification scores for normalized CM
        plt.xlabel("Predicted label")
    else:
        plt.xlabel("Predicted label\nAccuracy={:0.4f}; Misclass={:0.4f}".format(accuracy, misclass))

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname=fname, dpi=200)
    if not noshow:
        plt.show()
    return f


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figsize: Tuple[int, int] = (16, 16),
    fontsize: int = 12,
    normalize: bool = False,
    title: str = "Confusion matrix",
    cmap=None,
    fname=None,
    show_scores: bool = True,
    noshow: bool = False,
    backend: str = "Agg",
    format_string: Optional[str] = None,
):
    """
    Render the confusion matrix and return matplotlib's figure with it.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm: Numpy array of (N,N) shape - confusion matrix array
        class_names: List of [N] names of the classes
        figsize:
        fontsize:
        normalize: Whether to apply normalization for each row of CM
        title: Title of the confusion matrix
        cmap:
        fname: Filename of the rendered confusion matrix
        show_scores: Show scores in each cell
        noshow:
        backend:
        format_string:

    Returns:
        Matplotlib's figure
    """
    import matplotlib

    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.Oranges

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
    else:
        accuracy = np.trace(cm) / (float(np.sum(cm)) + 1e-8)
        misclass = 1 - accuracy

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    if format_string is None:
        format_string = ".3f" if normalize else "d"

    if show_scores:
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            text = format(cm[i, j], format_string) if np.isfinite(cm[i, j]) else "N/A"
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, text, horizontalalignment="center", fontsize=fontsize, color=color)

    plt.ylabel("True label")

    if normalize:
        # We don't show Accuracy & Misclassification scores for normalized CM
        plt.xlabel("Predicted label")
    else:
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


def hstack_autopad(images: Iterable[np.ndarray], pad_value: int = 0, spacing=0) -> np.ndarray:
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
    for img_index, img in enumerate(images):
        is_last_image = img_index == len(images) - 1
        height = img.shape[0]
        pad_top = 0
        pad_bottom = max_height - height
        pad_left = 0
        pad_right = 0 if is_last_image else spacing
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
        (rows, cols) = img.shape[0:2]
        padded_images.append(img)

    return np.hstack(padded_images)


def vstack_autopad(images: Iterable[np.ndarray], pad_value: int = 0, spacing: int = 0) -> np.ndarray:
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
    for img_index, img in enumerate(images):
        is_last_image = img_index == len(images) - 1
        width = img.shape[1]
        pad_top = 0
        pad_bottom = 0 if is_last_image else spacing
        pad_left = 0
        pad_right = max_width - width
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
        padded_images.append(img)

    return np.vstack(padded_images)


def wrap_text_to_width(text, font_face, font_scale, thickness, max_width):
    """
    Splits a text string into multiple lines so that each line's width (using the provided text parameters)
    does not exceed max_width.

    Priority is given to whitespace when wrapping; if there are no spaces,
    it wraps at the character level.
    After lines are formed, leading and trailing whitespace is stripped.

    :param text: The input text to wrap.
    :param font_face: OpenCV font face (e.g., cv2.FONT_HERSHEY_SIMPLEX).
    :param font_scale: Font scale factor that is multiplied by the base font size.
    :param thickness: Thickness of the strokes used to draw text.
    :param max_width: Maximum allowed width (in pixels) for one line of text.
    :return: A list of text lines (strings).
    """

    # Early exit for empty text
    if not text.strip():
        return []

    # If the text has no whitespace, we switch to character-level wrapping
    has_whitespace = " " in text

    # Depending on presence of whitespace, choose how to split the text initially
    if has_whitespace:
        tokens = text.split(" ")
    else:
        # No whitespace - treat every character as a separate "token"
        tokens = list(text)

    lines = []
    current_line = ""

    def fits_in_width(candidate_text):
        """Check if candidate_text fits within max_width using cv2.getTextSize."""
        size, _ = cv2.getTextSize(candidate_text, font_face, font_scale, thickness)
        return size[0] <= max_width

    for i, token in enumerate(tokens):
        # If there is whitespace, tokens are words; otherwise tokens are individual characters.
        # We add a space if it's word-based wrapping (has_whitespace and not the very first word)
        if has_whitespace:
            tentative_line = (current_line + " " + token) if current_line else token
        else:
            # If we are in character-mode, do not prepend a space
            tentative_line = current_line + token

        # Check if the tentative line fits
        if fits_in_width(tentative_line):
            current_line = tentative_line
        else:
            # If it doesn't fit, we need to finalize the current_line and start a new one.
            if current_line:
                lines.append(current_line.strip())
            # In word-based mode, if a single token (word) doesn't fit on an empty line,
            # we might need to break it further by characters:
            if has_whitespace and not fits_in_width(token):
                # Break this word by character
                char_line = ""
                for ch in token:
                    if fits_in_width(char_line + ch):
                        char_line += ch
                    else:
                        if char_line:
                            lines.append(char_line.strip())
                        char_line = ch
                current_line = char_line  # start the next line with leftover
            else:
                # Start new current_line with the token
                current_line = token if not has_whitespace else token

    # Append any leftover text in current_line
    if current_line:
        lines.append(current_line.strip())

    # Strip each line (remove leading/trailing whitespace) just in case
    lines = [line.strip() for line in lines if line.strip()]

    return lines


def vstack_header(
    image: np.ndarray,
    title: str,
    bg_color=(35, 41, 40),
    text_color=(242, 248, 248),
    text_thickness: int = 2,
    text_scale=1.5,
    wrap_text: bool = False,
    text_font_face=cv2.FONT_HERSHEY_PLAIN,
) -> np.ndarray:
    (rows, cols) = image.shape[:2]

    image_width = image.shape[1]
    (width, height), baseline = cv2.getTextSize(title, text_font_face, text_scale, text_thickness)
    padding_left = 10
    padding_right = 10

    row_height = int(height * 2 + 0.5)

    if wrap_text and (width + padding_left + padding_right) > image_width:
        lines = wrap_text_to_width(
            title, text_font_face, text_scale, text_thickness, image_width - padding_left - padding_right
        )
    else:
        lines = [title]

    title_images = []

    for line in lines:
        title_image = np.zeros((row_height, cols, 3), dtype=np.uint8)
        title_image[:] = bg_color
        cv2.putText(
            title_image,
            line,
            (padding_left, row_height - int(height * 0.5)),
            fontFace=text_font_face,
            fontScale=text_scale,
            color=text_color,
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )
        title_images.append(title_image)

    title_image = np.vstack(title_images)
    return vstack_autopad([title_image, image])


def grid_stack(
    images: List[np.ndarray], rows: int = None, cols: int = None, bg_color=0, spacing: int = 0
) -> np.ndarray:
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
        image_rows.append(hstack_autopad(images[r * cols : (r + 1) * cols], bg_color=bg_color, spacing=spacing))

    return vstack_autopad(image_rows, bg_color=bg_color, spacing=spacing)
