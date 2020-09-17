from __future__ import absolute_import

import itertools
import warnings
import numpy as np

from .torch_utils import tensor_from_rgb_image


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

    f = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    fmt = ".3f" if normalize else "d"
    thresh = (cm.max() - cm.min()) / 2.0
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

    image = tensor_from_rgb_image(image)
    return image
