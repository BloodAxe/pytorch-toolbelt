import numpy as np

from pytorch_toolbelt.utils import plot_confusion_matrix, plot_heatmap


def test_plot_confusion_matrix():
    cm = np.random.randint(0, 7, (7, 7))

    plot_confusion_matrix(
        cm, class_names=["A", "B", "C", "D", "E", "F", "G"], normalize=True, fname="test_plot_confusion_matrix.png"
    )


def test_plot_heatmap():
    cm = np.random.randn(20, 30)

    plot_heatmap(cm, title="Test", x_label="30", y_label="20", fname="test_plot_heatmap.png", noshow=False)
