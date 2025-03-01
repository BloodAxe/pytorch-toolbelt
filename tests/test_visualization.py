import matplotlib.pyplot as plt
import numpy as np

from pytorch_toolbelt.utils import plot_confusion_matrix, plot_heatmap, vstack_header


def test_plot_confusion_matrix():
    cm = np.random.randint(0, 7, (7, 7))

    plot_confusion_matrix(
        cm, class_names=["A", "B", "C", "D", "E", "F", "G"], normalize=True, fname="test_plot_confusion_matrix.png"
    )


def test_plot_heatmap():
    cm = np.random.randn(20, 30)

    plot_heatmap(cm, title="Test", x_label="30", y_label="20", fname="test_plot_heatmap.png", noshow=False)


def test_vstack_header():
    title = "A very long header text that would not fit in a single line and should be wrapped into multiple lines to fit the plot"
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    image2 = vstack_header(image, title, text_scale=2, wrap_text=True)
    plt.figure()
    plt.imshow(image2)
    plt.show()