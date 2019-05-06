from torch.nn import BCEWithLogitsLoss

from pytorch_toolbelt import losses as L
import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    losses = {
        "bce": BCEWithLogitsLoss(),
        "focal": L.BinaryFocalLoss(),
        "jaccard": L.BinaryJaccardLoss(),
        # "jaccard_log": L.BinaryJaccardLogLoss(),
        "lovasz": L.BinaryLovaszLoss(),
        # "bce+jaccard_log": L.BinaryJaccardLogLoss(),
        "reduced_focal": L.BinaryFocalLoss(reduced=True)
    }

    x_vec = torch.arange(-5, 5, 0.01)

    plt.figure()

    for name, loss in losses.items():
        x_arr = []
        y_arr = []
        target = torch.tensor(1.0)

        for x in x_vec:
            y = loss(x, target)

            x_arr.append(float(x))
            y_arr.append(float(y))

        plt.plot(x_arr, y_arr, label=name)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
