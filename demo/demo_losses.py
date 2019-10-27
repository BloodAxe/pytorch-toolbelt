from torch.nn import BCEWithLogitsLoss

from pytorch_toolbelt import losses as L
import numpy as np
import torch
import matplotlib.pyplot as plt


def main():
    losses = {
        "bce": BCEWithLogitsLoss(),
        # "focal": L.BinaryFocalLoss(),
        # "jaccard": L.BinaryJaccardLoss(),
        # "jaccard_log": L.BinaryJaccardLogLoss(),
        # "dice": L.BinaryDiceLoss(),
        # "dice_log": L.BinaryDiceLogLoss(),
        # "sdice": L.BinarySymmetricDiceLoss(),
        # "sdice_log": L.BinarySymmetricDiceLoss(log_loss=True),
        "bce+lovasz": L.JointLoss(BCEWithLogitsLoss(), L.BinaryLovaszLoss()),
        # "lovasz": L.BinaryLovaszLoss(),
        # "bce+jaccard": L.JointLoss(BCEWithLogitsLoss(),
        #                            L.BinaryJaccardLoss(), 1, 0.5),
        # "bce+log_jaccard": L.JointLoss(BCEWithLogitsLoss(),
        #                            L.BinaryJaccardLogLoss(), 1, 0.5),
        # "bce+log_dice": L.JointLoss(BCEWithLogitsLoss(),
        #                                L.BinaryDiceLogLoss(), 1, 0.5)
        # "reduced_focal": L.BinaryFocalLoss(reduced=True)
    }

    dx = 0.01
    x_vec = torch.arange(-5, 5, dx).view(-1, 1).expand((-1, 100))

    f, ax = plt.subplots(3, figsize=(16, 16))

    for name, loss in losses.items():
        x_arr = []
        y_arr = []
        target = torch.tensor(1.0).view(1).expand((100))

        for x in x_vec:
            y = loss(x, target).item()

            x_arr.append(float(x[0]))
            y_arr.append(float(y))

        ax[0].plot(x_arr, y_arr, label=name)
        ax[1].plot(x_arr, np.gradient(y_arr, dx))
        ax[2].plot(x_arr, np.gradient(np.gradient(y_arr, dx), dx))

    f.legend()
    f.show()


if __name__ == "__main__":
    main()
