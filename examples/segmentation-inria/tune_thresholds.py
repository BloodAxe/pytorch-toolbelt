from models.factory import optimize_threshold

from pytorch_toolbelt.utils.fs import find_in_dir


def main():
    gt = sorted(find_in_dir('c:\\Develop\\data\\inria\\train\\gt'))
    pred = sorted(find_in_dir('runs\\Apr10_23_47_hdfpn_resnext50_finetune\\evaluation'))
    print(len(gt), len(pred))

    thresholds, ious = optimize_threshold(gt, pred)

    for t, iou in zip(thresholds, ious):
        print(t, iou)


if __name__ == '__main__':
    main()
