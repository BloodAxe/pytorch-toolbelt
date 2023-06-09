import pytest
import torch
from pytorch_toolbelt.modules import HFF, ResidualDeconvolutionUpsample2d, GlobalKMaxPool2d

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda is not available")


def test_hff_dynamic_size():
    feature_maps = [
        torch.randn((4, 3, 512, 512)),
        torch.randn((4, 3, 256, 256)),
        torch.randn((4, 3, 128, 128)),
        torch.randn((4, 3, 64, 64)),
    ]

    hff = HFF(upsample_scale=2)
    output = hff(feature_maps)
    assert output.size(2) == 512
    assert output.size(3) == 512


def test_hff_static_size():
    feature_maps = [
        torch.randn((4, 3, 512, 512)),
        torch.randn((4, 3, 384, 384)),
        torch.randn((4, 3, 256, 256)),
        torch.randn((4, 3, 128, 128)),
        torch.randn((4, 3, 32, 32)),
    ]

    hff = HFF(sizes=[(512, 512), (384, 384), (256, 256), (128, 128), (32, 32)])
    output = hff(feature_maps)
    assert output.size(2) == 512
    assert output.size(3) == 512


# def test_upsample():
#     block = DepthToSpaceUpsample2d(1)
#     original = np.expand_dims(cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE), -1)
#     input = tensor_from_rgb_image(original / 255.0).unsqueeze(0).float()
#     output = block(input)
#
#     output_rgb = rgb_image_from_tensor(output.squeeze(0), mean=0, std=1, max_pixel_value=1, dtype=np.float32)
#
#     cv2.imshow("Original", original)
#     cv2.imshow("Upsampled (cv2)", cv2.resize(original, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR))
#     cv2.imshow("Upsampled", cv2.normalize(output_rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
#     cv2.waitKey(-1)


def test_residualdeconvolutionupsampleblock():
    x = torch.randn((4, 16, 32, 32))
    block = ResidualDeconvolutionUpsample2d(16)
    output = block(x, output_size=None)
    print(x.size(), x.mean(), x.std())
    print(output.size(), output.mean(), x.std())


def test_kmax_pool():
    x = torch.randn((8, 512, 16, 16))
    module1 = GlobalKMaxPool2d(k=4, flatten=True)
    module2 = GlobalKMaxPool2d(k=4, flatten=False)

    y1 = module1(x)
    y2 = module2(x)

    assert y1.size() == (8, 512)
    assert y2.size() == (8, 512, 1, 1)
