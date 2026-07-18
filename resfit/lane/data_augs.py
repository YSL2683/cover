import numpy as np


def random_crop(images, output_size=112):
    """
    args:
    images: np.array shape (B,C,H,W)
    output_size: output size, assuming square images
    returns: np.array
    """
    n, c, h, w = images.shape
    crop_max = h - output_size + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, output_size, output_size), dtype=images.dtype)
    for i, (image, w11, h11) in enumerate(zip(images, w1, h1)):
        cropped[i] = image[:, h11 : h11 + output_size, w11 : w11 + output_size]
    return cropped


def center_crop(image, output_size=112):
    h, w = image.shape[1:]
    assert h >= output_size
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top : top + new_h, left : left + new_w]
    return image


def batch_center_crop(images, output_size=112):
    n, c, h, w = images.shape
    cropped = np.empty((n, c, output_size, output_size), dtype=images.dtype)
    for i, image in enumerate(images):
        cropped[i] = center_crop(image, output_size)
    return cropped


def no_aug(x):
    return x
