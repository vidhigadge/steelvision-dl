from PIL import Image


def degrade_image(image, low_res_size):
    low_res = image.resize(
        (low_res_size, low_res_size),
        Image.BICUBIC,
    )

    restored = low_res.resize(
        image.size,
        Image.BICUBIC,
    )

    return restored