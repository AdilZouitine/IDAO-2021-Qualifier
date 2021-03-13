import numpy as np
from scipy.stats import entropy, kurtosis, skew


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def compute_feature(image):
    image = crop_center(img=image, cropx=128, cropy=128)
    flat_image = image.flatten()
    return {
        "mean": flat_image.mean(),
        "std": flat_image.std(),
        "min": flat_image.min(),
        "max": flat_image.max(),
        "q005": np.quantile(flat_image, 0.05),
        "q025": np.quantile(flat_image, 0.25),
        "q05": np.quantile(flat_image, 0.5),
        "q075": np.quantile(flat_image, 0.75),
        "q095": np.quantile(flat_image, 0.95),
        "entropy": entropy(flat_image),
        "kurtosis": kurtosis(flat_image),
        "skewness": skew(flat_image),
    }