import numpy as np
import flax.linen as nn


def pool_image(image, factor):
    return nn.avg_pool(image, window_shape=(factor, factor), strides=(factor, factor))


def get_coords_for_image(image, heigh_minmax=(-1, 1), width_minmax=(-1, 1)):
    x1 = np.linspace(*width_minmax, image.shape[0])
    x2 = np.linspace(*heigh_minmax, image.shape[1])
    X1, X2 = map(lambda x: x[..., None], np.meshgrid(x2, x1))
    coords = np.concatenate([X1, X2], axis=-1)
    return coords
