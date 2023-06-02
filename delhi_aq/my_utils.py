import pickle
import jax.numpy as jnp
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


def get_save_dir(config):
    keys = sorted(config.keys())
    values = [config[k] for k in keys]
    save_dir = "_||_".join([f"{k}={v}" for k, v in zip(keys, values)])
    return save_dir


def rmse_fn(x, y):
    if isinstance(x, np.ndarray):
        sqr = (x.ravel() - y.ravel()) ** 2
        sqr = np.where(np.isnan(sqr), 0, sqr)
        return np.sqrt(np.mean(sqr))
    elif isinstance(x, jnp.ndarray):
        sqr = (x.ravel() - y.ravel()) ** 2
        sqr = jnp.where(jnp.isnan(sqr), 0, sqr)
        return jnp.sqrt(jnp.mean(sqr))
    else:
        raise NotImplementedError(f"rmse_fn not implemented for {type(x)}")
