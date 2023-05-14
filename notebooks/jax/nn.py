from time import time
import numpy as np

import optax
import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.core import freeze, unfreeze


class NeRFReLU(nn.Module):
    n_hidden_layer_neurons: list
    output_shape: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden_layer_neurons[0])(x)
        x = nn.relu(x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(self.n_hidden_layer_neurons[i])(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_shape)(x)
        return x


# define initializers
def first_layer_init(key, shape, dtype=jnp.float32):
    num_input = shape[0]  # reverse compared to pytorch
    return jax.random.uniform(key, shape, dtype, minval=-1.0 / num_input, maxval=1.0 / num_input)


def other_layers_init(key, shape, dtype=jnp.float32):
    num_input = shape[0]  # reverse compared to pytorch
    return jax.random.uniform(
        key,
        shape,
        dtype,
        minval=-jnp.sqrt(6 / num_input) / 30,
        maxval=jnp.sqrt(6 / num_input) / 30,
    )


class SIREN(nn.Module):
    n_hidden_layer_neurons: list
    output_shape: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden_layer_neurons[0], kernel_init=first_layer_init)(x)
        x = jnp.sin(30 * x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(self.n_hidden_layer_neurons[i], kernel_init=other_layers_init)(x)
            x = jnp.sin(30 * x)
        x = nn.Dense(self.output_shape, kernel_init=other_layers_init)(x)
        return x

    def vapply(self, params, inputs, rngs=None):
        if len(inputs.shape) == 3:
            apply_fn = lambda x: self.apply(params, x, rngs=rngs)
            return jax.vmap(apply_fn)(inputs)
        else:
            return self.apply(params, inputs, rngs=rngs)


def fit(key, model, train_x, train_y, lr, batch_size, iterations):
    train_x = jnp.asarray(train_x)
    train_y = jnp.asarray(train_y)

    # initialize params
    params = model.init(key, jnp.ones((1, train_x.shape[-1])))

    # loss fun
    def loss_fn(params, x, y, key):
        y_hat = model.apply(params, x, rngs={"dropout": key})
        loss = (y - y_hat) ** 2
        return jnp.mean(loss)

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    optimizer = optax.adam(lr)
    state = optimizer.init(params)

    # lax scan loop
    @jax.jit
    def one_step(params_and_state, key):
        params, state = params_and_state
        if batch_size == -1:
            x = train_x
            y = train_y
        elif batch_size < train_y.shape[0]:
            batch_idx = jax.random.choice(key, jnp.arange(train_x.shape[0]), shape=(batch_size,), replace=False)
            x = train_x[batch_idx]
            y = train_y[batch_idx]
        else:
            raise ValueError(f"batch_size {batch_size} is larger than the number of samples {train_y.shape[0]}")

        value, grads = value_and_grad_fn(params, x, y, key)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    keys = jax.random.split(key, iterations)
    (params, state), losses = jax.lax.scan(one_step, (params, state), xs=keys)
    return params, losses


class Encoder(nn.Module):
    features: list
    encoding_dims: int

    @nn.compact
    def __call__(self, x_context, y_context):
        x = jnp.hstack([x_context, y_context.reshape(x_context.shape[0], -1)])
        for n_features in self.features:
            x = nn.Dense(n_features)(x)
            x = nn.relu(x)

        x = nn.Dense(self.encoding_dims)(x)

        representation = x.mean(axis=0, keepdims=True)  # option 1
        return representation  # (1, encoding_dims)
