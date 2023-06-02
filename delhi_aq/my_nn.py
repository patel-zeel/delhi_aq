from time import time
import numpy as np

import optax
import jax.numpy as jnp
import jax
import flax.linen as nn
from flax.core import freeze, unfreeze

from .my_utils import rmse_fn


class NeRF(nn.Module):
    n_hidden_layer_neurons: list
    output_shape: int
    activation: str

    @nn.compact
    def __call__(self, x):
        activation = getattr(nn, self.activation)
        x = nn.Dense(self.n_hidden_layer_neurons[0])(x)
        x = activation(x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(self.n_hidden_layer_neurons[i])(x)
            x = activation(x)
        x = nn.Dense(self.output_shape)(x)
        return x


class NeRFPE(nn.Module):
    n_hidden_layer_neurons: list
    output_shape: int
    activation: str
    activation_scale: float

    @nn.compact
    def __call__(self, x):
        activation = getattr(nn, self.activation)
        x = nn.Dense(32, use_bias=False)(x)
        x = jnp.concatenate(
            (jnp.sin(self.activation_scale * x), jnp.cos(self.activation_scale * x)),
            axis=-1,
        )
        x = nn.Dense(self.n_hidden_layer_neurons[0])(x)
        x = activation(x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(self.n_hidden_layer_neurons[i])(x)
            x = activation(x)
        x = nn.Dense(self.output_shape)(x)
        return x


# define initializers
def first_layer_init(key, shape, dtype=jnp.float32):
    num_input = shape[0]  # reverse compared to pytorch
    return jax.random.uniform(
        key, shape, dtype, minval=-1.0 / num_input, maxval=1.0 / num_input
    )


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
    activation_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden_layer_neurons[0], kernel_init=first_layer_init)(x)
        x = jnp.sin(self.activation_scale * x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(self.n_hidden_layer_neurons[i], kernel_init=other_layers_init)(
                x
            )
            x = jnp.sin(self.activation_scale * x)
        x = nn.Dense(self.output_shape, kernel_init=other_layers_init)(x)
        return x

    def vapply(self, params, inputs, rngs=None):
        if len(inputs.shape) == 3:
            apply_fn = lambda x: self.apply(params, x, rngs=rngs)
            return jax.vmap(apply_fn)(inputs)
        else:
            return self.apply(params, inputs, rngs=rngs)


def fit(
    key,
    model,
    train_x,
    train_y,
    config,
    val_x=None,
    val_y=None,
    test_x=None,
    test_y=None,
):
    lr, batch_size, iterations = (
        config["lr"],
        config["batch_size"],
        config["iterations"],
    )

    train_x = jnp.asarray(train_x)
    train_y = jnp.asarray(train_y)
    if val_x is not None:
        val_x = jnp.asarray(val_x)
        val_y = jnp.asarray(val_y)
    if test_x is not None:
        test_x = jnp.asarray(test_x)
        test_y = jnp.asarray(test_y)

    # initialize params
    params = model.init(key, jnp.ones((1, train_x.shape[-1])))

    # loss fun
    @jax.jit
    def loss_fn(params, x, y, key):
        y_hat = model.apply(params, x, rngs={"dropout": key})
        return rmse_fn(y, y_hat)

    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # warmup_scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=lr, peak_value=lr * 100, warmup_steps=int(iterations * 0.1), decay_steps=iterations, end_value=lr
    # )
    # warmup_scheduler = optax.exponential_decay(lr, transition_steps=iterations, decay_rate=0.1)
    optimizer = optax.adam(lr)
    state = optimizer.init(params)

    # lax scan loop
    @jax.jit
    def one_step(params_and_state, key):
        old_params, state = params_and_state
        if batch_size == -1:
            x = train_x
            y = train_y
        elif batch_size < train_y.shape[0]:
            batch_idx = jax.random.choice(
                key, jnp.arange(train_x.shape[0]), shape=(batch_size,), replace=False
            )
            x = train_x[batch_idx]
            y = train_y[batch_idx]
        else:
            raise ValueError(
                f"batch_size {batch_size} is larger than the number of samples {train_y.shape[0]}"
            )

        train_loss, grads = value_and_grad_fn(old_params, x, y, key)
        val_loss = (
            loss_fn(old_params, val_x, val_y, key) if val_x is not None else jnp.nan
        )
        test_loss = (
            loss_fn(old_params, test_x, test_y, key) if test_x is not None else jnp.nan
        )

        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(old_params, updates)

        return (params, state), (old_params, train_loss, val_loss, test_loss)

    keys = jax.random.split(key, iterations)
    (params, state), (
        old_params_history,
        train_losses,
        val_losses,
        test_losses,
    ) = jax.lax.scan(one_step, (params, state), xs=keys)
    return params, old_params_history, train_losses, val_losses, test_losses


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
