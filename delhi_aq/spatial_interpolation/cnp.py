import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn

# define initializers
def first_layer_init(key, shape, dtype=jnp.float32):
    num_input = shape[0]  # reverse order compared to torch
    return jax.random.uniform(key, shape, dtype, minval=-1.0/num_input, maxval=1.0/num_input)

def other_layers_init(key, shape, dtype=jnp.float32):
    num_input = shape[0]  # reverse order compared to torch
    return jax.random.uniform(key, shape, dtype, minval=-np.sqrt(6 / num_input)/30, maxval=np.sqrt(6 / num_input)/30)

class Encoder(nn.Module):
  features: list
  encoding_dims: int

  @nn.compact
  def __call__(self, x_context, y_context, y_context_mask):
    x = jnp.hstack([x_context, y_context.reshape(x_context.shape[0], -1)])
    
    x = nn.Dense(self.features[0], kernel_init=first_layer_init, bias_init=first_layer_init)(x)
    x = jnp.sin(30*x)
    # x = nn.Dense(self.features[0])(x)
    # x = nn.relu(x)
    
    
    for n_features in self.features[1:]:
      x = nn.Dense(n_features, kernel_init=other_layers_init, bias_init=other_layers_init)(x)
      x = jnp.sin(30*x)
    #   x = nn.Dense(n_features)(x)
    #   x = nn.relu(x)

    x = nn.Dense(self.encoding_dims)(x)

    representation = (x * y_context_mask).sum(axis=0, keepdims=True) / y_context_mask.sum(axis=0, keepdims=True)
    return representation  # (1, encoding_dims)

class Decoder(nn.Module):
  features: list
  output_dim: int

  @nn.compact
  def __call__(self, representation, x):
    representation = jnp.repeat(representation, x.shape[0], axis=0)
    x = jnp.hstack([representation, x])
    
    x = nn.Dense(self.features[0], kernel_init=first_layer_init, bias_init=first_layer_init)(x)
    x = jnp.sin(30*x)
    # x = nn.Dense(self.features[0])(x)
    # x = nn.relu(x)

    for n_features in self.features:
      x = nn.Dense(n_features, kernel_init=other_layers_init, bias_init=other_layers_init)(x)
      x = jnp.sin(30*x)
      # x = nn.Dense(n_features)(x)
      # x = nn.relu(x)

    x = nn.Dense(self.output_dim)(x)
    return x

class CNP(nn.Module):
    encoder_features: list
    encoding_dims: int
    decoder_features: list
    output_dim: int

    @nn.compact
    def __call__(self, x_content, y_context, y_context_mask, x_target):
        representation = Encoder(self.encoder_features, self.encoding_dims)(x_content, y_context, y_context_mask)
        y_pred = Decoder(self.decoder_features, self.output_dim)(representation, x_target)
        return y_pred

    def loss_fn(self, params, x_content, y_context, y_context_mask, x_target, y_target):
        y_pred = self.apply(params, x_content, y_context, x_target, y_context_mask)
        loss = jnp.mean((y_pred - y_target)**2)
        return loss
