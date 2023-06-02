import os
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jax.tree_util as jtu

from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from delhi_aq.jax.my_nn import SIREN, fit, NeRFGeLU, NeRFPEGeLU
from delhi_aq.jax.my_utils import (
    pool_image,
    get_coords_for_image,
    rmse_fn,
    get_save_dir,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

self_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(self_dir, "config.yml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

aq_data = (
    xr.open_dataset("../processed_data/delhi_cpcb_2022_cleaned.nc")
    .to_dataframe()
    .reset_index()
    .set_index("time", drop=False)
)
aq_data["time"] = aq_data["time"].astype(int)
aq_data = aq_data.dropna(subset="PM2.5")

config["features"] = sorted(config["features"])
x = aq_data[config["features"]].values
y = aq_data[[config["target"]]].values

train_val_x_, test_x_, train_val_y_, test_y_ = train_test_split(
    x, y, test_size=config["test_size"], random_state=config["seed"]
)
train_x_, val_x_, train_y_, val_y_ = train_test_split(
    train_val_x_,
    train_val_y_,
    test_size=config["val_size"],
    random_state=config["seed"],
)

x_scaler = MinMaxScaler((-1, 1))
y_scaler = MinMaxScaler((0, 1))
time_scale = 20
train_x = x_scaler.fit_transform(train_x_)
train_x[:, -1] = train_x[:, -1] * time_scale
val_x = x_scaler.transform(val_x_)
val_x[:, -1] = val_x[:, -1] * time_scale
test_x = x_scaler.transform(test_x_)
test_x[:, -1] = test_x[:, -1] * time_scale
train_y = y_scaler.fit_transform(train_y_)
val_y = y_scaler.transform(val_y_)
test_y = y_scaler.transform(test_y_)
# print(train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)

with open(os.path.join(self_dir, f"{args.model}.yml"), "r") as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)

if args.model == "siren":
    model = SIREN(
        n_hidden_layer_neurons=model_config["layers"],
        output_shape=train_y_.shape[-1],
        activation_scale=model_config["activation_scale"],
    )
elif args.model == "nerf":
    model = NeRFGeLU(
        n_hidden_layer_neurons=model_config["layers"], output_shape=train_y_.shape[-1]
    )
elif args.model == "nerfpe":
    model = NeRFPEGeLU(
        n_hidden_layer_neurons=model_config["layers"],
        output_shape=train_y_.shape[-1],
        activation_scale=model_config["activation_scale"],
    )

params, old_params_history, train_losses, val_losses, test_losses = fit(
    jax.random.PRNGKey(config["seed"]),
    model,
    train_x,
    train_y,
    model_config,
    val_x,
    val_y,
    test_x,
    test_y,
)

best_val_idx = jnp.argmin(val_losses)
best_val_loss = val_losses[best_val_idx] * (y_scaler.data_max_ - y_scaler.data_min_)
best_test_loss = test_losses[best_val_idx] * (y_scaler.data_max_ - y_scaler.data_min_)
best_train_loss = train_losses[best_val_idx] * (y_scaler.data_max_ - y_scaler.data_min_)

results_path = os.path.join(self_dir, "results")
exp_dir = get_save_dir(config)
model_dir = get_save_dir(model_config)
save_dir = os.path.join(results_path, exp_dir, model_dir)
