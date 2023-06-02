import os
import argparse

import yaml
import pandas as pd
import numpy as np
import xarray as xr

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import GPy

from delhi_aq.jax.my_nn import SIREN, fit
from delhi_aq.jax.my_utils import pool_image, get_coords_for_image, rmse_fn, get_save_dir

import matplotlib.pyplot as plt
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["siren", "rf", "gp"])
parser.add_argument("--config", type=str)
parser.add_argument("--model_config", type=str)
args = parser.parse_args()
config = yaml.load(open(args.config, "r"))
model_config = yaml.load(open(args.model_config, "r"))

## Get paths
self_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(self_path, "results")

## Load data
aq_data = xr.open_dataset("../processed_data/delhi_cpcb_2022_cleaned.nc")
aq_df = aq_data.to_dataframe().reset_index().set_index("time")

## Data selection and processing
useful_df = aq_df[f"2022-{config['month'].zfill(2)}"]
time_stamp = useful_df.sample(1, random_state=config["seed"]).index[0]
print(f"Time stamp: {time_stamp}")

training_df = useful_df[useful_df.index == time_stamp].dropna()

## Train, val, test split
config["features"] = sorted(config["features"])
x = training_df[["station"] + config["features"]].values
y = training_df[[config["target"]]].values

train_val_x_, test_x_, train_val_y_, test_y_ = train_test_split(x, y, test_size=config["test_size"], random_state=config["seed"])
train_x_, val_x_, train_y_, val_y_ = train_test_split(train_val_x_, train_val_y_, test_size=config["val_size"], random_state=config["seed"])
x_scaler = MinMaxScaler((-1, 1))
y_scaler = StandardScaler()
train_x = x_scaler.fit_transform(train_x_.drop(columns=["station"]))
val_x, test_x = map(x_scaler.transform, (val_x_.drop(columns=["station"]), test_x_.drop(columns=["station"])))
train_y = y_scaler.fit_transform(train_y_)
val_y, test_y = map(y_scaler.transform, (val_y_, test_y_))
print(train_x.shape, val_x.shape, test_x.shape, train_y.shape, val_y.shape, test_y.shape)

if args.model == "siren":
    model = SIREN(n_hidden_layer_neurons=model_config["layers"], output_shape=y.shape[-1], activation_scale=model_config["activation_scale"])
    # Run it with a separate process
    def local_fit(gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        results = fit(jax.random.PRNGKey(model_config["seed"]), model, train_x, train_y, model_config, val_x, val_y, test_x, test_y)
        pred_train = y_scaler.inverse_transform(model(train_x))
        pred_val = y_scaler.inverse_transform(model(val_x))
        pred_test = y_scaler.inverse_transform(model(test_x))
        return results, pred_train, pred_val, pred_test
    
    results, pred_train, pred_val, pred_test = Pool(1).apply(local_fit)
    params, old_params_history, train_losses, val_losses, test_losses = results
    
    
    
    exp_dir = get_save_dir(config)
    model_dir = get_save_dir(model_config)
    save_dir = os.path.join(results_path, exp_dir, model_dir)
    