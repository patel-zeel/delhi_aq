import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from time import time
import xarray as xr
import numpy as np

import jax
import jax.numpy as jnp

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

from cnp import CNP
import optax

# ignore deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


aq_data = xr.open_dataset("delhi_aq/processed_data/delhi_cpcb_2022_cleaned.nc")
aq_df = (
    aq_data.sel(time="2022-03-01 00:00:00", method="nearest")
    .to_dataframe()
    .reset_index()
    .set_index("time", drop=False)
)
# aq_df = aq_df.dropna(subset=["PM2.5"])

all_stations = aq_data.station.values
np.random.seed(2)
all_stations = np.random.permutation(all_stations)

test_stations = all_stations[0:1]
train_stations = all_stations[1:]

train_df = aq_df[aq_df.station.isin(train_stations)]
test_df = aq_df[aq_df.station.isin(test_stations)]
train_coords_df = train_df.drop_duplicates(subset=["station"]).set_index("station")
test_coords_df = test_df.drop_duplicates(subset=["station"]).set_index("station")
train_coords = train_coords_df.loc[train_stations][["latitude", "longitude"]].values
test_coords = test_coords_df.loc[test_stations][["latitude", "longitude"]].values

train_y_list = []
mask_list = []
for time_stamp in tqdm(train_df.time.unique()):
    tmp_df = train_df.loc[time_stamp]
    tmp_df.set_index("station", inplace=True)
    coords = tmp_df.loc[train_stations][["latitude", "longitude"]].values
    train_y = tmp_df.loc[train_stations]["PM2.5"].values
    mask = (~np.isnan(train_y)).astype(np.int).astype(np.float32)
    train_y_list.append(train_y[None, ..., None])
    mask_list.append(mask[None, ..., None])

train_y = np.concatenate(train_y_list, axis=0)
mask = np.concatenate(mask_list, axis=0)


x_scaler = MinMaxScaler((-1, 1))

train_x = x_scaler.fit_transform(train_coords)
test_x = x_scaler.transform(test_coords)
y_min = np.nanmin(train_y)
y_max = np.nanmax(train_y)
train_y = (train_y - y_min) / (y_max - y_min)

train_y[np.isnan(train_y)] = -1e10

train_x, train_y, mask, test_x = jax.tree_map(
    jnp.array, (train_x, train_y, mask, test_x)
)

print(train_x.shape, train_y.shape)

model = CNP([256] * 2, 128, [256] * 4, 1)
params = model.init(jax.random.PRNGKey(0), train_x, train_y[0], train_y[0], train_x)
optimizer = optax.adam(1e-6)
state = optimizer.init(params)


def loss_fn(
    params, context_x, context_y, context_mask, target_x, target_y, target_mask
):
    def loss_fn_per_sample(context_y, context_mask, target_y, target_mask):
        pred_y = model.apply(params, context_x, context_y, context_mask, target_x)
        loss = jnp.square(pred_y - target_y)
        loss = (loss * target_mask).sum() / target_mask.sum()
        return loss

    loss = jax.vmap(loss_fn_per_sample)(context_y, context_mask, target_y, target_mask)
    return loss.mean()


### Debug model
# loss = loss_fn(params, train_x, train_y, mask, train_x, train_y, mask)
###


value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))


def one_step(params_and_state, key):
    params, state = params_and_state
    tmp_index = jax.random.permutation(key, train_x.shape[0])
    context_x = train_x[tmp_index][n_targets:]
    context_y = train_y[tmp_index][:, n_targets:, :]
    context_mask = mask[tmp_index][:, n_targets:, :]
    target_x = train_x[tmp_index][:n_targets]
    target_y = train_y[tmp_index][:, :n_targets, :]
    target_mask = mask[tmp_index][:, :n_targets, :]
    context_x = context_x - target_x
    loss, grads = value_and_grad_fn(
        params, context_x, context_y, context_mask, target_x, target_y, target_mask
    )
    updates, state = optimizer.update(grads, state)
    params = optax.apply_updates(params, updates)
    return (params, state), loss


iterations = 50000
n_targets = 1

init = time()
(params, state), loss_history = jax.lax.scan(
    one_step, (params, state), jax.random.split(jax.random.PRNGKey(0), iterations)
)

np.save(
    f"delhi_aq/spatial_interpolation/results/cnp_loss_history_{test_stations}",
    loss_history.__array__(),
)
print(f"Training finished in {(time()-init)/60:.2f} minutes")

## testing
pred_y = jax.vmap(model.apply, in_axes=(None, None, 0, 0, None))(
    params, train_x - test_x, train_y, mask, test_x
).__array__()
pred_y = pred_y * (y_max - y_min) + y_min
print(pred_y.shape)
result_series = pd.Series(pred_y.squeeze(), index=aq_df.index.unique())

result_series.to_csv(
    f"delhi_aq/spatial_interpolation/results/cnp_result_{test_stations}.csv"
)

print("Done")
