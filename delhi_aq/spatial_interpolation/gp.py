from time import time
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
import GPy
from multiprocessing import Pool

aq_data = xr.open_dataset("delhi_aq/processed_data/delhi_cpcb_2022_cleaned.nc")
aq_df = (
    aq_data.sel(time="2022-01-01")
    .to_dataframe()
    .reset_index()
    .set_index("time", drop=False)
)
result_df = aq_df.copy()
result_df.set_index(["time", "station"], inplace=True, drop=False)
result_df["PM2.5_pred"] = np.nan

focus_df = aq_df.dropna(subset=["PM2.5"])


# apply IDW to each time step
def process(station_and_time):
    station, time = station_and_time
    tmp_df = focus_df.loc[time]
    n_minus_1_df = tmp_df[tmp_df.station != station]
    x = n_minus_1_df[["latitude", "longitude"]].values
    y = n_minus_1_df["PM2.5"].values.reshape(-1, 1)

    xscaler = MinMaxScaler()
    yscaler = StandardScaler()
    x = xscaler.fit_transform(x)
    y = yscaler.fit_transform(y)

    model = GPy.models.GPRegression(
        x, y, kernel=GPy.kern.Matern32(x.shape[1], ARD=True)
    )
    model.optimize()
    # model.optimize_restarts(num_restarts=5, verbose=False)

    x_test = tmp_df[tmp_df.station == station][["latitude", "longitude"]].values
    x_test = xscaler.transform(x_test)
    y_pred, y_var = model.predict(x_test)
    y_pred = yscaler.inverse_transform(y_pred)
    y_var = y_var * yscaler.scale_**2

    print(station, time)
    return y_pred, y_var


init = time()
preds = Pool(32).map(process, zip(focus_df.station, focus_df.time), chunksize=256)
result_df["PM2.5_pred"] = np.concatenate([pred[0] for pred in preds])
result_df["PM2.5_var"] = np.concatenate([pred[1] for pred in preds])

print(f"Time taken in minutes {(time() - init)/60:.2f}")

result_df.to_csv(f"delhi_aq/spatial_interpolation/results/loocv_gp_m32", index=None)
