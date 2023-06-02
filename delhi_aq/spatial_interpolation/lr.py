import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import argparse

aq_data = xr.open_dataset("delhi_aq/processed_data/delhi_cpcb_2022_cleaned.nc")
aq_df = (
    aq_data.sel(time="2022").to_dataframe().reset_index().set_index("time", drop=False)
)
result_df = aq_df.copy()
result_df.set_index(["time", "station"], inplace=True, drop=False)
result_df["PM2.5_pred"] = np.nan

x_scaler = MinMaxScaler()
y_scaler = StandardScaler()

focus_df = aq_df.dropna(subset=["PM2.5"])
# apply IDW to each time step
for time in tqdm(focus_df.index.unique()):
    tmp_df = focus_df.loc[time]
    for station in tmp_df.station:
        n_minus_1_df = tmp_df[tmp_df.station != station]
        model = LinearRegression()
        x = x_scaler.fit_transform(n_minus_1_df[["latitude", "longitude"]].values)
        y = y_scaler.fit_transform(n_minus_1_df["PM2.5"].values.reshape(-1, 1))
        model.fit(x, y)
        y_pred = model.predict(
            x_scaler.transform(
                tmp_df[tmp_df.station == station][["latitude", "longitude"]].values
            )
        )
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
        result_df.loc[(time, station), "PM2.5_pred"] = y_pred

result_df.to_csv(f"delhi_aq/spatial_interpolation/results/loocv_lr", index=None)
