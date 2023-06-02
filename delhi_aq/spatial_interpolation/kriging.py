import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm
from polire import Kriging
from multiprocessing import Pool

aq_data = xr.open_dataset("delhi_aq/processed_data/delhi_cpcb_2022_cleaned.nc")
aq_df = (
    aq_data.sel(time="2022").to_dataframe().reset_index().set_index("time", drop=False)
)
result_df = aq_df.copy()
result_df.set_index(["time", "station"], inplace=True, drop=False)
result_df["PM2.5_pred"] = np.nan

focus_df = aq_df.dropna(subset=["PM2.5"])
# apply IDW to each time step
for time in tqdm(focus_df.index.unique()):
    tmp_df = focus_df.loc[time]

    def process(station):
        n_minus_1_df = tmp_df[tmp_df.station != station]
        model = Kriging(variogram_model="spherical")
        x = n_minus_1_df[["latitude", "longitude"]].values
        y = n_minus_1_df["PM2.5"].values.reshape(-1, 1)

        xscaler = MinMaxScaler()
        yscaler = StandardScaler()

        x = xscaler.fit_transform(x)
        y = yscaler.fit_transform(y)
        x_test = tmp_df[tmp_df.station == station][["latitude", "longitude"]].values
        x_test = xscaler.transform(x_test)

        model.fit(x, y.ravel())
        y_pred = model.predict(x_test)
        y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred

    preds = Pool(24).map(process, tmp_df.station)
    for station, pred in zip(tmp_df.station, preds):
        result_df.loc[(time, station), "PM2.5_pred"] = pred

result_df.to_csv(f"delhi_aq/spatial_interpolation/results/loocv_kriging", index=None)
