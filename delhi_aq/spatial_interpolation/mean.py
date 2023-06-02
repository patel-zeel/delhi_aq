import xarray as xr
import numpy as np
import pandas as pd

from tqdm import tqdm
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

    for station in tmp_df.station:
        n_minus_1_df = tmp_df[tmp_df.station != station]
        x = n_minus_1_df[["latitude", "longitude"]].values
        y = n_minus_1_df["PM2.5"].values.reshape(-1, 1)
        y_pred = np.mean(y)
        result_df.loc[(time, station), "PM2.5_pred"] = y_pred

result_df.to_csv(f"delhi_aq/spatial_interpolation/results/loocv_mean", index=None)