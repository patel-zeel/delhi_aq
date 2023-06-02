import numpy as np

# GPy
from GPy.models import GPRegression
from GPy.kern import RBF, Matern32, Matern52, Exponential

# polire
from polire import Kriging, IDW as PolireIDW

# sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NoScaler:
    def fit_transform(self, x):
        return x

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


class AQModel:
    def __init__(self, train_x, train_y, x_scaler, y_scaler):
        assert train_x.ndim >= 2, "train_x must be 2D or higher"
        assert train_y.ndim >= 2, "train_y must be 2D or higher"
        self.y_dim = train_y.shape[-1]

        self.fit_transform(train_x, train_y, x_scaler, y_scaler)

    def fit_transform(self, x, y, x_scaler, y_scaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        x = self.x_scaler.fit_transform(x)
        y = self.y_scaler.fit_transform(y)
        return x, y

    def transform(self, x, y):
        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)
        return x, y

    def inverse_transform(self, x, y):
        x = self.x_scaler.inverse_transform(x)
        y = self.y_scaler.inverse_transform(y)
        return x, y

    def fit(self, train_x, train_y):
        train_x, train_y = self.transform(train_x, train_y)
        self._fit(train_x, train_y)


class Mean(AQModel):
    def __init__(self, train_x, train_y):
        super().__init__(train_x, train_y, NoScaler(), NoScaler())

    def _fit(self, train_x, train_y):
        train_x, train_y = self.transform(train_x, train_y)
        self.mean = train_y.mean(axis=tuple(range(train_y.ndim - 1))).reshape(
            1, self.y_dim
        )

    def _predict(self, x_test):
        x_test, _ = self.transform(x_test, np.zeros((1, self.y_dim)))
        batch_dim = x_test.shape[:-1]
        y_pred = np.zeros(batch_dim + (self.y_dim,)) * np.nan
        y_pred[:] = self.mean
        _, y_pred = self.inverse_transform(x_test, y_pred)
        return y_pred


class IDW(AQModel):
    def __init__(self, train_x, train_y, p=2):
        assert (train_x.shape[-1] == 2) and (
            train_x.ndim == 2
        ), "train_x must have (n, 2) shape"
        assert (train_y.ndim == 2) and (
            train_y.shape[-1] == 1
        ), "train_y must have (n, 1) shape"
        assert p >= 1, "p must be >= 1"
        self.p = p

        super().__init__(train_x, train_y, MinMaxScaler(), NoScaler())

    def _fit(self, train_x, train_y):
        train_x, train_y = self.transform(train_x, train_y)
        self.idw = PolireIDW(exponent=self.p)
        self.idw.fit(train_x, train_y.ravel())

    def _predict(self, x_test):
        x_test, _ = self.transform(x_test, np.zeros((1, 1)))
        y_pred = self.idw.predict(x_test).reshape(-1, 1)
        _, y_pred = self.inverse_transform(x_test, y_pred)

        return y_pred
