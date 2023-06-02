import pytest

import numpy as np
from delhi_aq.models import Mean, IDW

n = 100
m = 10
d1 = 3
d2 = 4

x_n_1 = np.random.rand(n, 1)
y_n_1 = np.random.rand(n, 1)
x_m_1 = np.random.rand(m, 1)
y_n = y_n_1.flatten()

x_n_2 = np.random.rand(n, 2)
y_n_2 = np.random.rand(n, 2)
x_m_2 = np.random.rand(m, 2)

x_n_m_d1 = np.random.rand(n, m, d1)
y_n_m_d2 = np.random.rand(n, m, d2)
x_nm1_mm1_d1 = np.random.rand(n - 1, m - 1, d1)


@pytest.mark.parametrize(
    ["x", "y", "x_test", "pred_shape"],
    [
        (x_n_1, y_n_1, x_m_1, (m, 1)),
        (x_n_2, y_n_1, x_m_2, (m, 1)),
        (x_n_1, y_n, x_m_1, (m, 1)),
        (x_n_2, y_n_2, x_m_2, (m, 2)),
        (x_n_m_d1, y_n_m_d2, x_nm1_mm1_d1, (n - 1, m - 1, d2)),
    ],
)
def test_mean(x, y, x_test, pred_shape):
    if y.ndim == 1:
        pytest.raises(AssertionError)
        return
    mean_model = Mean(x, y)
    mean_model.fit(x, y)
    y_pred = mean_model.predict(x_test)
    assert y_pred.shape == pred_shape
    assert np.all(np.isnan(y_pred) == False)
    assert np.all(y_pred == mean_model.mean)


@pytest.mark.parametrize(
    ["x", "y", "x_test", "pred_shape"],
    [
        (x_n_1, y_n_1, x_m_1, (m, 1)),
        (x_n_2, y_n_1, x_m_2, (m, 1)),
        (x_n_1, y_n, x_m_1, (m, 1)),
        (x_n_2, y_n_2, x_m_2, (m, 2)),
        (x_n_m_d1, y_n_m_d2, x_nm1_mm1_d1, (n - 1, m - 1, d2)),
    ],
)
def test_idw(x, y, x_test, pred_shape):
    if y.ndim == 1:
        pytest.raises(AssertionError)
        return
    if y.shape[-1] != 1:
        pytest.raises(AssertionError)
        return
    if x.ndim != 2:
        pytest.raises(AssertionError)
        return
    if x.shape[-1] != 2:
        pytest.raises(AssertionError)
        return
    idw_model = IDW(x, y)
    idw_model.fit(x, y)
    y_pred = idw_model.predict(x_test)
    assert y_pred.shape == pred_shape
    assert np.all(np.isnan(y_pred) == False)
