from typing import Callable

import numpy as np
import pandas as pd


# def constant(r: np.ndarray) -> np.ndarray:
#     return np.ones(r.shape[1], dtype=int)


def momentum(r: np.ndarray) -> np.ndarray:
    return (1.0 + r).prod(axis=0) - 1.0


def drawdown(r: np.ndarray) -> np.ndarray:
    r_cum = (1.0 + r).cumprod(axis=0)
    r_max = np.maximum.accumulate(r_cum, axis=0)
    dd = 1.0 - r_cum / (r_max + 1e-8)
    return dd.max(axis=0)


def momentum_vs(r: np.ndarray) -> np.ndarray:
    return momentum(r) / (volatility(r) * np.sqrt(r.shape[0]) + 1e-8)


def volatility(r: np.ndarray) -> np.ndarray:
    return r.std(axis=0, ddof=1)


def volatility_downside(r: np.ndarray) -> np.ndarray:
    r_neg: np.ndarray = np.minimum(r, 0.0)
    return np.sqrt((r_neg**2).mean(axis=0))


def mean_geom(r: np.ndarray) -> np.ndarray:
    return (1.0 + r).prod(axis=0) ** (1 / r.shape[0]) - 1.0


def sharpe_like(r: np.ndarray) -> np.ndarray:
    return mean_geom(r) / (volatility(r) + 1e-8)


def skewness(r: np.ndarray) -> np.ndarray:
    u = r.mean(axis=0)
    s = r.std(axis=0, ddof=0)
    return np.mean((r - u) ** 3, axis=0) / (s + 1e-8) ** 3
    # return ((r - mean_geom(r))**3).mean(axis=0) / (volatility(r) + 1e-8)**3


def kurtosis(r: np.ndarray) -> np.ndarray:
    u = r.mean(axis=0)
    s = r.std(axis=0, ddof=0)
    return np.mean((r - u) ** 4, axis=0) / (s + 1e-8) ** 4
    # return ((r - mean_geom(r))**4).mean(axis=0) / (volatility(r) + 1e-8)**4


def idiosyncratic_returns(
    rr: np.ndarray,  # assets
    xx: np.ndarray,  # factors
) -> np.ndarray:
    # R = X @ B + E
    T = rr.shape[0]
    xx = np.c_[np.ones(T), xx]
    r, rt = rr[: T - 1], rr[T - 1]  # NOTE try including last day, non causal regression
    x, xt = xx[: T - 1], xx[T - 1]
    beta = np.linalg.solve(x.T @ x, x.T @ r)
    eps = rt - xt @ beta
    return eps


def rolling_regression(
    rr: np.ndarray,  # assets
    xx: np.ndarray,  # factors
    lookback: int,
) -> np.ndarray:
    T, N = rr.shape
    r_eps = np.full((T, N), fill_value=np.nan)
    for t in range(lookback - 1, T):
        r = rr[t + 1 - lookback : t + 1]
        x = xx[t + 1 - lookback : t + 1]
        r_eps[t] = idiosyncratic_returns(r, x)
    return r_eps


def rolling_feature(
    rr: np.ndarray,  # assets
    lookback: int,
    callback: Callable[..., np.ndarray],
) -> np.ndarray:
    T, N = rr.shape
    feature = np.full((T, N), fill_value=np.nan)
    for t in range(lookback - 1, T):
        r = rr[t + 1 - lookback : t + 1]
        feature[t] = callback(r)
    return feature


class FeatureBuilder:
    def __init__(
        self,
        ret_d: pd.DataFrame,
        tickers: list[str],
        factors: list[str],
        lookback: int,
        first_date: str,
        final_date: str,
    ):
        rd_ = ret_d.loc[first_date:final_date]
        assert np.isin(tickers, rd_.columns).all()
        assert np.isin(factors, rd_.columns).all()

        self.d0 = first_date
        self.d1 = final_date
        self.T = rd_.shape[0]
        self.U = len(tickers)
        self.K = len(factors)
        self.lookback_regression = lookback
        self.rr = rd_[tickers].to_numpy()
        self.xx = rd_[factors].to_numpy()
        self.ee = rolling_regression(self.rr, self.xx, lookback)
        self.timeline = rd_.index
        self.tickers = tickers
        self.factors = factors

        self.F: int = None
        self.x: np.ndarray = None
        self.__x: list[np.ndarray] = []
        self.features: list[str] = []
        self.lookbacks: list[int] = []
        self.callbacks: list[Callable[..., np.ndarray]] = []
        self.regressed: list[bool] = []

    def add_feature_assets(
        self,
        name: str,
        regress: bool,
        lookback: int,
        callback: Callable[..., np.ndarray],
    ) -> None:
        assert lookback > 0, lookback
        assert name not in self.features
        assert (lookback, callback) not in zip(self.lookbacks, self.callbacks)

        rr_ = self.rr if regress == False else self.ee
        x = rolling_feature(rr_, lookback, callback)

        self.__x.append(x)
        self.features.append(name)
        self.lookbacks.append(lookback)
        self.callbacks.append(callback)
        self.regressed.append(regress)

    def add_feature_shared(
            self,
            name: str,
            aa: np.ndarray,
    ) -> None:
        assert aa.shape == (self.T,), aa.shape
        assert name not in self.features
        x = aa[:,None].repeat(self.U, axis=1)
        assert x.shape == (self.T, self.U), x.shape

        self.__x.append(x)
        self.features.append(name)
        self.lookbacks.append(None)
        self.callbacks.append(None)
        self.regressed.append(None)

    def consolidate(self) -> None:
        x = np.stack(self.__x, axis=2)
        T, U, F = self.T, self.U, len(self.__x)
        assert x.shape == (T, U, F)
        assert len(self.features) == F
        assert len(self.lookbacks) == F
        assert len(self.callbacks) == F
        assert len(self.regressed) == F
        self.F = F
        self.x = x


class FeatureView:
    def __init__(
        self,
        fb: FeatureBuilder,
        target: str,
        subset: list[str] = None,
    ):
        assert fb.F is not None
        assert fb.x is not None
        assert target in fb.features
        subset = subset if subset is not None else fb.features
        assert np.isin(subset, fb.features).all()
        T, N, _ = fb.x.shape
        F = len(subset)
        i_F = np.array([fb.features.index(f) for f in subset], dtype=int)
        i_tgt = fb.features.index(target)
        self.T = T
        self.N = N
        self.F = F
        self.timeline = fb.timeline.copy()
        self.tickers = fb.tickers.copy()
        self.target = target
        self.horizon = fb.lookbacks[i_tgt]
        self.features = subset
        self.y = fb.x[:, :, i_tgt]
        self.__x = fb.x[:, :, i_F]
        self.x = self.__x.copy()
        self.__mask = np.full((self.N, self.F), fill_value=False, dtype=bool)
        self.mask = self.__mask.copy()

    def apply_masking(self) -> None:
        mask = self.mask[None, :, :].repeat(self.T, axis=0)  # [T, N, F]

        x = self.__x.copy()
        x[mask] = np.nan
        self.x = x

        self.mask = self.__mask.copy()

    # for given tickers:
    # exclude=True:  excludes choosen features, includes rest
    # exclude=False: includes choosen features, excludes rest
    def add_mask(self, tickers: list[str], features: list[str], exclude=True) -> None:
        assert np.isin(tickers, self.tickers).all()
        assert np.isin(features, self.features).all()

        mask_tick = np.isin(self.tickers, tickers)
        mask_name = np.isin(self.features, features)

        mask = mask_tick[:, None] & mask_name[None, :]  # [N, F]
        if exclude == False:  # include, only selected tickers
            mask[mask_tick, :] = ~mask[mask_tick, :]

        self.mask |= mask

    def get_x(self, t: int, lookback: int) -> np.ndarray:
        assert lookback >= 1
        assert lookback <= t
        assert t <= self.T - 1
        t0 = t + 1 - lookback
        t1 = t + 1
        return self.x[t0:t1, :, :]  # [L, N, F]

    def get_y(self, t: int, lookback: int) -> np.ndarray:
        assert lookback >= 1
        assert lookback <= t
        assert t <= self.T - 1
        t0 = t + 1 - lookback
        t1 = t + 1
        return self.y[t0:t1, :]  # [L, N]
