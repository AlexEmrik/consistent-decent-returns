from abc import ABC, abstractmethod

import numpy as np

from decent_returns.alpha.modules.utils import signed_square, standard_scale, rank_cs

class AlphaPredictor(ABC):
    def __init__(self, lookback: int):
        super().__init__()
        self.lookback = lookback
        self.theta = None
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def fit(self, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, xt: np.ndarray) -> np.ndarray:
        pass


class RidgeRanker(AlphaPredictor):
    def __init__(
        self,
        lookback: int,
        halflife: int,
        gamma: float,
        sign_sq: bool = False,
        sign: bool = False,
        rank: bool = False,
    ):
        super().__init__(lookback)
        self.gamma = gamma
        self.lookback = lookback
        self.halflife = halflife
        self.sign_sq = sign_sq
        self.sign = sign
        self.rank = rank
        assert not sign == rank == True

        w_ewma = 0.5 ** (np.arange(lookback) / halflife)
        w_ewma = (w_ewma[::-1] / w_ewma.sum()) ** 0.5
        self.w_ewma = w_ewma

    def fit(self, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:
        T, N, F = xt.shape
        TN = T * N
        assert yt.shape == (T, N), yt.shape
        xt = standard_scale(xt, axis=1)
        xw = xt * self.w_ewma[:, None, None]
        yw = yt * self.w_ewma[:, None]

        if self.sign_sq:
            yw = signed_square(yw)
        if self.sign:
            yw = np.sign(yw)
        if self.rank:
            yw = rank_cs(yw)

        I_f = np.eye(F)
        xw = xw.reshape(TN, F)
        yw = yw.reshape(TN)

        theta = np.linalg.solve(xw.T @ xw + self.gamma * I_f, xw.T @ yw)  # [F,]
        self.theta = theta

    def predict(self, xt: np.ndarray) -> np.ndarray:
        _, _, F = xt.shape
        assert self.theta is not None
        assert self.theta.shape == (F,), self.theta.shape
        xt = standard_scale(xt, axis=1)
        pt = xt @ self.theta
        return pt


class RidgeRegressor(AlphaPredictor):
    def __init__(
            self, 
            lookback: int,
            halflife: int,
            gamma: float,
            ):
        super().__init__(lookback)
        self.gamma = gamma
        self.lookback = lookback
        self.halflife = halflife
        
        self.mu = None
        self.sd = None

        w_ewma = 0.5 ** (np.arange(lookback) / halflife)
        w_ewma = (w_ewma[::-1] / w_ewma.sum()) ** 0.5
        self.w_ewma = w_ewma

    def fit(self, xt: np.ndarray, yt: np.ndarray) -> np.ndarray:    
        T, N, F = xt.shape
        assert yt.shape == (T, N), yt.shape

        I_f = np.eye(F)
        I_f[0,0] = 0.0  # don't reg const
        self.mu = np.nanmean(xt[:, :, 1:], axis=0, keepdims=True)
        self.sd = np.nanstd( xt[:, :, 1:], axis=0, keepdims=True) + 1e-8
        xt = xt.copy()
        xt[:, :, 1:] = (xt[:, :, 1:] - self.mu) / self.sd
        
        xw: np.ndarray = xt * self.w_ewma[:, None, None]
        yw: np.ndarray = yt * self.w_ewma[:, None]

        xw = xw.transpose(1, 0, 2)     # [N, T, F]
        yw = yw.transpose(1, 0)        # [N, T]
        a  = xw.transpose(0, 2, 1) @ xw + self.gamma * I_f[None, :, :]  # [N, F, F]
        b  = (xw * yw[:, :, None]).sum(axis=1)[:, :, None]  # [N, F, 1]

        theta = np.linalg.solve(a, b).squeeze(2)  # [N, F]
        self.theta = theta

    def predict(self, xt: np.ndarray) -> np.ndarray:
        _, N, F = xt.shape
        xt.copy()
        xt[:, :, 1:] = (xt[:, :, 1:] - self.mu) / self.sd

        assert self.theta is not None
        assert self.theta.shape == (N, F), self.theta.shape
        pt = (xt * self.theta[None, :, :]).sum(axis=2)
        return pt

