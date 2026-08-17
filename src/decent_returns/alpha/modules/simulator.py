import time

import numpy as np
import pandas as pd

from decent_returns.alpha.modules.features import FeatureView
from decent_returns.alpha.modules.predictor import AlphaPredictor
from decent_returns.alpha.modules.utils import ic_score, nrmse_score, r2_score


class AlphaSimulator:
    def __init__(self, fv: FeatureView):
        self.fv = fv
        # self.thetas: np.ndarray | None = None
        self.prd: np.ndarray | None = None
        self.ref: np.ndarray | None = None
        self.predictor: AlphaPredictor | None = None
        self.time = -1

    def run(self, predictor: AlphaPredictor, verbose=False, permute=False):
        t0 = time.time()
        fv = self.fv
        h = fv.horizon
        L = predictor.lookback
        T, N, F = fv.T, fv.N, fv.F
        # thetas = np.full((T, F), fill_value=np.nan, dtype=float)
        alphas = np.full((T, N), fill_value=np.nan, dtype=float)
        gtruth = np.full((T, N), fill_value=np.nan, dtype=float)
        for t in range(h + L, T - h):
            x_trn = fv.get_x(t - h, L)  # [ t+1-h-l : t+1-h ]
            y_trn = fv.get_y(t, L)      # [ t+1  -l : t+1   ]
            x_tst = fv.get_x(t, 1)      # [ t       : t+1   ]
            y_tst = fv.get_y(t + h, 1)  # [ t  +h   : t+1+h ]

            if permute == True:
                index = np.argsort(np.random.rand(*y_trn.shape), axis=1)
                y_trn = np.take_along_axis(y_trn, index, axis=1)

            # skip if all assets are missing
            if np.isnan(x_trn).all(axis=2).all(): continue
            if np.isnan(x_tst).all(axis=2).all(): continue
            if np.isnan(y_trn).all(): continue
            if np.isnan(y_tst).all(): continue

            x_trn = np.nan_to_num(x_trn, nan=0.0)
            x_tst = np.nan_to_num(x_tst, nan=0.0)
            y_trn = np.nan_to_num(y_trn, nan=0.0)
            y_tst = np.nan_to_num(y_tst, nan=0.0)

            predictor.fit(x_trn, y_trn)
            # thetas[t] = predictor.theta
            alphas[t] = predictor.predict(x_tst)[0]
            gtruth[t] = y_tst[0]

        # self.thetas = thetas
        self.prd = alphas
        self.ref = gtruth
        self.time = time.time() - t0
        if verbose == True:
            print_simulator_results(self)

    def get_alpha(self, universe: list[str]) -> pd.DataFrame:
        tickers = self.fv.tickers
        timeline = self.fv.timeline
        T = timeline.shape[0]
        U = len(universe)
        i_N = np.array([universe.index(t) for t in tickers], dtype=int)
        alpha = np.full((T, U), fill_value=np.nan, dtype=float)
        alpha[:, i_N] = self.prd
        alpha = pd.DataFrame(data=alpha, columns=universe, index=timeline)
        alpha = alpha.fillna(1.0)
        return alpha

    @property
    def ic_spearman(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return ic_score(self.prd, self.ref, method="spearman")

    @property
    def ic_pearson(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return ic_score(self.prd, self.ref, method="pearson")

    @property
    def nrmse(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return nrmse_score(self.prd, self.ref, axis=0)

    @property
    def r2(self) -> np.ndarray:
        assert self.prd is not None
        assert self.ref is not None
        return r2_score(self.prd, self.ref, axis=0)

    @property
    def timeline(self) -> pd.DatetimeIndex:
        return self.fv.timeline


def print_simulator_results(sim: AlphaSimulator) -> None:
    print(f"Backtest Runtime: {round(sim.time * 1000)} ms")
    print(f"ic sprm:    {np.nanmean(sim.ic_spearman).round(4)}")
    print(f"ic prsn:    {np.nanmean(sim.ic_pearson ).round(4)}")
    print(f"nrmse  :    {np.nanmean(sim.nrmse      ).round(4)}")
    print(f"r2     :    {np.nanmean(sim.r2         ).round(4)}")
