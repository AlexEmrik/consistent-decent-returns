import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decent_returns.config import universe
from decent_returns.config import _2W, _1M, _3M, _6M, _1Y, _2Y, _3Y, _5Y, D0, D1
from decent_returns.config import FIRST_DATE, FINAL_DATE

from scipy.stats import rankdata

from decent_returns.alpha.modules.features import (
    momentum,
    momentum_vs,
    drawdown, 
    volatility, 
    skewness,
    kurtosis,
)

from decent_returns.alpha.modules.features import FeatureBuilder, FeatureView
from decent_returns.alpha.modules.predictor import RidgeRanker, RidgeRegressor
from decent_returns.alpha.modules.simulator import AlphaSimulator
from decent_returns.alpha.modules.utils import ic_score, signed_square, standard_scale, rank_cs

class Transformations:
    NONE = "none"
    SSQR = "ssqr"
    SIGN = "sign"
    # RANK = "rank"

# Hyperparameters
lookback       =  252
halflife       =  252
gamma          =    100000
transformation = Transformations.NONE
fctr = [] # E.g. "SPY", "AGG", "GLD"
tickers = universe

def main():
    closes = pd.read_parquet("./data/raw/closes.parquet")
    rd = closes.loc[FIRST_DATE:FINAL_DATE].pct_change(fill_method=None).dropna(how="all")
    rf = pd.read_parquet("./data/raw/ffr.parquet").reindex(rd.index)
    blueprint = pd.read_parquet("./data/processed/alphas/equal.parquet")

    rx = rd - rf.values
    fb = FeatureBuilder(ret_d=rx, tickers=tickers, factors=fctr, lookback=_1Y, first_date=FIRST_DATE, final_date=FINAL_DATE)
    REGRESS = False

    fb.add_feature_shared(name="const",  aa=np.ones(rx.shape[0]))
    fb.add_feature_assets(name="mom_1m", regress=REGRESS, lookback=_1M, lag=0  , callback=momentum)
    fb.add_feature_assets(name="mom_1y", regress=REGRESS, lookback=_1Y, lag=_1M, callback=momentum)
    fb.add_feature_assets(name="mom_3y", regress=REGRESS, lookback=_3Y, lag=_1Y, callback=momentum)
    fb.consolidate()
    fv = FeatureView(fb=fb, target="mom_1m", subset=None)


    sim = AlphaSimulator(fv)
    rr = RidgeRegressor(lookback=lookback, halflife=halflife, gamma=gamma,)
    sim.run(rr, verbose=False, permute=False)

    data = sim.prd
    match transformation:
        case Transformations.NONE:
            pass
        case Transformations.SSQR:
            data = np.sign(sim.prd) * (sim.prd ** 2)
        case Transformations.SIGN:
            data = np.sign(sim.prd)
        # case Transformations.RANK:  # NOTE Excluded due to not being slicable for subset of universe
        #     data = rankdata(sim.prd, axis=1, nan_policy="omit")
        case _:
            raise ValueError("Invalid transformation argument")

    alpha = pd.DataFrame(0.0, index=sim.timeline, columns=blueprint.columns)
    alpha.loc[sim.timeline, sim.fv.tickers] = data
    alpha = alpha.fillna(0.0)

    df_r2 = pd.DataFrame(sim.r2[:, None], index=tickers)
    print(df_r2)
    quit()
    
    print(f"halflife        :  {halflife}")
    print(f"lookback        :  {lookback}")
    print(f"gamma           :  {gamma}")
    print(f"transformation  :  {transformation}")
    print(f"r2              :  {sim.r2}")

    alpha.to_parquet(f"./data/processed/alphas/ridge_{transformation}.parquet")

if __name__=="__main__":
    main()