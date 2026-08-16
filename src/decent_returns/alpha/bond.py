import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decent_returns.config import _universe, mrkt, sect, bond, cmdy, gold
from decent_returns.config import _2W, _1M, _3M, _6M, _1Y, _2Y, _3Y, _5Y, D0, D1
from decent_returns.config import FIRST_DATE

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
from decent_returns.alpha.modules.utils import ic_score, rank, signed_square, standard_scale

FIRST_DATE = "1995-01-03"
FINAL_DATE = "2026-01-01"
SPREAD = 5e-4
LEVERAGE = 0.0
VC_LIMIT = 0.08
REBAL_FREQ = "M"

fctr = [] # , "AGG", "GLD"
tickers = bond

closes = pd.read_parquet("../../../data/raw/closes.parquet")
rd = closes.loc[FIRST_DATE:FINAL_DATE].pct_change(fill_method=None).dropna(how="all")
rf = pd.read_parquet("../../../data/raw/ffr.parquet").reindex(rd.index)
md = pd.read_parquet("../../../data/processed/features/macro.parquet").reindex(rd.index)
universe = rd.columns.to_list()

rx = rd - rf.values
T, N = rx.shape

fb = FeatureBuilder(ret_d=rx, tickers=tickers, factors=fctr, lookback=_1Y, first_date=FIRST_DATE, final_date=FINAL_DATE)
REGRESS = False

# fb.add_feature_shared(name="const",     aa=np.ones(T))
# fb.add_feature_shared(name="ffr_d1m",   aa=md["ffr_d1m"].to_numpy())
# fb.add_feature_shared(name="ffr_d1y",   aa=md["ffr_d1y"].to_numpy())
# fb.add_feature_shared(name="slp_lvl",   aa=md["slp_lvl"].to_numpy())
# fb.add_feature_shared(name="slp_d1m",   aa=md["slp_d1m"].to_numpy())
# fb.add_feature_shared(name="cs_lvl",    aa=md["cs_lvl"].to_numpy())
# fb.add_feature_shared(name="cs_d1m",    aa=md["cs_d1m"].to_numpy())
# fb.add_feature_shared(name="vix_lvl",   aa=md["vix_lvl"].to_numpy())
# fb.add_feature_shared(name="vix_d1m",   aa=md["vix_d1m"].to_numpy())
# fb.add_feature_shared(name="cpi_yoy",   aa=md["cpi_yoy"].to_numpy())
# fb.add_feature_shared(name="dgs10_d1m", aa=md["dgs10_d1m"].to_numpy())
# fb.add_feature_shared(name="dxy_d1m",   aa=md["dxy_d1m"].to_numpy())
# fb.add_feature_shared(name="spy_12m",   aa=md["spy_12m"].to_numpy())

# fb.add_feature_shared(name="carry_dgs10",  aa=md["dgs10_lvl"].to_numpy())
# fb.add_feature_shared(name="carry_dgs30",  aa=md["dgs30_lvl"].to_numpy())
# fb.add_feature_shared(name="carry_dfii10", aa=md["dfii10_lvl"].to_numpy())
# fb.add_feature_shared(name="carry_igy",    aa=md["igy_lvl"].to_numpy())
# fb.add_feature_shared(name="carry_hyy",    aa=md["hyy_lvl"].to_numpy())
# fb.add_feature_shared(name="carry_emy",    aa=md["emy_lvl"].to_numpy())

fb.add_feature_assets(name="mom_2w", regress=REGRESS, lookback=_2W, callback=momentum)
fb.add_feature_assets(name="mom_1m", regress=REGRESS, lookback=_1M, callback=momentum)
fb.add_feature_assets(name="mom_3m", regress=REGRESS, lookback=_3M, callback=momentum)
fb.add_feature_assets(name="mom_6m", regress=REGRESS, lookback=_6M, callback=momentum)
fb.add_feature_assets(name="mom_1y", regress=REGRESS, lookback=_1Y, callback=momentum)
fb.add_feature_assets(name="vol_2w", regress=REGRESS, lookback=_2W, callback=volatility)
fb.add_feature_assets(name="vol_1m", regress=REGRESS, lookback=_1M, callback=volatility)
fb.add_feature_assets(name="vol_3m", regress=REGRESS, lookback=_3M, callback=volatility)
fb.add_feature_assets(name="vol_6m", regress=REGRESS, lookback=_6M, callback=volatility)
fb.add_feature_assets(name="vol_1y", regress=REGRESS, lookback=_1Y, callback=volatility)
fb.add_feature_assets(name="mdd_2w", regress=REGRESS, lookback=_2W, callback=drawdown)
fb.add_feature_assets(name="mdd_1m", regress=REGRESS, lookback=_1M, callback=drawdown)
fb.add_feature_assets(name="mdd_3m", regress=REGRESS, lookback=_3M, callback=drawdown)
fb.add_feature_assets(name="mdd_6m", regress=REGRESS, lookback=_6M, callback=drawdown)
fb.add_feature_assets(name="mdd_1y", regress=REGRESS, lookback=_1Y, callback=drawdown)

fb.consolidate()

fv = FeatureView(fb=fb, target="mom_1m", subset=None)

# fv.add_mask(tickers=["TLT"],         features=["carry_dgs30"],  exclude=True)
# fv.add_mask(tickers=["IEF"],         features=["carry_dgs10"],  exclude=True)
# fv.add_mask(tickers=["TIP"],         features=["carry_dfii10"], exclude=True)
# fv.add_mask(tickers=["LQD", "LQDH"], features=["carry_igy"],    exclude=True)
# fv.add_mask(tickers=["HYG"],         features=["carry_hyy"],    exclude=True)
# fv.add_mask(tickers=["EMB"],         features=["carry_emy"],    exclude=True)

# fv.apply_masking()

fv.target, fv.horizon

lookbacks = [63, 126, 252]
halflifes = [63, 126, 252]
gammas = [1, 10, 100]

# BEST : bond_ts_sign_sq_g100_lb252_hl63 (const) - 0.506
# BEST : bond_ts_sign_sq_g1_lb252_hl63           - 0.349 negative corr with spy, no bonds held at end of period (Good?)


transforms = [
    ("raw",     False, False, False),
    ("sign_sq", True,  False, False),
    ("sign",    False, True,  False),
    ("rank",    False, False, True),
]

blueprint = pd.read_parquet("../../../data/processed/alphas/equal_alphas.parquet")

for g in gammas:
    for lb in lookbacks:
        for hl in halflifes:
            # run the sim ONCE per (g, lb, hl)
            sim = AlphaSimulator(fv)
            rr = RidgeRegressor(
                lookback=lb, halflife=hl, gamma=g,
                pooled=False, const=True,
            )
            sim.run(rr, verbose=False, permute=False)

            # write all 4 transforms from the same predictions
            for (tf_label, sign_sq, sign, rank) in transforms:
                data = sim.prd
                if sign_sq:
                    data = np.sign(sim.prd) * (sim.prd ** 2)
                elif sign:
                    data = np.sign(sim.prd)
                elif rank:
                    data = rankdata(sim.prd, axis=1, nan_policy="omit")

                alpha = pd.DataFrame(0.0, index=sim.timeline, columns=blueprint.columns)
                alpha.loc[sim.timeline, sim.fv.tickers] = data
                alpha = alpha.fillna(0.0)
                alpha.to_parquet(
                    f"../../../data/processed/alphas/bond_ts_{tf_label}_g{g}_lb{lb}_hl{hl}.parquet"
                )
                print(f"bond_ts_{tf_label}_g{g}_lb{lb}_hl{hl}")


# # lookbacks = [63, 126, 252]
# # halflifes = [63, 126, 252]
# # gammas = [1, 10, 100]

# # cs_raw_g100_lb63_hl63
# lookbacks = [57, 69]
# halflifes = [57, 69]
# gammas = [90, 110]

# transforms = [
#     ("raw",     False, False, False),
#     # ("sign_sq", True,  False, False),
#     # ("sign",    False, True,  False),
#     # ("rank",    False, False, True),
# ]

# blueprint = pd.read_parquet("../../../data/processed/alphas/equal_alphas.parquet")

# for g in gammas:
#     for lb in lookbacks:
#         for hl in halflifes:
#             # run the sim ONCE per (g, lb, hl)
#             sim = AlphaSimulator(fv)
#             rr = RidgeRanker(
#                 lookback=lb, halflife=hl, gamma=g,
#             )
#             sim.run(rr, verbose=False, permute=False)

#             # write all 4 transforms from the same predictions
#             for (tf_label, sign_sq, sign, rank) in transforms:
#                 data = sim.prd
#                 if sign_sq:
#                     data = np.sign(sim.prd) * (sim.prd ** 2)
#                 elif sign:
#                     data = np.sign(sim.prd)
#                 elif rank:
#                     data = rankdata(sim.prd, axis=1, nan_policy="omit")

#                 alpha = pd.DataFrame(0.0, index=sim.timeline, columns=blueprint.columns)
#                 alpha.loc[sim.timeline, sim.fv.tickers] = data
#                 alpha = alpha.fillna(0.0)
#                 alpha.to_parquet(
#                     f"../../../data/processed/alphas/bond_cs_{tf_label}_g{g}_lb{lb}_hl{hl}.parquet"
#                 )
#                 print(f"bond_cs_{tf_label}_g{g}_lb{lb}_hl{hl}")