# Simple SIGMA
import pandas as pd
import numpy as np 
from decent_returns.config import _1M, _1Y

FIRST_DATE = "1995-01-03"
FINAL_DATE = "2026-01-01"

closes = pd.read_parquet("./data/raw/closes.parquet")
rd = closes.loc[FIRST_DATE:FINAL_DATE].pct_change(fill_method=None).dropna(how="all")
rf = pd.read_parquet("./data/raw/ffr.parquet").reindex(rd.index)
universe = rd.columns.to_list()
rx = rd - rf.values

mom_1m = rx.ewm(min_periods=_1M, halflife=_1M).mean().fillna(0)
ma_1d  = closes.rolling(1).mean().fillna(0)
ma_50d = closes.rolling(50).mean().fillna(0)
mov_50d = (ma_1d > 1.0 * ma_50d).astype(float) * 2 - 1
# vol_1m = rx.ewm(min_periods=_1M, halflife=_1M).std().fillna(0)
# mvs_1m = mom_1m / vol_1m

mom_1m.to_parquet("./data/processed/alphas/mom_1m.parquet")
mov_50d.to_parquet("./data/processed/alphas/mov_50d.parquet")
