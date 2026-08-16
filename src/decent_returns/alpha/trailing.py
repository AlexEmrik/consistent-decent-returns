# Simple SIGMA
import pandas as pd
import numpy as np 
from decent_returns.config import _1M, _1Y, cmdy

FIRST_DATE = "1995-01-03"
FINAL_DATE = "2026-01-01"

closes = pd.read_parquet("../../../data/raw/closes.parquet")
rd = closes.loc[FIRST_DATE:FINAL_DATE].pct_change(fill_method=None).dropna(how="all")
rf = pd.read_parquet("../../../data/raw/ffr.parquet").reindex(rd.index)
universe = rd.columns.to_list()
rx = rd - rf.values

mom_1m = rx.ewm(min_periods=_1M, halflife=_1M).mean().fillna(0)
vol_1m = rx.ewm(min_periods=_1M, halflife=_1M).std().fillna(0)
mvs_1m = mom_1m / vol_1m
ma_1  = closes.rolling(1).mean().fillna(0)
ma_50 = closes.rolling(50).mean().fillna(0)

ma_1gt50 = (ma_1 > ma_50 * 1.3).astype(float) * 2 - 1
mom_1m.to_parquet("../../../data/processed/alphas/mom_1m.parquet")
mvs_1m.to_parquet("../../../data/processed/alphas/mvs_1m.parquet")
ma_1gt50.to_parquet("../../../data/processed/alphas/ma50.parquet")