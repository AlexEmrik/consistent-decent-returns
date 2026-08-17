import pandas as pd
from pathlib import Path
from decent_returns.config import _1M

FIRST_DATE = "1995-01-03"
FINAL_DATE = "2026-01-01"

processed_alphas_dir = Path("data/processed/alphas")
processed_alphas_dir.mkdir(parents=True, exist_ok=True)

closes = pd.read_parquet("./data/raw/closes.parquet")
rd = closes.loc[FIRST_DATE:FINAL_DATE].pct_change(fill_method=None).dropna(how="all")
rf = pd.read_parquet("./data/raw/ffr.parquet").reindex(rd.index)
universe = rd.columns.to_list()
rx = rd - rf.values

# Create equal alphas
alphas_equal = closes.copy()
alphas_equal[:] = 1.0
alphas_equal.to_parquet(processed_alphas_dir / "equal.parquet")

# Create 60/40 alphas
alphas_6040 = closes.copy()
alphas_6040[:] = 0.0
alphas_6040[["SPY", "AGG"]] = [0.6, 0.4]
alphas_equal.to_parquet(processed_alphas_dir / "6040.parquet")

# Create momemtum alpha
mom_1m = rx.ewm(min_periods=_1M, halflife=_1M).mean().fillna(0)
mom_1m.to_parquet(processed_alphas_dir / "mom_1m.parquet")

# Create moving average signal alpha
rho = 1.0
ma_1d  = closes.rolling(1).mean().fillna(0)
ma_50d = closes.rolling(50).mean().fillna(0)
mov_50d = (ma_1d > rho * ma_50d).astype(float) * 2 - 1
mov_50d.to_parquet(processed_alphas_dir / "mov_50d.parquet")

print("Alphas created.")






