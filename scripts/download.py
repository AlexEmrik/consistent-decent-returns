"""Download raw data and compute shared processed artifacts."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from tqdm import tqdm
import os

import decent_returns.backtest.dataloader as dl
from decent_returns.risk.stat import get_F_D_half
from decent_returns.utils import save_to_pkl

from decent_returns.config import universe, cryp
from decent_returns.config import START_DATE, END_DATE

raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

print("=== Stage 1: Downloading raw data ===")

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Parameters ────────────────────────────────────────────────────────────────

VOL_HALFLIFE = 42           
COV_HALFLIFE = 126          
UNIVERSE_MIN_PERIODS = 126  
MIN_UNIVERSE_SIZE = 15      
N_FACTORS = 10              
FIRST_DATE_CRYPTO =  "2017-09-07"
etfs = universe

# ── Stage 1: Download raw data ─────────────────────────────────────────────────

yf_data = yf.download(etfs, start=START_DATE, end=END_DATE, auto_adjust=True)
closes = yf_data["Close"]
volumes = yf_data["Volume"]

if "BTC-USD" in universe:
    closes.rename( {"ETH-USD": "ETHA", "BTC-USD":"IBIT"})
    closes.loc[:FIRST_DATE_CRYPTO,  "BTC-USD"] = np.nan
    volumes.loc[:FIRST_DATE_CRYPTO, "BTC-USD"] = np.nan

if "ETH-USD" in universe:
    volumes.rename({"ETH-USD": "ETHA", "BTC-USD":"IBIT"})
    closes.loc[:FIRST_DATE_CRYPTO,  "ETH-USD"] = np.nan
    volumes.loc[:FIRST_DATE_CRYPTO, "ETH-USD"] = np.nan

# exclude crypto from trading days since they trade on weekends as well
trading_days = closes.drop(columns=cryp).dropna(how="all").index
closes = closes.loc[trading_days]
volumes = volumes.loc[trading_days]
closes.to_parquet(raw_data_dir / "closes.parquet")
volumes.to_parquet(raw_data_dir / "volumes.parquet")

print("Saved closes.parquet and volumes.parquet")

FRED_API_KEY = os.environ.get("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

print("Downloading Fed Funds Rate...")
ffr_raw = fred.get_series("DFF", observation_start=START_DATE, observation_end=END_DATE).to_frame("FFR")
ffr_aligned = ffr_raw.reindex(yf_data.index).ffill()
ffr_daily = (1 + ffr_aligned / 100) ** (1 / 252) - 1.0
ffr_daily.to_parquet(raw_data_dir / "ffr.parquet")
print("Saved ffr.parquet")

# ── Stage 2: Full-universe risk model ─────────────────────────────────────────

proc_data_dir = Path("data/processed")
proc_data_dir.mkdir(parents=True, exist_ok=True)

returns = dl.load_asset_returns()
assets = returns.columns

factors = list(range(N_FACTORS))

Fs: dict = {}
D_halves: dict = {}

for ts in tqdm(returns.index[UNIVERSE_MIN_PERIODS:], desc="Computing Fs and D_halves"):
    returns_so_far = returns.loc[:ts].copy()
    universe = returns_so_far.tail(UNIVERSE_MIN_PERIODS).dropna(axis=1).columns
    if len(universe) < MIN_UNIVERSE_SIZE:
        Fs[ts] = pd.DataFrame(0.0, index=assets, columns=factors)
        D_halves[ts] = pd.Series(0.0, index=assets)
        continue
    returns_so_far = returns_so_far[universe].dropna()
    F, D_half = get_F_D_half(
        returns_so_far,
        k=N_FACTORS,
        vol_halflife=VOL_HALFLIFE,
        cov_halflife=COV_HALFLIFE,
    )
    Fs[ts] = F.reindex(index=assets, columns=factors, fill_value=0.0)
    D_halves[ts] = D_half.reindex(index=assets, fill_value=0.0)

risk_model_dir = proc_data_dir / "risk_model"
risk_model_dir.mkdir(parents=True, exist_ok=True)
save_to_pkl({"Fs": Fs, "D_halves": D_halves}, risk_model_dir / f"K{N_FACTORS}.pkl")
print(f"Saved K{N_FACTORS}.pkl")


print("\nSetup complete.")
