import pandas as pd
from pathlib import Path

processed_alphas_dir = Path("data/processed/alphas")
processed_alphas_dir.mkdir(parents=True, exist_ok=True)

closes = pd.read_parquet("./data/raw/closes.parquet")

alphas_equal = closes.copy()
alphas_equal[:] = 1.0
alphas_equal.to_parquet(processed_alphas_dir / "equal.parquet")

alphas_6040 = closes.copy()
alphas_6040[:] = 0.0
alphas_6040[["SPY", "AGG"]] = [0.6, 0.4]
alphas_equal.to_parquet(processed_alphas_dir / "6040.parquet")

print("Alphas created.")
