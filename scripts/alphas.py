import pandas as pd

closes = pd.read_parquet("./data/raw/closes.parquet")

alphas_equal = closes.copy()
alphas_equal[:] = 1.0
alphas_equal.to_parquet("./data/processed/alphas/equal.parquet")

alphas_6040 = closes.copy()
alphas_6040[:] = 0.0
alphas_6040[["SPY", "AGG"]] = [0.6, 0.4]
alphas_equal.to_parquet("./data/processed/alphas/6040.parquet")

print("Alphas created.")
