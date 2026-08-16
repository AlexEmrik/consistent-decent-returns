#%%
import pandas as pd

closes = pd.read_parquet("./data/raw/closes.parquet")

full_universe = closes.copy()
full_universe[:] = 1.0
full_universe.to_parquet("./data/processed/asset_universes/full_universe.parquet")

print("Full universe created.")
