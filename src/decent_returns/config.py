
FIRST_DATE     = "1954-01-03"  # first date
FINAL_DATE     = "2026-01-01"  # final date

START_DATE = "1995-01-01"
END_DATE   = "2026-01-01"

AUTO_ADJUST = True

D0 = "2005-01-03"
D1 = "2024-12-31"

# Equity (Eq.)
mrkt = ["SPY", "QQQ", "VTV", "VUG", "MDY", "IWM", "SCHD", "USMV", "QUAL"]
sect = ["XLK","XLV","XLF","XLY","XLI","XLP","XLE","XLU","XLB","IBB","IYR"]
ctry = ["EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWI", "EWD", "EWL", "EWC"]
eqty = [*mrkt, *sect, *ctry]

# Fixed Income (FI)
govm = ["AGG","TLT","IEF","TIP","MUB"]
corp = ["LQD","LQDH","HYG"]
glob = ["BNDX","EMB","IAGG","VWOB"]
fixd = [*govm, *corp, *glob]

# Alternatives (Alt.)
phys = ["GLD", "SLV", "PPLT"]
futr = ["CPER", "USO", "UGA", "CORN", "WEAT", "SOYB", "CANE"]
cryp = ["BTC-USD", "ETH-USD"]
alts = [*phys, *futr, *cryp]

universe = sorted([*mrkt, *sect, *ctry, *govm, *corp, *glob, *phys, *futr, *cryp])

# Time periods
_1W = 5
_2W = 10
_1M = 21
_3M = 63
_6M = 126
_1Y = 252 
_2Y = 504
_3Y = 756
_5Y = 1260
