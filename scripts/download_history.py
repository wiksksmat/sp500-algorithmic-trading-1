
import yfinance as yf
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# End date is exclusive — avoid today's unfinished session
end_date = datetime.now(
    ZoneInfo("America/New_York")
).date().isoformat()

print("Downloading SPY historical prices...")

df = yf.Ticker("SPY").history(
    start="2005-01-01",
    end=end_date,
    interval="1d",
    auto_adjust=True,
    actions=False,
    raise_errors=True
)

if df.empty:
    raise RuntimeError("No SPY data received")

# Keep original OHLCV data
df = df[["Open", "High", "Low", "Close", "Volume"]]

# Convert dates to timezone-independent session dates
df.index = pd.to_datetime(df.index.date)
df.index.name = "Date"

df = df.sort_index()

if df.index.has_duplicates:
    raise RuntimeError("Duplicate session dates detected")

if df.isna().any().any():
    raise RuntimeError("Missing OHLCV values detected")

# Save historical data locally
df.to_parquet("spy_prices.parquet")

print("First session:", df.index.min().date())
print("Last session:", df.index.max().date())
print("Total sessions:", len(df))
print("Saved: spy_prices.parquet")
print(df.tail())


