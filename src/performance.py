
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CAPITAL = 10_000
COST = 0.001  # 0.1% per executed transaction
START = "2026-01-01"
END = "2026-12-31"

Path("results").mkdir(exist_ok=True)

# 1. Load all historical data
df = pd.read_parquet("spy_prices.parquet")
df = df.sort_index().copy()

# 2. Calculate signals using historical data
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

df["target"] = np.where(
    df["SMA20"].notna() & df["SMA50"].notna(),
    (df["SMA20"] > df["SMA50"]).astype(int),
    0
)

# Signal after close, execution next morning
df["position"] = df["target"].shift(1).fillna(0).astype(int)

# Open-to-open returns
df["market_return"] = (
    df["Open"].shift(-1) / df["Open"] - 1
)

# Filter only after indicators and returns are computed
df = df.loc[START:END].dropna(
    subset=["market_return"]
).copy()

if df.empty:
    raise ValueError("No data for the selected period")

# 3. Transaction costs and changes in exposure
df["traded"] = df["position"].diff().abs()

# Start a fresh portfolio on first day of the test.
# Opening an initial long position incurs a fee.
df.iloc[
    0, df.columns.get_loc("traded")
] = df["position"].iloc[0]

df["strategy_return"] = (
    df["position"] * df["market_return"]
    - df["traded"] * COST
)

# Benchmark: buy SPY on the first test day
df["benchmark_return"] = df["market_return"].copy()
df.iloc[
    0, df.columns.get_loc("benchmark_return")
] -= COST

# 4. Portfolio equity
df["strategy_equity"] = CAPITAL * (
    1 + df["strategy_return"]
).cumprod()

df["benchmark_equity"] = CAPITAL * (
    1 + df["benchmark_return"]
).cumprod()


def longest_underwater(equity):
    peak = CAPITAL
    current = 0
    longest = 0

    for value in equity:
        peak = max(peak, value)

        if value < peak:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def calculate_metrics(returns, equity, positions=None, trades=None):
    n = len(returns)
    final_value = equity.iloc[-1]

    total_return = final_value / CAPITAL - 1
    annualized_return = (final_value / CAPITAL) ** (252 / n) - 1

    volatility = returns.std()
    sharpe = (
        np.sqrt(252) * returns.mean() / volatility
        if pd.notna(volatility) and volatility > 0
        else np.nan
    )

    equity_with_start = pd.concat([
        pd.Series([CAPITAL]),
        equity.reset_index(drop=True)
    ], ignore_index=True)

    drawdown = (
        equity_with_start / equity_with_start.cummax() - 1
    )

    return {
        "Final Capital": round(final_value, 2),
        "Total Return (%)": round(total_return * 100, 2),
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Sharpe (rf=0)": round(sharpe, 2),
        "Max Drawdown (%)": round(drawdown.min() * 100, 2),
        "Longest Underwater (sessions)": longest_underwater(
            equity_with_start
        ),
        "Market Exposure (%)": (
            round(positions.mean() * 100, 2)
            if positions is not None else 100.0
        ),
        "Transactions": (
            int(trades.sum())
            if trades is not None else 1
        )
    }


strategy = calculate_metrics(
    df["strategy_return"],
    df["strategy_equity"],
    df["position"],
    df["traded"]
)

benchmark = calculate_metrics(
    df["benchmark_return"],
    df["benchmark_equity"]
)

# 5. Performance table
report = pd.DataFrame({
    "SMA20/50": strategy,
    "Buy & Hold": benchmark
})

print("\nSPY PERFORMANCE REPORT — 2026")
print(report.to_string())
print("\nFirst test session:", df.index.min())
print("Last test session:", df.index.max())

report.to_csv("results/performance_2026.csv")

# 6. Equity chart
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df.index, df["strategy_equity"],
    label="SMA20/50"
)
ax.plot(
    df.index, df["benchmark_equity"],
    label="Buy & Hold SPY"
)

ax.set_title("SPY Strategy vs Buy & Hold — 2026")
ax.set_ylabel("Portfolio Value (USD)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("results/equity_2026.png", dpi=150)
plt.close(fig)

# 7. Position chart
fig, ax = plt.subplots(figsize=(12, 3))

ax.step(
    df.index, df["position"],
    where="post"
)

ax.set_title("SMA20/50 Market Exposure — 2026")
ax.set_ylabel("Position")
ax.set_yticks([0, 1])
ax.set_yticklabels(["CASH", "SPY"])
ax.set_ylim(-0.2, 1.2)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("results/positions_2026.png", dpi=150)
plt.close(fig)

print("\nSaved:")
print("results/performance_2026.csv")
print("results/equity_2026.png")
print("results/positions_2026.png")

