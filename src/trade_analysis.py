
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
START = "2026-01-01"
END = "2026-12-31"
INITIAL_CAPITAL = 10_000.0
COST = 0.001  # 0.1% on each buy or sell

Path("results").mkdir(exist_ok=True)

# Load all history, including data before 2026
df = pd.read_parquet("spy_prices.parquet")
df = df.sort_index().copy()

# Calculate indicators using past closes
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()

# Position desired AFTER each session closes
df["target"] = np.where(
    df["SMA50"].notna(),
    (df["SMA20"] > df["SMA50"]).astype(int),
    0
)

# Yesterday's signal determines today's OPEN trade
df["desired_position"] = df["target"].shift(1).fillna(0)

# Only now select the testing period
df = df.loc[START:END].copy()

if df.empty:
    raise ValueError("No historical data for 2026")

# Portfolio state
cash = INITIAL_CAPITAL
shares = 0.0
active_trade = None

trades = []
orders = []
daily = []

# Buy & Hold benchmark
first_open = float(df.iloc[0]["Open"])
benchmark_shares = (
    INITIAL_CAPITAL / (first_open * (1 + COST))
)

for date, row in df.iterrows():
    price_open = float(row["Open"])
    price_close = float(row["Close"])
    desired = int(row["desired_position"])
    day = str(date.date())

    # BUY at the next session open
    if desired == 1 and shares == 0:
        invested = cash
        shares = cash / (price_open * (1 + COST))
        entry_fee = invested - shares * price_open
        cash = 0.0

        active_trade = {
            "Entry Date": day,
            "Entry Price": price_open,
            "Entry Capital": invested,
            "Entry Fee": entry_fee
        }

        orders.append({
            "Date": day,
            "Action": "BUY",
            "Price": price_open
        })

    # SELL at the next session open
    elif desired == 0 and shares > 0:
        gross_proceeds = shares * price_open
        exit_fee = gross_proceeds * COST
        cash = gross_proceeds - exit_fee

        entry_capital = active_trade["Entry Capital"]
        pnl = cash - entry_capital

        trades.append({
            "Entry Date": active_trade["Entry Date"],
            "Exit Date": day,
            "Entry Price": active_trade["Entry Price"],
            "Exit Price": price_open,
            "Net P&L": pnl,
            "Return (%)": 100 * pnl / entry_capital,
            "Holding Days": (
                pd.Timestamp(day)
                - pd.Timestamp(active_trade["Entry Date"])
            ).days,
            "Entry Fee": active_trade["Entry Fee"],
            "Exit Fee": exit_fee
        })

        orders.append({
            "Date": day,
            "Action": "SELL",
            "Price": price_open
        })

        shares = 0.0
        active_trade = None

    # Mark portfolios to market at the day's close
    equity = cash + shares * price_close
    benchmark = benchmark_shares * price_close

    daily.append({
        "Date": date,
        "Strategy Equity": equity,
        "Buy Hold Equity": benchmark,
        "Cash": cash,
        "Shares": shares,
        "Position": int(shares > 0)
    })

# Convert output to dataframes
daily = pd.DataFrame(daily).set_index("Date")
trades = pd.DataFrame(trades)
orders = pd.DataFrame(orders)

# Daily performance: first return measured from $10,000
def metrics(equity):
    values = np.r_[INITIAL_CAPITAL, equity.to_numpy()]
    returns = pd.Series(values).pct_change().dropna()

    annual_return = (
        (values[-1] / INITIAL_CAPITAL) ** (252 / len(equity)) - 1
    )

    volatility = returns.std()
    sharpe = (
        np.sqrt(252) * returns.mean() / volatility
        if volatility > 0 else np.nan
    )

    drawdown = values / np.maximum.accumulate(values) - 1

    return {
        "Final Capital": round(values[-1], 2),
        "Total Return (%)": round(
            (values[-1] / INITIAL_CAPITAL - 1) * 100, 2
        ),
        "Annualized Return (%)": round(annual_return * 100, 2),
        "Sharpe (rf=0)": round(sharpe, 2),
        "Max Drawdown (%)": round(drawdown.min() * 100, 2)
    }

report = pd.DataFrame({
    "SMA20/50": metrics(daily["Strategy Equity"]),
    "Buy & Hold": metrics(daily["Buy Hold Equity"])
})

print("\nPERFORMANCE REPORT")
print(report.to_string())

print("\nORDERS")
print(orders.to_string(index=False))

print("\nCLOSED TRADES")
if trades.empty:
    print("No completed trades")
else:
    print(trades.to_string(index=False))

    win_rate = (trades["Net P&L"] > 0).mean() * 100
    expectancy = trades["Net P&L"].mean()

    print(f"\nWin Rate: {win_rate:.2f}%")
    print(f"Expectancy: ${expectancy:.2f} per closed trade")

print(f"\nOpen Position: {shares:.6f} SPY")
print(f"Cash: ${cash:.2f}")

# Save results
daily.to_csv("results/daily_portfolio_2026.csv")
trades.to_csv("results/trades_2026.csv", index=False)
orders.to_csv("results/orders_2026.csv", index=False)
report.to_csv("results/trading_report_2026.csv")

# Equity chart
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(daily.index, daily["Strategy Equity"], label="SMA20/50")
ax.plot(daily.index, daily["Buy Hold Equity"], label="Buy & Hold")

ax.set_title("SPY Portfolio Simulation - 2026")
ax.set_ylabel("Portfolio Value (USD)")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("results/trading_equity_2026.png", dpi=150)
plt.close(fig)

print("\nResults saved successfully.")

