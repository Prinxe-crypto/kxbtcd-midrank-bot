"""
Paper trading bot for the Mid-Rank Liquidity Edge strategy (KXBTCD rank 5-10).

Each run:
  1. Pulls currently OPEN KXBTCD markets (live, not yet settled)
  2. Groups by event, ranks strikes by volume within each event
  3. For strikes at rank 5-10 with real volume, checks the favored side's bid
  4. If not already logged, opens a paper position at that bid price
  5. Checks previously-open positions: if now settled, logs win/loss and moves
     to closed_positions.csv
  6. Writes a markdown summary for the GitHub Actions run

State persists via open_positions.csv / closed_positions.csv committed back
to the repo, same pattern as the reference-gap-bot.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXBTCD"

MIN_RANK = 5
MAX_RANK = 10
MIN_FAVORED_PRICE = 0.60  # don't bother logging near-coinflip entries

OPEN_POSITIONS_PATH = Path("open_positions.csv")
CLOSED_POSITIONS_PATH = Path("closed_positions.csv")


def get_open_markets():
    all_markets = []
    cursor = None
    while True:
        params = {"series_ticker": SERIES_TICKER, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or not markets:
            break
    return pd.DataFrame(all_markets)


def get_settled_status(tickers):
    # Batch-check settlement status for a list of tickers we have open positions on
    results = {}
    for ticker in tickers:
        resp = requests.get(f"{BASE_URL}/markets/{ticker}")
        if resp.status_code == 200:
            m = resp.json().get("market", {})
            results[ticker] = m
    return results


def load_positions():
    open_pos = pd.read_csv(OPEN_POSITIONS_PATH) if OPEN_POSITIONS_PATH.exists() else pd.DataFrame(
        columns=["ticker", "event_ticker", "rank", "favored_side", "entry_price", "opened_at"]
    )
    closed_pos = pd.read_csv(CLOSED_POSITIONS_PATH) if CLOSED_POSITIONS_PATH.exists() else pd.DataFrame(
        columns=["ticker", "event_ticker", "rank", "favored_side", "entry_price", "result", "won", "pnl", "closed_at"]
    )
    return open_pos, closed_pos


def check_and_close_settled(open_pos, closed_pos):
    if len(open_pos) == 0:
        return open_pos, closed_pos

    still_open_rows = []
    newly_closed_rows = []

    settlement_info = get_settled_status(open_pos["ticker"].tolist())

    for _, row in open_pos.iterrows():
        market = settlement_info.get(row["ticker"])
        if market and market.get("status") == "settled":
            result = str(market.get("result", "")).lower()
            won = (result == row["favored_side"])
            # PnL per $1 risked: if won, receive $1 - entry_price; if lost, lose entry_price
            pnl = (1 - row["entry_price"]) if won else -row["entry_price"]
            newly_closed_rows.append({
                **row.to_dict(),
                "result": result,
                "won": won,
                "pnl": pnl,
                "closed_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            still_open_rows.append(row.to_dict())

    if newly_closed_rows:
        closed_pos = pd.concat([closed_pos, pd.DataFrame(newly_closed_rows)], ignore_index=True)
    open_pos = pd.DataFrame(still_open_rows, columns=open_pos.columns)
    return open_pos, closed_pos


def find_new_entries(markets_df, open_pos):
    markets_df["volume_fp"] = pd.to_numeric(markets_df["volume_fp"], errors="coerce")
    markets_df["yes_bid_dollars"] = pd.to_numeric(markets_df["yes_bid_dollars"], errors="coerce")
    markets_df["no_bid_dollars"] = pd.to_numeric(markets_df["no_bid_dollars"], errors="coerce")

    markets_df = markets_df[markets_df["volume_fp"] > 0].copy()
    markets_df["rank_in_event"] = markets_df.groupby("event_ticker")["volume_fp"].rank(ascending=False, method="first")

    candidates = markets_df[
        (markets_df["rank_in_event"] >= MIN_RANK) & (markets_df["rank_in_event"] <= MAX_RANK)
    ]

    already_open_tickers = set(open_pos["ticker"]) if len(open_pos) > 0 else set()

    new_rows = []
    for _, row in candidates.iterrows():
        if row["ticker"] in already_open_tickers:
            continue

        yes_bid = row["yes_bid_dollars"]
        no_bid = row["no_bid_dollars"]

        if pd.isna(yes_bid) or pd.isna(no_bid):
            continue

        if yes_bid >= MIN_FAVORED_PRICE:
            side, price = "yes", yes_bid
        elif no_bid >= MIN_FAVORED_PRICE:
            side, price = "no", no_bid
        else:
            continue  # neither side favored enough

        new_rows.append({
            "ticker": row["ticker"],
            "event_ticker": row["event_ticker"],
            "rank": int(row["rank_in_event"]),
            "favored_side": side,
            "entry_price": price,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        })

    return pd.DataFrame(new_rows)


def write_summary(open_pos, closed_pos):
    total_closed = len(closed_pos)
    wins = closed_pos["won"].sum() if total_closed > 0 else 0
    win_rate = wins / total_closed if total_closed > 0 else 0
    total_pnl = closed_pos["pnl"].sum() if total_closed > 0 else 0

    summary = f"""## KXBTCD Mid-Rank Edge — Paper Trading Dashboard

**Last run:** {datetime.now(timezone.utc).isoformat()}

### Open positions: {len(open_pos)}
### Closed positions: {total_closed}
### Win rate: {win_rate:.1%}
### Total PnL (per $1 risked per trade): {total_pnl:.4f}

"""
    if len(open_pos) > 0:
        summary += "### Currently open:\n\n"
        summary += open_pos[["ticker", "rank", "favored_side", "entry_price"]].to_markdown(index=False)
        summary += "\n\n"

    if total_closed > 0:
        summary += "### Last 10 closed:\n\n"
        summary += closed_pos.tail(10)[["ticker", "rank", "favored_side", "entry_price", "won", "pnl"]].to_markdown(index=False)

    print(summary)

    # Write to GitHub Actions step summary if running in CI
    import os
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)


def main():
    print("Fetching open KXBTCD markets...")
    markets_df = get_open_markets()
    print(f"Found {len(markets_df)} open strikes")

    open_pos, closed_pos = load_positions()
    print(f"Loaded {len(open_pos)} open positions, {len(closed_pos)} closed positions")

    print("Checking for settled positions...")
    open_pos, closed_pos = check_and_close_settled(open_pos, closed_pos)

    print("Scanning for new rank 5-10 entries...")
    new_entries = find_new_entries(markets_df, open_pos)
    print(f"Found {len(new_entries)} new entries")

    if len(new_entries) > 0:
        open_pos = pd.concat([open_pos, new_entries], ignore_index=True)

    open_pos.to_csv(OPEN_POSITIONS_PATH, index=False)
    closed_pos.to_csv(CLOSED_POSITIONS_PATH, index=False)

    write_summary(open_pos, closed_pos)


if __name__ == "__main__":
    main()
