"""
Binance limit order book data collector.

Collects full-depth LOB snapshots via the Binance REST API and stores
them as compressed parquet files, partitioned by symbol and date.

Design decisions:
- REST snapshots (not websocket diff stream) for simplicity and reproducibility.
  A diff stream would require careful sequence number reconciliation; for
  research purposes clean periodic snapshots are preferable.
- Parquet + snappy compression: columnar, fast to read, ~10x smaller than CSV.
- Each snapshot is a single row with 40 price/size columns (10 bid levels,
  10 ask levels) plus metadata. This matches the LOBSTER format convention
  closely enough that feature engineering code transfers directly.
- Loguru for structured logging — every collection run is fully auditable.
"""

import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"

# Binance allows up to 5000 weight per minute. A depth request with
# limit=1000 costs 50 weight. We stay well under by limiting frequency.
MAX_DEPTH_LEVELS = 20        # We collect 20 levels; use top 10 for modeling
SNAPSHOT_INTERVAL_SEC = 0.5  # 2 snapshots/second per symbol — safe rate limit
REQUEST_TIMEOUT_SEC = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_snapshot(snapshot: dict) -> bool:
    """
    Validate a raw Binance depth snapshot.

    Checks:
    - Required keys present
    - Bid price > Ask price would indicate a crossed book (data error)
    - No negative prices or sizes
    - Prices are monotonically ordered (bids descending, asks ascending)

    Returns True if the snapshot passes all checks.
    """
    if "bids" not in snapshot or "asks" not in snapshot:
        logger.warning("Snapshot missing bids or asks keys")
        return False

    bids = snapshot["bids"]
    asks = snapshot["asks"]

    if len(bids) == 0 or len(asks) == 0:
        logger.warning("Empty bids or asks")
        return False

    try:
        bid_prices = [float(b[0]) for b in bids]
        ask_prices = [float(a[0]) for a in asks]
        bid_sizes  = [float(b[1]) for b in bids]
        ask_sizes  = [float(a[1]) for a in asks]
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse prices/sizes: {e}")
        return False

    # No negative values
    if any(p <= 0 for p in bid_prices + ask_prices):
        logger.warning("Non-positive price detected")
        return False
    if any(s < 0 for s in bid_sizes + ask_sizes):
        logger.warning("Negative size detected")
        return False

    # Crossed book check
    if bid_prices[0] >= ask_prices[0]:
        logger.warning(
            f"Crossed book: best_bid={bid_prices[0]}, best_ask={ask_prices[0]}"
        )
        return False

    # Monotonic ordering
    if any(bid_prices[i] <= bid_prices[i+1] for i in range(len(bid_prices)-1)):
        logger.warning("Bids not strictly descending")
        return False
    if any(ask_prices[i] >= ask_prices[i+1] for i in range(len(ask_prices)-1)):
        logger.warning("Asks not strictly ascending")
        return False

    return True


# ---------------------------------------------------------------------------
# Snapshot flattening
# ---------------------------------------------------------------------------

def flatten_snapshot(
    snapshot: dict,
    symbol: str,
    timestamp_ms: int,
    n_levels: int = 10
) -> dict:
    """
    Flatten a raw Binance depth snapshot into a single dict row.

    Output column naming mirrors LOBSTER conventions:
        bid_price_1, bid_size_1, ..., bid_price_N, bid_size_N
        ask_price_1, ask_size_1, ..., ask_price_N, ask_size_N

    Level 1 = best bid/ask (closest to mid). This matches LOBSTER.

    Also computes derived fields used frequently in feature engineering:
        mid_price, spread, spread_bps, order_imbalance_l1
    """
    row = {
        "timestamp_ms": timestamp_ms,
        "symbol": symbol,
        "last_update_id": snapshot.get("lastUpdateId", -1),
    }

    bids = snapshot["bids"][:n_levels]
    asks = snapshot["asks"][:n_levels]

    for i, (price, size) in enumerate(bids, start=1):
        row[f"bid_price_{i}"] = float(price)
        row[f"bid_size_{i}"]  = float(size)

    for i, (price, size) in enumerate(asks, start=1):
        row[f"ask_price_{i}"] = float(price)
        row[f"ask_size_{i}"]  = float(size)

    # Pad with NaN if fewer than n_levels returned
    for i in range(len(bids)+1, n_levels+1):
        row[f"bid_price_{i}"] = np.nan
        row[f"bid_size_{i}"]  = np.nan
    for i in range(len(asks)+1, n_levels+1):
        row[f"ask_price_{i}"] = np.nan
        row[f"ask_size_{i}"]  = np.nan

    # Derived fields
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    bid_vol_l1 = float(bids[0][1])
    ask_vol_l1 = float(asks[0][1])

    row["mid_price"]            = (best_bid + best_ask) / 2.0
    row["spread"]               = best_ask - best_bid
    row["spread_bps"]           = row["spread"] / row["mid_price"] * 10_000
    total_vol_l1 = bid_vol_l1 + ask_vol_l1
    row["order_imbalance_l1"]   = (
        (bid_vol_l1 - ask_vol_l1) / total_vol_l1
        if total_vol_l1 > 0 else 0.0
    )

    return row


# ---------------------------------------------------------------------------
# HTTP fetch with retry
# ---------------------------------------------------------------------------

def fetch_depth(symbol: str, limit: int = MAX_DEPTH_LEVELS) -> Optional[dict]:
    """
    Fetch a single LOB depth snapshot from Binance REST API.

    Args:
        symbol: Binance trading pair, e.g. 'BTCUSDT'
        limit:  Number of price levels to request (max 5000 for Binance)

    Returns:
        Raw snapshot dict, or None on failure after retries.

    Note on limit parameter and API weight:
        limit <= 100  → weight 1
        limit <= 500  → weight 5
        limit <= 1000 → weight 10
        limit <= 5000 → weight 50
    We use limit=20 (weight=1) to stay conservative.
    """
    params = {"symbol": symbol.upper(), "limit": limit}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                BINANCE_DEPTH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SEC
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{MAX_RETRIES}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e}")
            if e.response.status_code in (400, 404):
                return None  # Bad symbol — don't retry
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))

    logger.error(f"All {MAX_RETRIES} attempts failed for {symbol}")
    return None


# ---------------------------------------------------------------------------
# Collection session
# ---------------------------------------------------------------------------

class LOBCollector:
    """
    Collects and stores LOB snapshot sequences for one or more symbols.

    Usage:
        collector = LOBCollector(
            symbols=["BTCUSDT", "ETHUSDT"],
            output_dir="data/raw",
            n_levels=10,
            interval_sec=0.5
        )
        collector.collect(duration_seconds=3600)  # collect for 1 hour

    Storage layout:
        data/raw/
            BTCUSDT/
                2024-01-15.parquet
                2024-01-16.parquet
            ETHUSDT/
                2024-01-15.parquet
    """

    def __init__(
        self,
        symbols: list[str],
        output_dir: str | Path,
        n_levels: int = 10,
        interval_sec: float = SNAPSHOT_INTERVAL_SEC,
    ):
        self.symbols     = [s.upper() for s in symbols]
        self.output_dir  = Path(output_dir)
        self.n_levels    = n_levels
        self.interval_sec = interval_sec

        # In-memory buffer: flushed to parquet every 1000 rows per symbol
        self._buffers: dict[str, list[dict]] = {s: [] for s in self.symbols}
        self._flush_every = 1000

        # Stats
        self._collected: dict[str, int] = {s: 0 for s in self.symbols}
        self._failed:    dict[str, int] = {s: 0 for s in self.symbols}
        self._invalid:   dict[str, int] = {s: 0 for s in self.symbols}

        logger.info(
            f"LOBCollector initialized | symbols={self.symbols} | "
            f"levels={n_levels} | interval={interval_sec}s"
        )

    def collect(self, duration_seconds: int) -> None:
        """
        Collect LOB snapshots for `duration_seconds` seconds.

        Rotates through symbols sequentially. For each symbol, fetches
        one snapshot then sleeps to respect the interval budget.

        Args:
            duration_seconds: Total collection wall-clock time.
        """
        start_time = time.time()
        end_time   = start_time + duration_seconds

        logger.info(
            f"Starting collection | duration={duration_seconds}s | "
            f"estimated snapshots per symbol: "
            f"~{int(duration_seconds / (self.interval_sec * len(self.symbols)))}"
        )

        try:
            while time.time() < end_time:
                for symbol in self.symbols:
                    if time.time() >= end_time:
                        break

                    t_fetch_start = time.time()
                    timestamp_ms  = int(t_fetch_start * 1000)

                    snapshot = fetch_depth(symbol, limit=MAX_DEPTH_LEVELS)

                    if snapshot is None:
                        self._failed[symbol] += 1
                        continue

                    if not validate_snapshot(snapshot):
                        self._invalid[symbol] += 1
                        continue

                    row = flatten_snapshot(
                        snapshot, symbol, timestamp_ms, self.n_levels
                    )
                    self._buffers[symbol].append(row)
                    self._collected[symbol] += 1

                    if len(self._buffers[symbol]) >= self._flush_every:
                        self._flush(symbol)

                    # Rate control: sleep for remaining interval budget
                    elapsed = time.time() - t_fetch_start
                    sleep_time = max(0, self.interval_sec - elapsed)
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Collection interrupted by user")

        finally:
            # Flush all remaining buffers
            for symbol in self.symbols:
                if self._buffers[symbol]:
                    self._flush(symbol)
            self._log_summary()

    def _flush(self, symbol: str) -> None:
        """
        Write the in-memory buffer for `symbol` to a parquet file,
        appending to today's partition if it already exists.
        """
        if not self._buffers[symbol]:
            return

        df = pd.DataFrame(self._buffers[symbol])
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64")

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        symbol_dir = self.output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        out_path = symbol_dir / f"{date_str}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(out_path, compression="snappy", index=False)
        logger.debug(
            f"Flushed {len(self._buffers[symbol])} rows → {out_path} "
            f"(total rows in file: {len(df)})"
        )
        self._buffers[symbol] = []

    def _log_summary(self) -> None:
        logger.info("=== Collection summary ===")
        for symbol in self.symbols:
            total = self._collected[symbol] + self._failed[symbol] + self._invalid[symbol]
            success_rate = (
                self._collected[symbol] / total * 100 if total > 0 else 0
            )
            logger.info(
                f"  {symbol}: collected={self._collected[symbol]} | "
                f"failed={self._failed[symbol]} | "
                f"invalid={self._invalid[symbol]} | "
                f"success_rate={success_rate:.1f}%"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Binance LOB snapshots")
    parser.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Trading pairs to collect (e.g. BTCUSDT ETHUSDT)"
    )
    parser.add_argument(
        "--duration", type=int, default=3600,
        help="Collection duration in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--levels", type=int, default=10,
        help="Number of LOB levels to store (default: 10)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.5,
        help="Seconds between snapshots per symbol (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory (default: data/raw)"
    )
    args = parser.parse_args()

    collector = LOBCollector(
        symbols=args.symbols,
        output_dir=args.output,
        n_levels=args.levels,
        interval_sec=args.interval,
    )
    collector.collect(duration_seconds=args.duration)
