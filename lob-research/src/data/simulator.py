"""
LOB simulator for pipeline development and testing.

Generates synthetic limit order book snapshots with realistic statistical
properties. This is NOT for producing research results — it's for developing
and testing your data pipeline, feature engineering, and model architecture
before real data is available.

The simulator uses a mean-reverting mid-price process (Ornstein-Uhlenbeck)
with stochastic spread and depth, producing LOB snapshots that match the
schema of real Binance data exactly. You can swap in real data by simply
pointing the pipeline at a different parquet directory.

Statistical properties:
- Mid-price: OU process around a drift, calibrated to BTC-like volatility
- Spread: log-normal, regime-switching between tight/wide states
- Depth: exponentially decaying with level, with noise
- Order imbalance: correlated with recent price direction (realistic)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger


# ---------------------------------------------------------------------------
# Simulation parameters (calibrated to approximate BTC/USDT dynamics)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # Mid-price OU process
    "mid_price_init":   65_000.0,   # Starting mid-price (USDT)
    "ou_mean_reversion": 0.01,      # Speed of mean reversion (per tick)
    "ou_vol":            0.0002,    # Per-tick volatility (~0.02% per snapshot)

    # Spread: log-normal, switches between tight/wide regimes
    "spread_tight_mean_bps": 1.0,   # ~$0.65 on BTC at 65k
    "spread_wide_mean_bps":  3.5,
    "spread_vol_bps":        0.3,
    "regime_switch_prob":    0.005, # Probability of switching regime per tick

    # Depth: base volume at level 1, decays exponentially
    "depth_l1_base":    2.5,        # BTC at level 1
    "depth_decay":      0.7,        # Multiplicative decay per level
    "depth_noise":      0.3,        # Fractional noise on depth

    # Order imbalance: correlated with recent price change
    "imbalance_autocorr": 0.4,
}


class LOBSimulator:
    """
    Generates sequences of LOB snapshots matching the real collector schema.

    Usage:
        sim = LOBSimulator(n_levels=10, seed=42)
        df = sim.generate(n_snapshots=10_000, symbol="BTCUSDT_SIM")
        df.to_parquet("data/raw/BTCUSDT_SIM/2024-01-15.parquet")

    The output DataFrame has identical columns to real collector output,
    so all downstream pipeline code works without modification.
    """

    def __init__(
        self,
        n_levels: int = 10,
        params: Optional[dict] = None,
        seed: int = 42,
    ):
        self.n_levels = n_levels
        self.params   = {**DEFAULT_PARAMS, **(params or {})}
        self.rng      = np.random.default_rng(seed)

        # State
        self._mid     = self.params["mid_price_init"]
        self._regime  = "tight"   # "tight" or "wide" spread regime
        self._imb     = 0.0       # Current order imbalance state

        logger.info(
            f"LOBSimulator initialized | levels={n_levels} | seed={seed} | "
            f"mid_init={self._mid}"
        )

    def _step_mid_price(self) -> float:
        """OU step for mid-price."""
        p = self.params
        drift   = p["ou_mean_reversion"] * (p["mid_price_init"] - self._mid)
        shock   = p["ou_vol"] * self._mid * self.rng.standard_normal()
        self._mid = max(1.0, self._mid + drift + shock)
        return self._mid

    def _step_spread(self) -> float:
        """Log-normal spread with regime switching."""
        p = self.params
        if self.rng.random() < p["regime_switch_prob"]:
            self._regime = "wide" if self._regime == "tight" else "tight"

        mean_bps = (
            p["spread_tight_mean_bps"]
            if self._regime == "tight"
            else p["spread_wide_mean_bps"]
        )
        spread_bps = max(
            0.1,
            mean_bps + p["spread_vol_bps"] * self.rng.standard_normal()
        )
        return (spread_bps / 10_000) * self._mid

    def _step_imbalance(self, mid_return: float) -> float:
        """Order imbalance correlated with recent mid-price return."""
        p = self.params
        noise = self.rng.standard_normal() * 0.2
        self._imb = (
            p["imbalance_autocorr"] * self._imb
            + (1 - p["imbalance_autocorr"]) * np.sign(mid_return) * 0.3
            + noise
        )
        return np.clip(self._imb, -1.0, 1.0)

    def _generate_depth(self, base: float, n: int) -> np.ndarray:
        """Exponentially decaying depth with multiplicative noise."""
        p = self.params
        levels = np.array([
            base * (p["depth_decay"] ** i) for i in range(n)
        ])
        noise  = 1.0 + p["depth_noise"] * self.rng.standard_normal(n)
        return np.maximum(0.001, levels * noise)

    def generate_snapshot(
        self,
        symbol: str,
        timestamp_ms: int,
        update_id: int,
    ) -> dict:
        """Generate one synthetic LOB snapshot."""
        prev_mid = self._mid
        mid      = self._step_mid_price()
        spread   = self._step_spread()
        mid_ret  = (mid - prev_mid) / prev_mid
        imb      = self._step_imbalance(mid_ret)

        best_bid = mid - spread / 2
        best_ask = mid + spread / 2

        # Depth: imbalance tilts bid vs ask volume
        bid_base = self.params["depth_l1_base"] * (1 + 0.3 * imb)
        ask_base = self.params["depth_l1_base"] * (1 - 0.3 * imb)
        bid_depths = self._generate_depth(bid_base, self.n_levels)
        ask_depths = self._generate_depth(ask_base, self.n_levels)

        # Price tick size: 0.01 USDT for BTC
        tick = 0.01
        row = {
            "timestamp_ms":       timestamp_ms,
            "symbol":             symbol,
            "last_update_id":     update_id,
            "mid_price":          mid,
            "spread":             spread,
            "spread_bps":         spread / mid * 10_000,
            "order_imbalance_l1": imb,
        }

        for i in range(self.n_levels):
            row[f"bid_price_{i+1}"] = round(best_bid - i * tick * (1 + i * 0.5), 2)
            row[f"bid_size_{i+1}"]  = round(bid_depths[i], 4)
            row[f"ask_price_{i+1}"] = round(best_ask + i * tick * (1 + i * 0.5), 2)
            row[f"ask_size_{i+1}"]  = round(ask_depths[i], 4)

        return row

    def generate(
        self,
        n_snapshots: int,
        symbol: str = "BTCUSDT_SIM",
        start_time_ms: Optional[int] = None,
        interval_ms: int = 500,
    ) -> pd.DataFrame:
        """
        Generate n_snapshots LOB snapshots.

        Args:
            n_snapshots:   Number of snapshots to generate
            symbol:        Symbol label (suffix _SIM signals synthetic data)
            start_time_ms: Starting timestamp in ms (defaults to now)
            interval_ms:   Time between snapshots in ms (500 = 2/sec)

        Returns:
            DataFrame with identical schema to real collector output.
        """
        if start_time_ms is None:
            start_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        rows = []
        for i in range(n_snapshots):
            ts = start_time_ms + i * interval_ms
            row = self.generate_snapshot(symbol, ts, update_id=i)
            rows.append(row)

            if (i + 1) % 10_000 == 0:
                logger.info(f"Generated {i+1:,}/{n_snapshots:,} snapshots")

        df = pd.DataFrame(rows)
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64")

        logger.info(
            f"Simulation complete | snapshots={len(df):,} | "
            f"mid_range=[{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}] | "
            f"mean_spread_bps={df['spread_bps'].mean():.3f}"
        )
        return df

    def save(
        self,
        df: pd.DataFrame,
        output_dir: str | Path,
        symbol: str,
    ) -> Path:
        """Save simulated data in same layout as real collector."""
        output_dir = Path(output_dir)
        symbol_dir = output_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = symbol_dir / f"{date_str}.parquet"
        df.to_parquet(out_path, compression="snappy", index=False)
        logger.info(f"Saved {len(df):,} rows → {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic LOB data")
    parser.add_argument("--snapshots", type=int, default=50_000)
    parser.add_argument("--symbol",    type=str, default="BTCUSDT_SIM")
    parser.add_argument("--output",    type=str, default="data/raw")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    sim = LOBSimulator(n_levels=10, seed=args.seed)
    df  = sim.generate(n_snapshots=args.snapshots, symbol=args.symbol)
    sim.save(df, args.output, args.symbol)
