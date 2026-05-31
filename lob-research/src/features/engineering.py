"""
LOB feature engineering.

All features are causal, normalized with rolling z-score, and physically
interpretable. Labels use per-symbol computation to prevent cross-symbol
contamination. Cross-day splitting keeps train and test on entirely
separate calendar days for rigorous generalization testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _compute_labels_single(
    mid: np.ndarray,
    k: int,
    cal_alpha: float,
) -> np.ndarray:
    """Compute labels for a single contiguous price series."""
    labels = np.full(len(mid), np.nan)
    for t in range(len(mid) - k):
        future_smooth = mid[t+1 : t+k+1].mean()
        ret = (future_smooth - mid[t]) / mid[t]
        if abs(ret) > 0.10:
            continue
        if ret > cal_alpha:
            labels[t] = 1
        elif ret < -cal_alpha:
            labels[t] = -1
        else:
            labels[t] = 0
    return labels


def compute_labels(
    df: pd.DataFrame,
    horizons: list[int] = [10, 20, 50],
    alpha: Optional[float] = None,
    target_flat_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Compute mid-price movement labels per symbol.

    If alpha is None, auto-calibrates on the full dataset's return
    distribution to achieve approximately target_flat_pct stationary labels.
    If alpha is provided (e.g. calibrated from a train set), applies it
    directly — use this for test sets to prevent leakage.
    """
    result = df.copy()

    if "symbol" in df.columns:
        symbol_groups = list(df.groupby("symbol", sort=False))
    else:
        symbol_groups = [("ALL", df)]

    for k in horizons:
        all_labels = pd.Series(np.nan, index=df.index)

        if alpha is None:
            all_returns = []
            for sym, grp in symbol_groups:
                mid = grp["mid_price"].values
                for t in range(len(mid) - k):
                    future_smooth = mid[t+1 : t+k+1].mean()
                    ret = (future_smooth - mid[t]) / mid[t]
                    if abs(ret) < 0.10:
                        all_returns.append(abs(ret))
            cal_alpha = float(np.quantile(all_returns, 1.0 - target_flat_pct))
            logger.info(
                f"k={k}: auto-calibrated alpha={cal_alpha:.6f} "
                f"({cal_alpha*100:.4f}% | targets {target_flat_pct*100:.0f}% flat)"
            )
        else:
            cal_alpha = alpha

        total_up = total_flat = total_dn = 0
        for sym, grp in symbol_groups:
            mid    = grp["mid_price"].values
            labels = _compute_labels_single(mid, k, cal_alpha)
            all_labels.loc[grp.index] = labels
            total_up   += int((labels == 1).sum())
            total_flat += int((labels == 0).sum())
            total_dn   += int((labels == -1).sum())

        result[f"label_k{k}"] = all_labels
        result[f"alpha_k{k}"] = cal_alpha
        n = len(df)
        logger.info(
            f"Labels k={k:>3}: "
            f"up={total_up:>6,} ({total_up/n*100:.1f}%) | "
            f"flat={total_flat:>6,} ({total_flat/n*100:.1f}%) | "
            f"down={total_dn:>6,} ({total_dn/n*100:.1f}%)"
        )

    return result


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _rolling_zscore(
    x: np.ndarray,
    window: int = 500,
    min_periods: int = 10,
    eps: float = 1e-8,
) -> np.ndarray:
    s         = pd.Series(x)
    roll_mean = s.rolling(window, min_periods=min_periods).mean()
    roll_std  = s.rolling(window, min_periods=min_periods).std()
    return ((s - roll_mean) / (roll_std + eps)).values


def compute_features(df: pd.DataFrame, n_levels: int = 10) -> pd.DataFrame:
    """Compute full causal feature matrix from raw LOB snapshots."""
    result = df.copy()
    mid    = df["mid_price"].values

    # Group A: Price features
    result["f_mid_return"]    = np.log(mid / np.concatenate([[mid[0]], mid[:-1]]))
    result["f_log_spread"]    = np.log(df["spread"].clip(lower=1e-8))
    result["f_spread_zscore"] = _rolling_zscore(df["spread"].values)

    # Group B: Order imbalance
    result["f_oib_l1"] = df["order_imbalance_l1"]

    for i in range(1, n_levels + 1):
        bid_col, ask_col = f"bid_size_{i}", f"ask_size_{i}"
        if bid_col not in df.columns:
            continue
        bid_v = df[bid_col].fillna(0).values
        ask_v = df[ask_col].fillna(0).values
        total = bid_v + ask_v
        result[f"f_oib_l{i}"] = np.where(total > 0, (bid_v - ask_v) / total, 0.0)

    for k in [3, 5, 10]:
        bid_total = sum(df[f"bid_size_{i}"].fillna(0) for i in range(1, min(k, n_levels)+1))
        ask_total = sum(df[f"ask_size_{i}"].fillna(0) for i in range(1, min(k, n_levels)+1))
        total     = bid_total + ask_total
        result[f"f_cum_oib_top{k}"] = np.where(
            total > 0, (bid_total - ask_total) / total, 0.0
        )

    weights = np.array([1.0 / i for i in range(1, n_levels + 1)])
    weights /= weights.sum()
    bid_w = sum(weights[i-1] * df[f"bid_size_{i}"].fillna(0) for i in range(1, n_levels+1))
    ask_w = sum(weights[i-1] * df[f"ask_size_{i}"].fillna(0) for i in range(1, n_levels+1))
    tot_w = bid_w + ask_w
    result["f_weighted_oib"] = np.where(tot_w > 0, (bid_w - ask_w) / tot_w, 0.0)

    # Group C: Depth features
    result["f_bid_depth_5"] = sum(df[f"bid_size_{i}"].fillna(0) for i in range(1, 6))
    result["f_ask_depth_5"] = sum(df[f"ask_size_{i}"].fillna(0) for i in range(1, 6))
    depth_ratio = result["f_bid_depth_5"] / result["f_ask_depth_5"].replace(0, np.nan)
    result["f_log_depth_ratio"] = np.log(depth_ratio.fillna(1.0))

    for side in ["bid", "ask"]:
        l1 = df[f"{side}_size_1"].replace(0, np.nan)
        for i in range(2, 6):
            result[f"f_{side}_depth_rel_{i}"] = (
                df[f"{side}_size_{i}"].fillna(0) / l1
            ).fillna(0.0)

    # Group D: Pressure / momentum
    for window in [5, 20, 100]:
        result[f"f_oib_ma_{window}"] = (
            result["f_oib_l1"].rolling(window, min_periods=1).mean()
        )
    result["f_oib_momentum"] = result["f_oib_ma_5"] - result["f_oib_ma_100"]
    for window in [5, 20]:
        result[f"f_return_ma_{window}"] = (
            result["f_mid_return"].rolling(window, min_periods=1).mean()
        )

    # Rolling z-score on select features
    for col in [
        "f_log_spread", "f_bid_depth_5", "f_ask_depth_5",
        "f_log_depth_ratio", "f_cum_oib_top3", "f_cum_oib_top5",
        "f_weighted_oib",
    ]:
        if col in result.columns:
            result[f"{col}_z"] = _rolling_zscore(result[col].values)

    logger.info(
        f"Feature engineering complete | "
        f"input_cols={len(df.columns)} | output_cols={len(result.columns)} | "
        f"new_features={len(result.columns) - len(df.columns)}"
    )
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def get_label_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("label_")]


# ---------------------------------------------------------------------------
# Within-day dataset builder (used for single-day experiments)
# ---------------------------------------------------------------------------

def build_dataset(
    df: pd.DataFrame,
    horizon: int = 20,
    sequence_len: int = 100,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    drop_stationary: bool = False,
) -> dict:
    """Per-symbol temporal split within a single DataFrame."""
    label_col = f"label_k{horizon}"
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")

    feat_cols = get_feature_columns(df)

    if "symbol" in df.columns:
        symbol_groups = [(sym, grp) for sym, grp in df.groupby("symbol", sort=False)]
    else:
        symbol_groups = [("ALL", df)]

    splits_train = {"X": [], "y": []}
    splits_val   = {"X": [], "y": []}
    splits_test  = {"X": [], "y": []}

    for sym, grp in symbol_groups:
        df_clean = grp.dropna(subset=feat_cols + [label_col]).copy()
        if drop_stationary:
            df_clean = df_clean[df_clean[label_col] != 0]
            df_clean[label_col] = (df_clean[label_col] == 1).astype(int)

        X_raw = df_clean[feat_cols].values.astype(np.float32)
        y_raw = df_clean[label_col].values.astype(np.int64)
        n     = len(X_raw)

        if n <= sequence_len + 10:
            logger.warning(f"Symbol {sym}: insufficient rows ({n}), skipping.")
            continue

        X_seq, y_seq = [], []
        for i in range(sequence_len, n):
            X_seq.append(X_raw[i - sequence_len : i])
            y_seq.append(y_raw[i])

        X = np.stack(X_seq)
        y = np.array(y_seq)
        n_total = len(X)
        n_train = int(n_total * train_frac)
        n_val   = int(n_total * val_frac)

        splits_train["X"].append(X[:n_train])
        splits_train["y"].append(y[:n_train])
        splits_val["X"].append(X[n_train : n_train + n_val])
        splits_val["y"].append(y[n_train : n_train + n_val])
        splits_test["X"].append(X[n_train + n_val :])
        splits_test["y"].append(y[n_train + n_val :])

        logger.info(f"  {sym}: train={n_train:,} | val={n_val:,} | "
                    f"test={n_total-n_train-n_val:,}")

    X_train = np.concatenate(splits_train["X"])
    y_train = np.concatenate(splits_train["y"])
    X_val   = np.concatenate(splits_val["X"])
    y_val   = np.concatenate(splits_val["y"])
    X_test  = np.concatenate(splits_test["X"])
    y_test  = np.concatenate(splits_test["y"])

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_names": feat_cols,
        "n_features":    len(feat_cols),
        "sequence_len":  sequence_len,
        "horizon":       horizon,
    }

    logger.info(
        f"Dataset built | horizon=k{horizon} | seq_len={sequence_len} | "
        f"features={len(feat_cols)} | "
        f"train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}"
    )
    for split_name, y_s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_s, return_counts=True)
        balance = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(f"  {split_name} class distribution: {balance}")

    return result


# ---------------------------------------------------------------------------
# Cross-day dataset builder — train and test are separate DataFrames
# ---------------------------------------------------------------------------

def build_dataset_cross_day(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int = 20,
    sequence_len: int = 100,
    val_frac: float = 0.15,
) -> dict:
    """
    Build train/val/test splits where test is a completely separate day.

    Train and val come from df_train (split per-symbol temporally).
    Test comes entirely from df_test (different calendar day).

    This is a strict out-of-sample generalization test:
    the model never sees any data from the test day during training.

    Alpha must already be applied to both DataFrames before calling this.

    Args:
        df_train:     Labeled feature DataFrame for training day(s)
        df_test:      Labeled feature DataFrame for test day(s)
        horizon:      Prediction horizon k
        sequence_len: Input sequence length
        val_frac:     Fraction of train data to hold out for validation
    """
    label_col = f"label_k{horizon}"
    for df, name in [(df_train, "train"), (df_test, "test")]:
        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found in {name} DataFrame. "
                "Run compute_labels() first."
            )

    feat_cols = get_feature_columns(df_train)

    # Verify test has same features
    missing = [c for c in feat_cols if c not in df_test.columns]
    if missing:
        raise ValueError(f"Test DataFrame missing features: {missing[:5]}...")

    # ------------------------------------------------------------------
    # Build train + val from df_train (per symbol)
    # ------------------------------------------------------------------
    if "symbol" in df_train.columns:
        train_groups = list(df_train.groupby("symbol", sort=False))
    else:
        train_groups = [("ALL", df_train)]

    X_train_parts, y_train_parts = [], []
    X_val_parts,   y_val_parts   = [], []

    for sym, grp in train_groups:
        df_clean = grp.dropna(subset=feat_cols + [label_col]).copy()
        X_raw = df_clean[feat_cols].values.astype(np.float32)
        y_raw = df_clean[label_col].values.astype(np.int64)
        n     = len(X_raw)

        if n <= sequence_len + 10:
            logger.warning(f"Train symbol {sym}: insufficient rows ({n}), skipping.")
            continue

        X_seq, y_seq = [], []
        for i in range(sequence_len, n):
            X_seq.append(X_raw[i - sequence_len : i])
            y_seq.append(y_raw[i])

        X   = np.stack(X_seq)
        y   = np.array(y_seq)
        n_val   = int(len(X) * val_frac)
        n_train = len(X) - n_val

        X_train_parts.append(X[:n_train])
        y_train_parts.append(y[:n_train])
        X_val_parts.append(X[n_train:])
        y_val_parts.append(y[n_train:])

        unique, counts = np.unique(y[:n_train], return_counts=True)
        logger.info(
            f"  Train {sym}: train={n_train:,} | val={n_val:,} | "
            f"classes={dict(zip(unique.tolist(), counts.tolist()))}"
        )

    # ------------------------------------------------------------------
    # Build test from df_test (per symbol)
    # ------------------------------------------------------------------
    if "symbol" in df_test.columns:
        test_groups = list(df_test.groupby("symbol", sort=False))
    else:
        test_groups = [("ALL", df_test)]

    X_test_parts, y_test_parts = [], []

    for sym, grp in test_groups:
        df_clean = grp.dropna(subset=feat_cols + [label_col]).copy()
        X_raw = df_clean[feat_cols].values.astype(np.float32)
        y_raw = df_clean[label_col].values.astype(np.int64)
        n     = len(X_raw)

        if n <= sequence_len + 10:
            logger.warning(f"Test symbol {sym}: insufficient rows ({n}), skipping.")
            continue

        X_seq, y_seq = [], []
        for i in range(sequence_len, n):
            X_seq.append(X_raw[i - sequence_len : i])
            y_seq.append(y_raw[i])

        X_test_parts.append(np.stack(X_seq))
        y_test_parts.append(np.array(y_seq))

        unique, counts = np.unique(y_raw, return_counts=True)
        logger.info(
            f"  Test {sym}: sequences={len(X_seq):,} | "
            f"classes={dict(zip(unique.tolist(), counts.tolist()))}"
        )

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_val   = np.concatenate(X_val_parts)
    y_val   = np.concatenate(y_val_parts)
    X_test  = np.concatenate(X_test_parts)
    y_test  = np.concatenate(y_test_parts)

    result = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_names": feat_cols,
        "n_features":    len(feat_cols),
        "sequence_len":  sequence_len,
        "horizon":       horizon,
    }

    logger.info(
        f"\nCross-day dataset | horizon=k{horizon} | seq_len={sequence_len} | "
        f"features={len(feat_cols)}"
    )
    logger.info(f"  train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")
    for split_name, y_s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        unique, counts = np.unique(y_s, return_counts=True)
        balance = {int(u): int(c) for u, c in zip(unique, counts)}
        logger.info(f"  {split_name} class distribution: {balance}")

    return result
