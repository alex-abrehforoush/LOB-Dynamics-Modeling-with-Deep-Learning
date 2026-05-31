"""
Master pipeline script — cross-day generalization experiment.

Train/val: May 25 data (BTCUSDT + ETHUSDT + SOLUSDT)
Test:      May 13 data (BTCUSDT + ETHUSDT)

This design tests whether models trained on one market session
generalize to a different session with different price levels —
a much stronger claim than within-session temporal splitting.

Key methodological points:
- Alpha (label threshold) calibrated on TRAIN only, applied to test
- Rolling z-score normalization is per-split (no leakage)
- Val set is last 15% of each train symbol, temporally
- Test set is an entirely different calendar day
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.features.engineering import (
    compute_features, compute_labels, build_dataset_cross_day
)
from src.models.dataset import make_dataloaders
from src.models.baselines import run_baselines
from src.models.architectures import build_model
from src.models.trainer import train_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_files(file_list: list[Path]) -> pd.DataFrame:
    """Load and concatenate a specific list of parquet files."""
    dfs = []
    for f in sorted(file_list):
        if not f.exists():
            logger.warning(f"File not found, skipping: {f}")
            continue
        df = pd.read_parquet(f)
        logger.info(f"Loaded {len(df):,} rows from {f}")
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No valid files found.")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp_ms").reset_index(drop=True)
    return combined


def resolve_files(
    data_dir: Path,
    symbols: list[str],
    dates: list[str],
) -> list[Path]:
    files = []
    for sym in symbols:
        for date in dates:
            f = data_dir / sym / f"{date}.parquet"
            if f.exists():
                files.append(f)
            else:
                logger.warning(f"Not found: {f}")
    return files


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_dir   = Path("experiments") / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.add(exp_dir / "run.log", level="DEBUG")
    logger.info(f"Experiment started | output={exp_dir}")
    logger.info(
        f"Design: train={args.train_dates}/{args.train_symbols} | "
        f"test={args.test_dates}/{args.test_symbols}"
    )

    data_dir = Path("data/raw")

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    train_files = resolve_files(data_dir, args.train_symbols, args.train_dates)
    test_files  = resolve_files(data_dir, args.test_symbols,  args.test_dates)

    logger.info(f"Train files: {[f.name for f in train_files]}")
    logger.info(f"Test files:  {[f.name for f in test_files]}")

    df_train_raw = load_files(train_files)
    df_test_raw  = load_files(test_files)
    logger.info(f"Train raw: {len(df_train_raw):,} rows")
    logger.info(f"Test raw:  {len(df_test_raw):,} rows")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    #    Rolling z-score is computed independently on each split
    #    No information flows from test to train
    # ------------------------------------------------------------------
    logger.info("Computing features...")
    df_train_feat = compute_features(df_train_raw, n_levels=10)
    df_test_feat  = compute_features(df_test_raw,  n_levels=10)

    all_results = {}

    for horizon in args.horizons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Horizon k={horizon}")
        logger.info(f"{'='*60}")

        # ------------------------------------------------------------------
        # 3. Labels
        #    Alpha calibrated on train only, then fixed for test
        # ------------------------------------------------------------------
        df_train_labeled = compute_labels(
            df_train_feat,
            horizons=[horizon],
            alpha=None,
            target_flat_pct=0.15,
        )
        cal_alpha = float(df_train_labeled[f"alpha_k{horizon}"].iloc[0])
        logger.info(
            f"Train alpha (calibrated): {cal_alpha:.6f} "
            f"({cal_alpha*100:.4f}% of mid-price)"
        )

        df_test_labeled = compute_labels(
            df_test_feat,
            horizons=[horizon],
            alpha=cal_alpha,   # fixed — no leakage
        )

        # ------------------------------------------------------------------
        # 4. Build cross-day splits
        # ------------------------------------------------------------------
        splits = build_dataset_cross_day(
            df_train=df_train_labeled,
            df_test=df_test_labeled,
            horizon=horizon,
            sequence_len=args.seq_len,
            val_frac=0.15,
        )

        if len(splits["X_train"]) < 500:
            logger.error(f"Insufficient training samples ({len(splits['X_train'])}), skipping.")
            continue

        horizon_results = {}

        # ------------------------------------------------------------------
        # 5. Baselines
        # ------------------------------------------------------------------
        logger.info("Running sklearn baselines...")
        baseline_results = run_baselines(splits, output_dir=str(exp_dir))
        horizon_results["baselines"] = baseline_results

        # ------------------------------------------------------------------
        # 6. Deep models
        # ------------------------------------------------------------------
        loaders    = make_dataloaders(splits, batch_size=args.batch_size, num_workers=0)
        n_features = loaders["n_features"]
        seq_len    = loaders["seq_len"]

        # DeepLOB
        logger.info("Training DeepLOB...")
        deeplob = build_model("deeplob", n_features=n_features, seq_len=seq_len)
        try:
            _, deeplob_test = train_model(
                model=deeplob, loaders=loaders,
                model_name=f"DeepLOB_k{horizon}",
                epochs=args.epochs, lr=args.lr,
                patience=args.patience, exp_dir=str(exp_dir),
            )
            horizon_results["deeplob"] = deeplob_test
        except Exception as e:
            logger.error(f"DeepLOB failed: {e}")
            horizon_results["deeplob"] = {
                "error": str(e), "kappa": 0, "macro_f1": 0,
                "f1_down": 0, "f1_up": 0
            }

        # LOBAttention
        logger.info("Training LOBAttention...")
        lob_attn = build_model(
            "lob_attention", n_features=n_features,
            seq_len=seq_len, n_levels=10,
        )
        _, lob_attn_test = train_model(
            model=lob_attn, loaders=loaders,
            model_name=f"LOBAttention_k{horizon}",
            epochs=args.epochs, lr=args.lr,
            patience=args.patience, exp_dir=str(exp_dir),
        )
        horizon_results["lob_attention"] = lob_attn_test

        # ------------------------------------------------------------------
        # 7. Summary table
        # ------------------------------------------------------------------
        logger.info(f"\n{'='*60}")
        logger.info(
            f"RESULTS — k={horizon} | "
            f"trained on {args.train_dates} | tested on {args.test_dates}"
        )
        logger.info(
            f"{'Model':<30} {'Kappa':>8} {'MacroF1':>8} {'F1-dn':>7} {'F1-up':>7}"
        )
        logger.info("-" * 65)

        def _row(name, res):
            if "error" in res:
                logger.info(f"{name:<30} {'ERROR':>8}")
                return
            k  = res.get("kappa",    0)
            f1 = res.get("macro_f1", 0)
            fd = res.get("f1_down",  0)
            fu = res.get("f1_up",    0)
            logger.info(f"{name:<30} {k:>8.4f} {f1:>8.4f} {fd:>7.4f} {fu:>7.4f}")

        _row("LogisticRegression",
             baseline_results["logistic_regression"]["test"])
        _row("MLP",
             baseline_results["mlp"]["test"])
        _row("DeepLOB",      horizon_results["deeplob"])
        _row("LOBAttention", horizon_results["lob_attention"])

        all_results[f"k{horizon}"] = horizon_results

    # Save full results
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "model_obj"}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = exp_dir / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)

    logger.info(f"\nAll results saved → {out_path}")
    logger.info(f"Experiment complete | dir={exp_dir}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOB cross-day generalization experiment"
    )
    parser.add_argument(
        "--train-symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        help="Symbols to train on"
    )
    parser.add_argument(
        "--train-dates", nargs="+",
        default=["2026-05-25"],
        help="Dates to use for training (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Symbols to test on"
    )
    parser.add_argument(
        "--test-dates", nargs="+",
        default=["2026-05-13"],
        help="Dates to use for testing (YYYY-MM-DD)"
    )
    parser.add_argument("--horizons",   nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seq-len",    type=int,   default=100)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=512)

    args = parser.parse_args()
    run_experiment(args)
