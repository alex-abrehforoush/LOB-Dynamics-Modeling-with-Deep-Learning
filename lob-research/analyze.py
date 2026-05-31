"""
Phase 4: Interpretability and ablation study.

Two analyses:
    1. Attention weight visualization — what does the model attend to,
       and does it change with market state?
    2. Ablation: LOBAttention vs LOBAttention-Uniform (same architecture,
       fixed uniform attention weights) — proves attention is doing work,
       not just model capacity.

Run after a completed experiment:
    python3 analyze.py \
        --exp-dir experiments/20260524_XXXXXX \
        --train-dates 2026-05-25 \
        --test-dates 2026-05-13 \
        --horizon 50
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.features.engineering import (
    compute_features, compute_labels, build_dataset_cross_day
)
from src.models.dataset import make_dataloaders, LOBDataset
from src.models.architectures import LevelConditionalAttention, build_model
from src.models.trainer import train_model, get_device
from src.models.baselines import evaluate


# ---------------------------------------------------------------------------
# Uniform-attention ablation model
# ---------------------------------------------------------------------------

class LOBAttentionUniform(LevelConditionalAttention):
    """
    Ablation: same architecture as LOBAttention but attention weights
    are fixed uniform (1/n_levels) rather than learned.

    If LOBAttention >> LOBAttentionUniform at the same parameter count,
    the learned attention is doing real work. If they're similar,
    the gain comes from architecture capacity, not attention.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        batch, seq_len, n_feat = x.shape

        # State encoding (same as parent)
        state_out, _ = self.state_encoder(x)

        # Level feature extraction (same as parent)
        n_level_feats = self.n_levels * self.level_feat_dim
        if n_feat >= n_level_feats:
            level_feats = x[:, :, :n_level_feats]
            level_feats = level_feats.view(
                batch, seq_len, self.n_levels, self.level_feat_dim
            )
        else:
            per_level  = n_feat // self.n_levels
            level_feats = x[:, :, :self.n_levels * per_level]
            level_feats = level_feats.view(batch, seq_len, self.n_levels, per_level)
            level_feats = level_feats[:, :, :, :self.level_feat_dim]

        level_proj = self.level_proj(level_feats)

        # UNIFORM attention weights — override learned attention
        attn_weights = torch.ones(
            batch, seq_len, self.n_levels,
            device=x.device
        ) / self.n_levels

        self._last_attention_weights = attn_weights.detach()

        attn_weights_exp = attn_weights.unsqueeze(-1)
        attended = (level_proj * attn_weights_exp).sum(dim=2)

        combined = torch.cat([attended, x], dim=-1)
        combined = F.relu(self.attended_proj(combined))

        temporal_out, _ = self.temporal_lstm(combined)
        out = temporal_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# ---------------------------------------------------------------------------
# Attention weight extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(
    model: LevelConditionalAttention,
    loader,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """
    Run forward passes and collect attention weights alongside
    market state features for interpretability analysis.

    Returns dict with:
        attn_weights: (N, seq_len, n_levels) — attention weights
        spread_bps:   (N,) — spread in bps at last timestep
        oib_l1:       (N,) — order imbalance at last timestep
        mid_return:   (N,) — recent mid-price return
        labels:       (N,) — true labels
    """
    model.eval()
    all_attn    = []
    all_spread  = []
    all_oib     = []
    all_return  = []
    all_labels  = []

    # Feature column indices — these correspond to the feature order
    # from engineering.py. We extract a few key ones for conditioning.
    # f_spread_zscore is index 2, f_oib_l1 is index 3, f_mid_return is index 0
    IDX_RETURN  = 0
    IDX_SPREAD  = 2
    IDX_OIB     = 3

    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        X_batch = X_batch.to(device)
        _       = model(X_batch)   # forward pass to populate attention cache

        attn = model.get_attention_weights()   # (B, seq_len, n_levels)
        if attn is None:
            continue

        # Take mean attention across sequence length for each sample
        attn_mean = attn.mean(dim=1).cpu().numpy()    # (B, n_levels)
        all_attn.append(attn_mean)

        # Extract market state features from last timestep
        last_step = X_batch[:, -1, :].cpu().numpy()  # (B, n_features)
        all_spread.append(last_step[:, IDX_SPREAD])
        all_oib.append(last_step[:, IDX_OIB])
        all_return.append(last_step[:, IDX_RETURN])
        all_labels.append(y_batch.numpy())

    if not all_attn:
        logger.error("No attention weights collected — check model forward pass.")
        return {}

    return {
        "attn_weights": np.concatenate(all_attn),    # (N, n_levels)
        "spread_zscore": np.concatenate(all_spread),
        "oib_l1":       np.concatenate(all_oib),
        "mid_return":   np.concatenate(all_return),
        "labels":       np.concatenate(all_labels),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_attention_analysis(
    attn_data: dict,
    output_path: Path,
    n_levels: int = 10,
):
    """
    Four-panel attention interpretability figure.

    Panel 1: Mean attention weight per level (overall)
    Panel 2: Attention by spread regime (tight vs wide)
    Panel 3: Attention by order imbalance direction (bid-heavy vs ask-heavy)
    Panel 4: Attention by predicted label (up vs down vs flat)
    """
    attn    = attn_data["attn_weights"]    # (N, n_levels)
    spread  = attn_data["spread_zscore"]
    oib     = attn_data["oib_l1"]
    labels  = attn_data["labels"]
    levels  = np.arange(1, n_levels + 1)

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
    COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    # Panel 1: Overall mean attention per level
    ax1 = fig.add_subplot(gs[0, 0])
    mean_attn = attn.mean(axis=0)
    std_attn  = attn.std(axis=0)
    ax1.bar(levels, mean_attn, color=COLORS[0], alpha=0.8, label="Mean attention")
    ax1.errorbar(levels, mean_attn, yerr=std_attn, fmt="none",
                 color="black", capsize=3, linewidth=1)
    ax1.axhline(1/n_levels, color="gray", linestyle="--", linewidth=1,
                label=f"Uniform (1/{n_levels})")
    ax1.set_xlabel("LOB Level (1 = best bid/ask)")
    ax1.set_ylabel("Mean Attention Weight")
    ax1.set_title("Overall Attention Distribution")
    ax1.legend(fontsize=8)
    ax1.set_xticks(levels)

    # Panel 2: Attention by spread regime
    ax2 = fig.add_subplot(gs[0, 1])
    tight_mask = spread < np.median(spread)
    wide_mask  = ~tight_mask
    ax2.plot(levels, attn[tight_mask].mean(axis=0), "o-",
             color=COLORS[1], label=f"Tight spread (n={tight_mask.sum():,})")
    ax2.plot(levels, attn[wide_mask].mean(axis=0),  "s--",
             color=COLORS[0], label=f"Wide spread (n={wide_mask.sum():,})")
    ax2.axhline(1/n_levels, color="gray", linestyle=":", linewidth=1)
    ax2.set_xlabel("LOB Level")
    ax2.set_ylabel("Mean Attention Weight")
    ax2.set_title("Attention by Spread Regime")
    ax2.legend(fontsize=8)
    ax2.set_xticks(levels)

    # Panel 3: Attention by order imbalance
    ax3 = fig.add_subplot(gs[1, 0])
    bid_heavy  = oib > np.percentile(oib, 66)
    ask_heavy  = oib < np.percentile(oib, 33)
    neutral    = ~bid_heavy & ~ask_heavy
    ax3.plot(levels, attn[bid_heavy].mean(axis=0),  "o-",
             color=COLORS[2], label=f"Bid pressure (n={bid_heavy.sum():,})")
    ax3.plot(levels, attn[ask_heavy].mean(axis=0),  "s--",
             color=COLORS[0], label=f"Ask pressure (n={ask_heavy.sum():,})")
    ax3.plot(levels, attn[neutral].mean(axis=0),    "^:",
             color="gray",    label=f"Neutral (n={neutral.sum():,})")
    ax3.axhline(1/n_levels, color="gray", linestyle=":", linewidth=0.8)
    ax3.set_xlabel("LOB Level")
    ax3.set_ylabel("Mean Attention Weight")
    ax3.set_title("Attention by Order Imbalance")
    ax3.legend(fontsize=8)
    ax3.set_xticks(levels)

    # Panel 4: Attention by true label
    ax4 = fig.add_subplot(gs[1, 1])
    label_map = {0: ("Down", COLORS[1]), 1: ("Flat", "gray"), 2: ("Up", COLORS[2])}
    for cls, (name, color) in label_map.items():
        mask = labels == cls
        if mask.sum() < 10:
            continue
        ax4.plot(levels, attn[mask].mean(axis=0), "o-",
                 color=color, label=f"{name} (n={mask.sum():,})")
    ax4.axhline(1/n_levels, color="gray", linestyle=":", linewidth=0.8)
    ax4.set_xlabel("LOB Level")
    ax4.set_ylabel("Mean Attention Weight")
    ax4.set_title("Attention by True Price Direction")
    ax4.legend(fontsize=8)
    ax4.set_xticks(levels)

    fig.suptitle(
        "LOBAttention: Level-Conditional Attention Weight Analysis\n"
        "Each panel shows which LOB depth levels the model attends to "
        "under different market conditions",
        fontsize=11, y=1.01
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Attention analysis saved → {output_path}")


def plot_model_comparison(results: dict, output_path: Path):
    """Bar chart comparing all models across horizons."""
    horizons = [10, 20, 50]
    models   = ["logistic_regression", "mlp", "deeplob", "lob_attention"]
    labels   = ["LogReg", "MLP", "DeepLOB", "LOBAttn"]
    colors   = ["#78909C", "#90A4AE", "#EF9A9A", "#66BB6A"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, k in zip(axes, horizons):
        key  = f"k{k}"
        if key not in results:
            continue

        kappas = []
        for model in models:
            if model in ["logistic_regression", "mlp"]:
                res = results[key]["baselines"][model]["test"]
            else:
                res = results[key].get(model, {})
            kappas.append(res.get("kappa", 0))

        bars = ax.bar(labels, kappas, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Horizon k={k}\n({k/2:.0f}s prediction)", fontsize=10)
        ax.set_ylabel("Cohen's Kappa (test set)" if k == 10 else "")
        ax.set_ylim(min(-0.01, min(kappas) - 0.005), max(kappas) + 0.02)

        for bar, val in zip(bars, kappas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    fig.suptitle(
        "Cross-Day Generalization: Cohen's Kappa by Model and Horizon\n"
        "Trained on May 25 session | Tested on May 13 session (different price levels)",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    exp_dir  = Path(args.exp_dir)
    out_dir  = exp_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    device   = get_device()
    data_dir = Path("data/raw")

    # Load results
    results_path = exp_dir / "all_results.json"
    with open(results_path) as f:
        results = json.load(f)

    # ------------------------------------------------------------------
    # 1. Model comparison figure
    # ------------------------------------------------------------------
    plot_model_comparison(results, out_dir / "model_comparison.png")

    # ------------------------------------------------------------------
    # 2. Reload data and rebuild test dataset for attention analysis
    # ------------------------------------------------------------------
    logger.info("Reloading data for attention analysis...")

    def load_sym_date(symbols, dates):
        dfs = []
        for sym in symbols:
            for date in dates:
                f = data_dir / sym / f"{date}.parquet"
                if f.exists():
                    dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)

    df_train_raw = load_sym_date(args.train_symbols, args.train_dates)
    df_test_raw  = load_sym_date(args.test_symbols,  args.test_dates)

    df_train_feat = compute_features(df_train_raw, n_levels=10)
    df_test_feat  = compute_features(df_test_raw,  n_levels=10)

    df_train_lab = compute_labels(df_train_feat, horizons=[args.horizon],
                                  alpha=None, target_flat_pct=0.15)
    cal_alpha    = float(df_train_lab[f"alpha_k{args.horizon}"].iloc[0])
    df_test_lab  = compute_labels(df_test_feat,  horizons=[args.horizon],
                                  alpha=cal_alpha)

    splits = build_dataset_cross_day(
        df_train=df_train_lab,
        df_test=df_test_lab,
        horizon=args.horizon,
        sequence_len=100,
        val_frac=0.15,
    )

    loaders    = make_dataloaders(splits, batch_size=256, num_workers=0)
    n_features = loaders["n_features"]

    # ------------------------------------------------------------------
    # 3. Load best LOBAttention checkpoint and extract attention weights
    # ------------------------------------------------------------------
    ckpt_path = exp_dir / f"LOBAttention_k{args.horizon}_best.pt"
    if not ckpt_path.exists():
        logger.error(
            f"Checkpoint not found: {ckpt_path}\n"
            "Make sure --exp-dir points to the experiment that produced "
            "the LOBAttention checkpoint."
        )
        return

    model = LevelConditionalAttention(
        n_features=n_features, n_levels=10, seq_len=100
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    logger.info(f"Loaded LOBAttention checkpoint from {ckpt_path}")

    attn_data = extract_attention_weights(model, loaders["test"], device)
    if attn_data:
        plot_attention_analysis(attn_data, out_dir / "attention_analysis.png")

    # ------------------------------------------------------------------
    # 4. Ablation: train LOBAttention-Uniform and compare
    # ------------------------------------------------------------------
    logger.info("\nRunning ablation: LOBAttention vs LOBAttention-Uniform...")

    uniform_model = LOBAttentionUniform(
        n_features=n_features, n_levels=10, seq_len=100
    )
    n_params_attn    = sum(p.numel() for p in model.parameters())
    n_params_uniform = sum(p.numel() for p in uniform_model.parameters())
    logger.info(f"LOBAttention params:        {n_params_attn:,}")
    logger.info(f"LOBAttention-Uniform params: {n_params_uniform:,}")

    _, uniform_test = train_model(
        model=uniform_model,
        loaders=loaders,
        model_name=f"LOBAttention_Uniform_k{args.horizon}",
        epochs=args.epochs,
        lr=1e-3,
        patience=10,
        exp_dir=str(exp_dir),
    )

    # Compare
    attn_test    = results[f"k{args.horizon}"]["lob_attention"]
    ablation_gap = attn_test["kappa"] - uniform_test["kappa"]

    logger.info(f"\n{'='*50}")
    logger.info(f"ABLATION RESULTS — horizon k={args.horizon}")
    logger.info(f"{'Model':<35} {'Kappa':>8} {'MacroF1':>8}")
    logger.info("-" * 55)
    logger.info(
        f"{'LOBAttention (learned attn)':<35} "
        f"{attn_test['kappa']:>8.4f} {attn_test['macro_f1']:>8.4f}"
    )
    logger.info(
        f"{'LOBAttention-Uniform (fixed attn)':<35} "
        f"{uniform_test['kappa']:>8.4f} {uniform_test['macro_f1']:>8.4f}"
    )
    logger.info(f"\nAttention gain (Δkappa): {ablation_gap:+.4f}")

    if ablation_gap > 0.005:
        logger.info("→ Learned attention provides meaningful improvement over uniform weighting.")
    elif ablation_gap > 0:
        logger.info("→ Learned attention provides marginal improvement.")
    else:
        logger.info("→ Uniform attention matches or beats learned — attention may not be the key factor.")

    # Save ablation results
    ablation_results = {
        "lob_attention_learned": attn_test,
        "lob_attention_uniform": uniform_test,
        "kappa_gap": round(ablation_gap, 4),
        "params_learned": n_params_attn,
        "params_uniform": n_params_uniform,
    }
    with open(out_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    logger.info(f"\nAnalysis complete | outputs in {out_dir}")
    logger.info("Files generated:")
    for p in sorted(out_dir.iterdir()):
        logger.info(f"  {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOB attention analysis and ablation")
    parser.add_argument("--exp-dir",       required=True,
                        help="Path to experiment directory (e.g. experiments/20260524_XXXXXX)")
    parser.add_argument("--train-symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--train-dates",   nargs="+",
                        default=["2026-05-25"])
    parser.add_argument("--test-symbols",  nargs="+",
                        default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--test-dates",    nargs="+",
                        default=["2026-05-13"])
    parser.add_argument("--horizon",       type=int, default=50,
                        help="Which horizon to analyze (default: 50, strongest result)")
    parser.add_argument("--epochs",        type=int, default=50)
    args = parser.parse_args()
    main(args)
