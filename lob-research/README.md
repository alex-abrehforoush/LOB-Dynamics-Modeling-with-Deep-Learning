# LOB Dynamics Modeling — Research Project

Limit order book mid-price prediction using deep learning, with a focus
on level-conditional attention mechanisms for capturing depth structure.

## Research Question

Standard LOB models (DeepLOB) aggregate features across all price levels
uniformly. This project tests the hypothesis that a level-conditional
attention mechanism — one that learns *which depth levels are informative
conditional on current market state* — outperforms fixed-depth aggregation,
particularly around volatility regime transitions.

## Project Structure

```
lob-research/
├── data/
│   ├── raw/          # Collected snapshots (parquet, partitioned by symbol/date)
│   ├── processed/    # Cleaned and aligned LOB states
│   └── features/     # Engineered feature matrices
├── src/
│   ├── data/
│   │   ├── collector.py    # Binance REST API collector
│   │   └── simulator.py    # Synthetic LOB generator (for dev/testing)
│   ├── features/
│   │   └── engineering.py  # Feature construction + label generation
│   ├── models/             # Model architectures (Phase 3)
│   └── evaluation/         # Walk-forward CV + significance tests (Phase 4)
├── notebooks/              # EDA and result visualization
├── experiments/            # Config files for each experiment run
└── research/               # Notes and writeup drafts
```

## Setup

```bash
# Clone / download project
cd lob-research

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify install
python3 -c "import torch, pandas, numpy; print('Setup OK')"
```

## Phase 1: Data Collection

### Option A — Real Binance data (recommended)

```bash
# Collect 1 hour of BTC + ETH LOB data
python3 src/data/collector.py \
    --symbols BTCUSDT ETHUSDT \
    --duration 3600 \
    --levels 10 \
    --interval 0.5 \
    --output data/raw

# For overnight collection (8 hours):
python3 src/data/collector.py \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --duration 28800 \
    --output data/raw
```

Data is saved to: `data/raw/{SYMBOL}/{YYYY-MM-DD}.parquet`

### Option B — Synthetic data (for pipeline testing)

```bash
python3 src/data/simulator.py \
    --snapshots 50000 \
    --symbol BTCUSDT_SIM \
    --output data/raw
```

## Phase 2: Feature Engineering

```bash
# Run on collected data
python3 src/features/engineering.py

# Output: data/features/{SYMBOL}_features.parquet
```

Features computed:
- Mid-price returns, log spread, spread z-score
- Per-level order imbalance (levels 1–10)
- Cumulative imbalance (top 3, 5, 10 levels)
- Weighted imbalance (inverse-distance weights)
- Depth ratios and relative depth at levels 2–5
- Rolling imbalance momentum (windows: 5, 20, 100)

Labels:
- `label_k10`, `label_k20`, `label_k50`: mid-price direction at horizon k
- Classes: 1 (up), 0 (stationary), -1 (down)

## Data Notes

### Why Binance (not LOBSTER)?
LOBSTER provides equities LOB data but requires academic license approval
that may take weeks. Binance provides the deepest crypto LOB freely via
public API. The core microstructure phenomena (order imbalance, depth
effects, spread dynamics) are present in both markets.

**Limitation to acknowledge in writeup**: crypto microstructure differs
from equities in several ways — 24/7 trading, no designated market makers,
higher idiosyncratic volatility. Results should be interpreted with this
in mind. The methodology transfers; the exact coefficients may not.

### Data volume targets
- Minimum for meaningful results: 100k snapshots (~14 hours at 2/sec)
- Recommended: 500k–1M snapshots (3–7 days)
- For multi-symbol experiments: collect 3+ symbols simultaneously

### Collection tips
- Run collector overnight (Linux: `nohup python3 src/data/collector.py ... &`)
- Monitor with: `tail -f lob_collection.log`
- Check file sizes: `du -sh data/raw/**/*.parquet`

## Phase 3: Models (coming next)

- Logistic regression baseline
- MLP baseline
- DeepLOB (CNN + LSTM) replication
- Level-conditional attention architecture (research contribution)

## Phase 4: Evaluation (coming next)

- Walk-forward cross-validation
- Cohen's kappa, F1 per class (accuracy alone is misleading with class imbalance)
- McNemar's test for pairwise model comparison
- Regime-stratified analysis (performance in high vs low volatility)

## Citation

If this project produces publishable results, primary reference:

```bibtex
@article{zhang2019deeplob,
  title={DeepLOB: Deep Learning for Limit Order Books},
  author={Zhang, Zihao and Zohren, Stefan and Roberts, Stephen},
  journal={IEEE Transactions on Signal Processing},
  year={2019}
}
```
