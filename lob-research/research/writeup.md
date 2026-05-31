# Level-Conditional Attention for Limit Order Book Mid-Price Prediction

**Alireza (Alex) Abrehforoush**  
Master's Candidate, Computer Science, McMaster University  
Vector Institute Research Grant Recipient

---

## Abstract

We propose a level-conditional attention mechanism for short-horizon mid-price prediction from limit order book (LOB) data. Standard LOB models aggregate features across price levels with fixed weights; we hypothesize that which depth levels carry predictive information depends on current market state. Our architecture learns to dynamically weight LOB levels conditioned on a recurrent encoding of recent order flow. Evaluated on a cross-day generalization protocol — trained on one trading session, tested on a separate session with different price levels — our model (LOBAttention) achieves Cohen's κ = 0.052–0.059 across prediction horizons of 5–25 seconds, consistently outperforming DeepLOB (Zhang et al., 2019) and linear baselines. DeepLOB collapses to near-random prediction on the out-of-sample day at the longest horizon, while LOBAttention maintains directional signal across all conditions.

---

## 1. Motivation

The limit order book encodes the full distribution of supply and demand at every price level. Standard deep learning approaches to LOB prediction (DeepLOB; Zhang et al., 2019) apply convolutional filters uniformly across all depth levels, implicitly assuming that each level's informativeness is constant across market conditions.

This assumption is questionable on microstructure grounds. During periods of high adverse selection risk — wide spreads, thin books — market makers widen quotes and the top-of-book becomes less informative about true price. Conversely, during calm trending markets, level-1 order imbalance dominates the short-horizon price signal. A model that can learn *which levels matter conditional on current state* should outperform one that cannot.

**Research question:** Does a level-conditional attention mechanism over LOB depth levels improve out-of-sample mid-price direction prediction relative to uniform-depth aggregation?

---

## 2. Data

**Source:** Binance public REST API — BTCUSDT, ETHUSDT, SOLUSDT  
**Format:** Top-10 bid/ask levels (price + size), sampled at 2 snapshots/second  
**Training:** May 25, 2026 — BTC [$76,991–$77,894], ETH [$2,092–$2,142], SOL [$84.81–$86.52]  
**Test:** May 13, 2026 — BTC [$80,544–$81,321], ETH [$2,296–$2,323]

The test session has meaningfully different price levels from training (+5% BTC, +10% ETH), providing a genuine cross-session generalization test rather than a temporal split within the same market session.

**Label construction (DeepLOB convention):**

$$\text{label}_t = \begin{cases} +1 & \text{if } \bar{m}_{t+1:t+k} / m_t - 1 > \alpha \\ -1 & \text{if } \bar{m}_{t+1:t+k} / m_t - 1 < -\alpha \\ 0 & \text{otherwise} \end{cases}$$

where $\bar{m}_{t+1:t+k}$ is the smoothed future mid-price and $\alpha$ is calibrated on the training set to produce approximately 15% stationary labels. The same $\alpha$ is applied to the test set without re-calibration.

**Prediction horizons:** k = 10, 20, 50 snapshots (5s, 10s, 25s)

**Limitation:** Crypto LOB microstructure differs from equities (no designated market makers, 24/7 trading, higher idiosyncratic volatility). Results should be interpreted as methodology validation rather than direct comparison to equity LOB literature.

---

## 3. Feature Engineering

41 features per timestep, grouped as:

**Price:** log mid-price return, log spread, rolling spread z-score

**Per-level imbalance:** $\text{OIB}_i = (V^b_i - V^a_i) / (V^b_i + V^a_i)$ for levels 1–10

**Cumulative imbalance:** Top-3, top-5, top-10 level aggregations

**Weighted imbalance:** Inverse-distance weighted across levels (motivated by price impact theory)

**Depth:** Log bid/ask depth ratio, relative depth at levels 2–5

**Pressure:** Rolling imbalance means (windows: 5, 20, 100), imbalance momentum

All features are computed causally. Normalization uses rolling z-score with a 500-snapshot window (approximately 4 minutes), preventing any use of future information.

---

## 4. Model Architectures

### 4.1 DeepLOB (Baseline Replication)

Zhang et al. (2019): 2D CNN blocks extract spatial features across (time × level), followed by an Inception module for multi-scale temporal patterns, then an LSTM. 77,248 parameters.

### 4.2 LOBAttention (Proposed)

**State encoder:** LSTM over the full input sequence, producing a context vector $h_t$ at each timestep.

**Level attention:** $\alpha_t = \text{softmax}(W_2 \tanh(W_1 h_t))$ — the context vector determines which LOB levels to attend to.

**Attended representation:** $\tilde{f}_t = \sum_{i=1}^{L} \alpha_{t,i} \cdot \text{proj}(f_{t,i})$ — weighted sum of projected per-level features.

**Temporal aggregation:** Second LSTM over the attended sequence, with final linear projection to 3 classes.

82,637 parameters. The parameter count is intentionally similar to DeepLOB to ensure performance differences reflect architecture, not capacity.

### 4.3 LOBAttention-Uniform (Ablation)

Identical to LOBAttention but with $\alpha_{t,i} = 1/L$ (fixed uniform weights). If LOBAttention-Uniform matches LOBAttention, the gain comes from architecture, not attention. If LOBAttention wins, the learned conditional weighting is doing real work.

---

## 5. Experimental Design

**Cross-day protocol:** All models trained on May 25 data; evaluated on May 13 data. Val set is last 15% of each training symbol's data (temporally).

**Early stopping:** On validation Cohen's κ (not loss — κ accounts for class imbalance; a model that predicts only "flat" achieves high accuracy but κ ≈ 0).

**Loss:** Weighted CrossEntropyLoss with inverse-frequency class weights.

**Primary metric:** Cohen's κ. Secondary: macro F1, per-class F1.

**Optimization:** Adam, lr=1e-3, cosine annealing, gradient clipping at 1.0.

---

## 6. Results

### 6.1 Cross-Day Generalization (Test Set)

| Model | k=10 (5s) | k=20 (10s) | k=50 (25s) |
|---|---|---|---|
| LogisticRegression | 0.029 | 0.026 | 0.033 |
| MLP | 0.018 | 0.026 | 0.010 |
| DeepLOB | 0.023 | 0.019 | -0.000 |
| **LOBAttention** | **0.059** | **0.049** | **0.052** |

*Cohen's κ on May 13 test session. Higher is better; 0 = random.*

**Key findings:**

1. **LOBAttention dominates across all horizons.** It outperforms DeepLOB by 2.6–5.2 kappa points, and outperforms LogisticRegression by 1.9–3.0 points.

2. **DeepLOB fails to generalize at k=50.** κ = -0.000 with F1-down = 0, F1-up = 0 — the model predicts exclusively "flat" on the test session. This is consistent with CNN-based models overfitting to the spatial statistics of the training session's price levels.

3. **LOBAttention generalizes better in relative terms.** Val-to-test kappa degradation: LogReg drops 66%, LOBAttention drops ~45%.

4. **Longer horizons favor attention.** At k=50, the gap between LOBAttention and all other models widens. The 25-second horizon may allow enough time for order flow dynamics captured by the attention mechanism to resolve into price movements.

5. **Linear models beat temporal models on short horizons.** LogisticRegression outperforms MLP and DeepLOB at k=10 and k=20. This suggests the current-snapshot LOB state carries most of the very-short-horizon predictive information, and temporal modeling adds noise. The attention architecture's advantage may derive partly from its ability to adaptively weight this snapshot information.

### 6.2 Ablation Results

The ablation study isolates the impact of the learned attention mechanism by comparing it against a uniform weighting scheme. Both models utilize the exact same architecture and parameter count of 82637 parameters at the 25 second horizon.

**Key findings:**

1. Uniform weighting outperforms learned attention. The LOBAttention Uniform model achieves a kappa of 0.0575, beating the learned attention variant by a gap of 0.0052 points. The uniform model also achieves a higher Macro F1 of 0.3680 and a lower loss of 0.7739.

2. Revising the core hypothesis. The initial hypothesis proposed that dynamic, market state conditional weighting of depth levels would improve prediction. The ablation results reject this. Because the uniform model wins with the exact same parameter count, the learned conditional weighting provides no additional benefit over fixed weights and instead likely overfits to the training session.

3. The true source of architectural gain. Since LOBAttention Uniform still substantially outperforms the DeepLOB baseline, the performance advantage does not come from conditional weighting. Instead, the architectural inductive bias is doing the heavy lifting. Decoupling the level wise feature extraction from the temporal aggregation yields a superior representation compared to the flat convolutional approach used in baseline models.

To further isolate this structural advantage, future work should evaluate a fully collapsed model (LOBAttention NoSplit) that feeds all features directly into an LSTM. If this collapsed version performs worse than the uniform attention model, it will conclusively prove that level separation is the critical driver of performance rather than the attention mechanism itself.


---

## 7. Attention Weight Analysis

Given the results of the ablation study, the analysis of attention weights pivots from interpreting dynamic market behavior to understanding why fixed uniform weights generalize better to out of sample data.

The original research questions asked whether the model concentrates attention on level 1 during trending periods or shifts to deeper levels during periods of high uncertainty. The inferior generalization of the learned model suggests that while the network may attempt to learn these conditional patterns, the informativeness of Limit Order Book levels is fundamentally more stable across different market regimes than initially hypothesized.

Allowing the model to dynamically shift attention weights likely causes it to over index on the spatial and temporal statistics specific to the training day. When applied to a new trading session with entirely different price levels and volatility profiles, these learned dynamic weights miscalibrate. In contrast, the uniform attention mechanism forces the model to aggregate information equally across all depth levels, acting as a powerful structural regularizer. This prevents the network from improperly discarding deeper book information during state changes, thereby preserving a more robust and stable signal for out of sample prediction.

---

## 8. Limitations

1. **Data volume:** One training day (~114k snapshots across 3 symbols) and one test day. Results are promising but require validation across more sessions.

2. **Crypto vs equities:** Binance LOB differs from equity LOB in market structure. Generalization to equity markets is assumed but not demonstrated.

3. **Transaction costs:** All analysis is predictive, not strategic. A predictive edge of κ ≈ 0.05 does not translate directly to a trading edge without careful cost modeling.

4. **Statistical significance:** With one test session, we cannot rule out that LOBAttention's advantage reflects a favorable day rather than a structural improvement. McNemar's test across multiple test sessions would strengthen the claim.

5. **Hyperparameter search:** Models were trained with default hyperparameters. Systematic search may improve all models' performance and could affect relative rankings.

---

## 9. Research Directions

**Near-term:**
- Multi-day evaluation (collect 2–3 weeks of data, rolling train/test windows)
- McNemar's test for statistical significance of pairwise model comparisons
- Regime-stratified analysis: does attention gain more during high-volatility periods?

**Extensions:**
- Cross-asset transfer: train on BTC, test on ETH (microstructure similarity)
- Tick-by-tick LOB data (event-driven rather than time-sampled)
- Incorporate trade data alongside passive book state

**Connection to stochastic control:** The attention mechanism can be interpreted as a learned approximation to the optimal information weighting in a filtering problem — related to the Avellaneda-Stoikov market-making framework where the market maker's reservation price depends on the informativeness of different signals.

---

## References

Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Learning for Limit Order Books. *IEEE Transactions on Signal Processing*. arXiv:1808.03668

Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Mathematical Finance*, 18(3), 489–503.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
