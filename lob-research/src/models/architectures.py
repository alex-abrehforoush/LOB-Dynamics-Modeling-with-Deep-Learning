"""
Deep learning models for LOB mid-price direction prediction.

Models implemented:
    1. DeepLOB      — replication of Zhang et al. (2019), arXiv:1808.03668
                      CNN feature extraction + Inception + LSTM
    2. LOBAttention — research contribution: level-conditional attention
                      mechanism that learns which depth levels are informative
                      conditional on current market state

Architecture motivation is documented inline — you should be able to
explain every design choice in an interview.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------

class InceptionModule(nn.Module):
    """
    Inception-style temporal feature extractor.

    Applies convolutions at three different kernel sizes (1, 3, 5) in
    parallel and concatenates the outputs. Captures patterns at multiple
    temporal scales simultaneously — short-term microstructure noise vs
    medium-term order flow trends.

    This is the key component from DeepLOB that makes it outperform
    a plain CNN: LOB dynamics have structure at multiple timescales,
    and a fixed kernel size would miss one or the other.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Each branch produces out_channels // 3 channels
        # (rounded to keep total consistent)
        branch_ch = out_channels // 3

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(branch_ch),
            nn.LeakyReLU(0.01),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_ch),
            nn.LeakyReLU(0.01),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_ch),
            nn.LeakyReLU(0.01),
        )
        self.out_channels = branch_ch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, seq_len)
        return torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
        ], dim=1)


# ---------------------------------------------------------------------------
# Model 1: DeepLOB replication
# ---------------------------------------------------------------------------

class DeepLOB(nn.Module):
    """
    Replication of DeepLOB (Zhang et al., 2019).

    Architecture:
        Input: (batch, seq_len, n_features)
        → Reshape to (batch, 1, seq_len, n_features)  [treat as 2D image]
        → Conv2d blocks extract local spatial features across LOB levels
        → Reshape to (batch, channels, seq_len)
        → InceptionModule captures multi-scale temporal patterns
        → LSTM captures long-range temporal dependencies
        → Linear → 3-class softmax

    Key design choice from the paper: treating the LOB snapshot as a
    2D structure (time × price level) and applying 2D convolutions before
    temporal modeling. This lets the CNN learn joint price-level patterns
    before the LSTM sees the sequence.

    Args:
        n_features:  Number of input features per timestep
        seq_len:     Length of input sequence
        lstm_hidden: LSTM hidden state dimension
        n_classes:   Output classes (3: down, flat, up)
        dropout:     Dropout rate after LSTM
    """

    def __init__(
        self,
        n_features:  int,
        seq_len:     int,
        lstm_hidden: int = 64,
        n_classes:   int = 3,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.n_features  = n_features
        self.seq_len     = seq_len
        self.lstm_hidden = lstm_hidden

        # --- Spatial CNN blocks ---
        # Treat (seq_len, n_features) as a 2D image
        # Kernels slide across time and feature dimensions
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # Dynamically compute CNN output size
        self._cnn_out_size = self._get_cnn_out_size()

        # --- Inception temporal module ---
        self.inception  = InceptionModule(32, 64)
        inception_out   = self.inception.out_channels

        # --- LSTM temporal module ---
        self.lstm = nn.LSTM(
            input_size=inception_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, n_classes)

    def _get_cnn_out_size(self) -> tuple:
        """Run a dummy forward pass to compute CNN output shape."""
        dummy = torch.zeros(1, 1, self.seq_len, self.n_features)
        with torch.no_grad():
            try:
                x = self.cnn_block1(dummy)
                x = self.cnn_block2(x)
                x = self.cnn_block3(x)
                return x.shape  # (1, 32, seq_dim, feat_dim)
            except Exception:
                return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, 3)
        """
        batch = x.shape[0]

        # Reshape to 2D image format
        x = x.unsqueeze(1)                          # (B, 1, seq_len, n_feat)

        # CNN blocks
        try:
            x = self.cnn_block1(x)
            x = self.cnn_block2(x)
            x = self.cnn_block3(x)
        except RuntimeError as e:
            raise RuntimeError(
                f"CNN forward failed. Input shape may be incompatible. "
                f"Try increasing seq_len or n_features. Error: {e}"
            )

        # Collapse feature dimension: (B, 32, seq', 1) → (B, 32, seq')
        x = x.squeeze(-1)

        # Inception module
        x = self.inception(x)                       # (B, inception_out, seq')

        # LSTM expects (B, seq', features)
        x = x.permute(0, 2, 1)                      # (B, seq', inception_out)
        x, _ = self.lstm(x)                         # (B, seq', lstm_hidden)

        # Take last timestep
        x = x[:, -1, :]                             # (B, lstm_hidden)
        x = self.dropout(x)
        return self.fc(x)                           # (B, 3)


# ---------------------------------------------------------------------------
# Model 2: Level-Conditional Attention (research contribution)
# ---------------------------------------------------------------------------

class LevelConditionalAttention(nn.Module):
    """
    Level-conditional attention mechanism.

    Research hypothesis: which LOB depth levels carry predictive information
    is not fixed — it depends on current market state (volatility regime,
    order flow pressure). A static aggregation (as in DeepLOB) cannot adapt.

    Architecture:
        Input: (batch, seq_len, n_features)

        1. State encoder: lightweight LSTM encodes recent market state
           into a context vector h_t

        2. Level attention: for each timestep, h_t is used to compute
           attention weights over the n_levels LOB depth levels.
           Weights are softmax-normalized, so the model learns to focus
           on whichever levels are currently informative.

        3. Attended features: weighted sum of level features, concatenated
           with global features (spread, returns, imbalance sums)

        4. Temporal aggregation: second LSTM over attended sequence

        5. Linear → 3-class output

    The key ablation (Phase 4) will compare:
        - Fixed uniform weighting (→ standard LOB model)
        - Fixed inverse-distance weighting (→ our feature engineering default)
        - Learned conditional weighting (→ this model)

    Args:
        n_features:     Total features per timestep
        n_levels:       Number of LOB depth levels (for attention)
        seq_len:        Input sequence length
        state_hidden:   State encoder LSTM hidden size
        attn_hidden:    Attention MLP hidden size
        temporal_hidden: Temporal aggregation LSTM hidden size
        n_classes:      Output classes (3)
        dropout:        Dropout rate
    """

    def __init__(
        self,
        n_features:      int,
        n_levels:        int = 10,
        seq_len:         int = 100,
        state_hidden:    int = 32,
        attn_hidden:     int = 32,
        temporal_hidden: int = 64,
        n_classes:       int = 3,
        dropout:         float = 0.2,
    ):
        super().__init__()
        self.n_features      = n_features
        self.n_levels        = n_levels
        self.seq_len         = seq_len
        self.state_hidden    = state_hidden
        self.temporal_hidden = temporal_hidden

        # Features per level (bid_price, bid_size, ask_price, ask_size = 4)
        self.level_feat_dim = 4

        # Global features: everything that's not per-level
        # (mid_return, log_spread, imbalances, depth ratios, etc.)
        self.global_feat_dim = n_features - n_levels * self.level_feat_dim
        self.global_feat_dim = max(self.global_feat_dim, n_features // 2)

        # --- State encoder: encodes recent market context ---
        self.state_encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=state_hidden,
            num_layers=1,
            batch_first=True,
        )

        # --- Attention network: context → level weights ---
        # Input: state_hidden (context vector)
        # Output: n_levels (unnormalized attention scores)
        self.attention_net = nn.Sequential(
            nn.Linear(state_hidden, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, n_levels),
        )

        # --- Per-level feature projection ---
        # Each level has level_feat_dim raw features; project to attn_hidden
        self.level_proj = nn.Linear(self.level_feat_dim, attn_hidden)

        # --- Attended representation projection ---
        # attended: attn_hidden (weighted sum of projected levels)
        # global:   n_features (full feature vector, as context)
        self.attended_proj = nn.Linear(attn_hidden + n_features, temporal_hidden)

        # --- Temporal aggregation LSTM ---
        self.temporal_lstm = nn.LSTM(
            input_size=temporal_hidden,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(temporal_hidden, n_classes)

        # Store attention weights for interpretability
        self._last_attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            logits: (batch, 3)
        """
        batch, seq_len, n_feat = x.shape

        # --- State encoding: encode full sequence for context ---
        state_out, _ = self.state_encoder(x)    # (B, seq_len, state_hidden)

        # --- Level feature extraction ---
        # Extract per-level features: bid_price_i, bid_size_i, ask_price_i, ask_size_i
        # These are the first n_levels*4 columns of x (by our feature engineering convention)
        # Shape: (B, seq_len, n_levels, 4)
        n_level_feats = self.n_levels * self.level_feat_dim
        if n_feat >= n_level_feats:
            level_feats = x[:, :, :n_level_feats]
            level_feats = level_feats.view(batch, seq_len, self.n_levels, self.level_feat_dim)
        else:
            # Fallback: split features evenly if exact structure not available
            per_level = n_feat // self.n_levels
            level_feats = x[:, :, :self.n_levels * per_level]
            level_feats = level_feats.view(batch, seq_len, self.n_levels, per_level)
            # Project to level_feat_dim
            level_feats = level_feats[:, :, :, :self.level_feat_dim]

        # Project each level's features: (B, seq_len, n_levels, attn_hidden)
        level_proj = self.level_proj(level_feats)

        # --- Attention weights from state context ---
        # Use last state_hidden at each timestep as context
        # attn_scores: (B, seq_len, n_levels)
        attn_scores   = self.attention_net(state_out)
        attn_weights  = F.softmax(attn_scores, dim=-1)

        # Store for interpretability (detached from graph)
        self._last_attention_weights = attn_weights.detach()

        # --- Attended level features ---
        # Weighted sum: (B, seq_len, attn_hidden)
        attn_weights_exp = attn_weights.unsqueeze(-1)   # (B, seq_len, n_levels, 1)
        attended = (level_proj * attn_weights_exp).sum(dim=2)  # (B, seq_len, attn_hidden)

        # Concatenate with full feature vector as global context
        combined = torch.cat([attended, x], dim=-1)   # (B, seq_len, attn_h + n_feat)
        combined = F.relu(self.attended_proj(combined))  # (B, seq_len, temporal_hidden)

        # --- Temporal aggregation ---
        temporal_out, _ = self.temporal_lstm(combined)  # (B, seq_len, temporal_hidden)

        # Last timestep
        out = temporal_out[:, -1, :]                   # (B, temporal_hidden)
        out = self.dropout(out)
        return self.fc(out)                             # (B, 3)

    def get_attention_weights(self) -> torch.Tensor | None:
        """
        Return the attention weights from the last forward pass.

        Shape: (batch, seq_len, n_levels)
        Use for interpretability analysis: which levels does the model
        attend to, and how does this change with market state?
        """
        return self._last_attention_weights


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    name:       str,
    n_features: int,
    seq_len:    int,
    n_levels:   int = 10,
    **kwargs,
) -> nn.Module:
    """
    Build a model by name.

    Args:
        name:       "deeplob" or "lob_attention"
        n_features: Number of input features per timestep
        seq_len:    Input sequence length
        n_levels:   Number of LOB depth levels (for attention model)
        **kwargs:   Additional model hyperparameters

    Returns:
        Initialized nn.Module
    """
    name = name.lower()

    if name == "deeplob":
        model = DeepLOB(n_features=n_features, seq_len=seq_len, **kwargs)
    elif name in ("lob_attention", "lobattention"):
        model = LevelConditionalAttention(
            n_features=n_features,
            n_levels=n_levels,
            seq_len=seq_len,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'deeplob' or 'lob_attention'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Built {model.__class__.__name__} | params={n_params:,}")
    return model
