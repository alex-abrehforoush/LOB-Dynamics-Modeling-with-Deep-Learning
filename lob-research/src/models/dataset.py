"""
PyTorch dataset and dataloader utilities for LOB sequence data.

Wraps the numpy arrays produced by features/engineering.py into
PyTorch-compatible Dataset objects with proper dtype handling.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LOBDataset(Dataset):
    """
    PyTorch Dataset wrapping LOB sequence arrays.

    Args:
        X: float32 array of shape (n_samples, seq_len, n_features)
        y: int64 array of shape (n_samples,)
           Labels are {-1, 0, 1} remapped to {0, 1, 2} for CrossEntropyLoss
    """

    # Remap {-1, 0, 1} → {0, 1, 2} for nn.CrossEntropyLoss
    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    LABEL_NAMES = {0: "down", 1: "flat", 2: "up"}

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Remap labels
        y_mapped = np.vectorize(self.LABEL_MAP.get)(y)
        self.y = torch.tensor(y_mapped, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    @property
    def seq_len(self) -> int:
        return self.X.shape[1]

    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted CrossEntropyLoss.
        Addresses the heavy class imbalance typical in LOB label distributions.
        """
        counts = torch.bincount(self.y, minlength=3).float()
        counts = torch.clamp(counts, min=1)
        weights = counts.sum() / (3 * counts)
        return weights


def make_dataloaders(
    splits: dict,
    batch_size: int = 512,
    num_workers: int = 4,
) -> dict:
    """
    Build train/val/test DataLoaders from the splits dict produced by
    features.engineering.build_dataset().

    Args:
        splits:      Output of build_dataset()
        batch_size:  Samples per batch (512 works well for LOB sequences)
        num_workers: Parallel data loading workers

    Returns:
        Dict with keys: train, val, test (DataLoader objects)
        Plus: n_features, seq_len, class_weights
    """
    train_ds = LOBDataset(splits["X_train"], splits["y_train"])
    val_ds   = LOBDataset(splits["X_val"],   splits["y_val"])
    test_ds  = LOBDataset(splits["X_test"],  splits["y_test"])

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,   # CRITICAL: never shuffle time series
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "n_features":    train_ds.n_features,
        "seq_len":       train_ds.seq_len,
        "class_weights": train_ds.class_weights(),
    }
    return loaders
