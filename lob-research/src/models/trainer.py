"""
Training loop for LOB deep learning models.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import cohen_kappa_score, f1_score, classification_report
from loguru import logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU found — training on CPU will be slow.")
    return device


@torch.no_grad()
def evaluate_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits  = model(X_batch)
        loss    = criterion(logits, y_batch)
        preds   = logits.argmax(dim=1)
        total_loss += loss.item() * len(y_batch)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

    y_true   = np.concatenate(all_labels)
    y_pred   = np.concatenate(all_preds)
    avg_loss = total_loss / len(y_true)

    # Dynamic class handling — don't assume 3 classes are present
    present = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if len(present) < 2:
        kappa    = float("nan")
        macro_f1 = 0.0
    else:
        kappa    = cohen_kappa_score(y_true, y_pred, labels=present)
        macro_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)

    return {
        "loss":     round(avg_loss, 6),
        "kappa":    round(kappa, 4) if not np.isnan(kappa) else float("nan"),
        "macro_f1": round(macro_f1, 4),
        "y_true":   y_true,
        "y_pred":   y_pred,
    }


class Trainer:
    def __init__(self, model: nn.Module, loaders: dict, config: dict):
        self.model   = model
        self.loaders = loaders
        self.config  = config
        self.device  = get_device()
        self.model.to(self.device)

        class_weights  = loaders["class_weights"].to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 50),
            eta_min=1e-6,
        )

        self._best_kappa   = -np.inf
        self._patience_ctr = 0
        self._best_state   = None
        self.history       = {"train": [], "val": []}

        self.exp_dir    = Path(config.get("exp_dir", "experiments"))
        self.model_name = config.get("model_name", model.__class__.__name__)
        self.ckpt_path  = self.exp_dir / f"{self.model_name}_best.pt"
        self.exp_dir.mkdir(exist_ok=True)

    def train(self) -> dict:
        epochs   = self.config.get("epochs", 50)
        patience = self.config.get("patience", 10)
        clip     = self.config.get("grad_clip", 1.0)

        logger.info(
            f"Training {self.model_name} | "
            f"epochs={epochs} | patience={patience} | device={self.device}"
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            t0         = time.time()
            train_loss = 0.0
            n_batches  = 0

            for X_batch, y_batch in self.loaders["train"]:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss   = self.criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            avg_train_loss = train_loss / max(n_batches, 1)

            val_metrics = evaluate_epoch(
                self.model, self.loaders["val"], self.criterion, self.device
            )
            epoch_time = time.time() - t0
            lr_now     = self.optimizer.param_groups[0]["lr"]

            self.history["train"].append({"epoch": epoch, "loss": round(avg_train_loss, 6)})
            self.history["val"].append({
                "epoch": epoch,
                "loss":     val_metrics["loss"],
                "kappa":    val_metrics["kappa"],
                "macro_f1": val_metrics["macro_f1"],
            })

            kappa_str = f"{val_metrics['kappa']:.4f}" if not np.isnan(val_metrics["kappa"]) else "nan"
            logger.info(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_kappa={kappa_str} | "
                f"val_f1={val_metrics['macro_f1']:.4f} | "
                f"lr={lr_now:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            # Early stopping — skip if kappa is nan (single-class val set)
            cur_kappa = val_metrics["kappa"]
            if np.isnan(cur_kappa):
                # Can't improve on nan — just keep going until patience expires
                self._patience_ctr += 1
            elif cur_kappa > self._best_kappa:
                self._best_kappa   = cur_kappa
                self._patience_ctr = 0
                self._best_state   = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                torch.save(self._best_state, self.ckpt_path)
                logger.info(f"  ✓ New best kappa={self._best_kappa:.4f} — checkpoint saved")
            else:
                self._patience_ctr += 1

            if self._patience_ctr >= patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            logger.info(f"Restored best model (kappa={self._best_kappa:.4f})")

        return self.history

    def evaluate_test(self) -> dict:
        logger.info(f"Evaluating {self.model_name} on test set...")
        metrics = evaluate_epoch(
            self.model, self.loaders["test"], self.criterion, self.device
        )

        y_true = metrics.pop("y_true")
        y_pred = metrics.pop("y_pred")

        label_map       = {0: "down", 1: "flat", 2: "up"}
        present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        present_names   = [label_map[c] for c in present_classes if c in label_map]

        report = classification_report(
            y_true, y_pred,
            labels=present_classes,
            target_names=present_names,
            zero_division=0,
        )

        f1_arr = f1_score(
            y_true, y_pred,
            labels=present_classes,
            average=None,
            zero_division=0,
        )
        f1_per = {name: round(float(v), 4) for name, v in zip(present_names, f1_arr)}

        results = {
            "model":    self.model_name,
            "kappa":    metrics["kappa"],
            "macro_f1": metrics["macro_f1"],
            "loss":     metrics["loss"],
            "f1_down":  f1_per.get("down", 0.0),
            "f1_flat":  f1_per.get("flat", 0.0),
            "f1_up":    f1_per.get("up",   0.0),
        }

        logger.info(f"\nTest Results — {self.model_name}")
        logger.info(f"Kappa:    {results['kappa']}")
        logger.info(f"Macro F1: {results['macro_f1']:.4f}")
        logger.info(f"\n{report}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path  = self.exp_dir / f"{self.model_name}_test_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Test results saved → {out_path}")

        return results


def train_model(
    model:      nn.Module,
    loaders:    dict,
    model_name: str,
    epochs:     int   = 50,
    lr:         float = 1e-3,
    patience:   int   = 10,
    exp_dir:    str   = "experiments",
) -> tuple[dict, dict]:
    config = {
        "model_name":   model_name,
        "epochs":       epochs,
        "lr":           lr,
        "patience":     patience,
        "weight_decay": 1e-4,
        "grad_clip":    1.0,
        "exp_dir":      exp_dir,
    }
    trainer      = Trainer(model, loaders, config)
    history      = trainer.train()
    test_results = trainer.evaluate_test()
    return history, test_results
