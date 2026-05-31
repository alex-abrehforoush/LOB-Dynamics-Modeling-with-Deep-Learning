"""
Baseline models for LOB mid-price direction prediction.

These are not throwaway — they serve three purposes:
1. Sanity check: if deep models can't beat these, something is wrong
2. Performance floor: all improvements are relative to this
3. Interview talking point: you can explain exactly what each baseline captures
   and why it's insufficient, motivating your architecture

Baselines:
    LogisticRegression  — linear model on last snapshot's features only
    MLP                 — nonlinear model on flattened sequence (no temporal structure)

Both are sklearn-based for simplicity. Results are saved to experiments/.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
    confusion_matrix,
)
from loguru import logger


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict = {0: "down", 1: "flat", 2: "up"},
) -> dict:
    """
    Compute and log a comprehensive evaluation report.

    Dynamically handles 2-class (no flat) or 3-class label distributions.

    Metrics used and why:
    - Cohen's Kappa: accounts for class imbalance (accuracy alone is misleading
      when 95%+ of samples are 'flat')
    - F1 per class: captures precision/recall tradeoff for each direction
    - Macro F1: unweighted mean across classes (penalizes ignoring minorities)
    - Confusion matrix: reveals systematic errors (e.g. always predicting flat)
    """
    # Detect which classes are actually present
    present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    present_names   = [label_map[c] for c in present_classes if c in label_map]

    kappa   = cohen_kappa_score(y_true, y_pred)
    f1_mac  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per  = f1_score(
        y_true, y_pred,
        labels=present_classes,
        average=None,
        zero_division=0,
    )
    cm     = confusion_matrix(y_true, y_pred, labels=present_classes)
    report = classification_report(
        y_true, y_pred,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {name}")
    logger.info(f"Classes present: {present_names}")
    logger.info(f"Cohen's Kappa:   {kappa:.4f}")
    logger.info(f"Macro F1:        {f1_mac:.4f}")
    f1_str = " | ".join(f"{n}={v:.4f}" for n, v in zip(present_names, f1_per))
    logger.info(f"F1 per class:    {f1_str}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Build result dict — always include all three keys, default 0.0 if absent
    class_f1 = {name: round(float(v), 4) for name, v in zip(present_names, f1_per)}
    return {
        "model":            name,
        "kappa":            round(kappa, 4),
        "macro_f1":         round(f1_mac, 4),
        "f1_down":          class_f1.get("down", 0.0),
        "f1_flat":          class_f1.get("flat", 0.0),
        "f1_up":            class_f1.get("up", 0.0),
        "classes_present":  present_names,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------------

def run_logistic_regression(splits: dict) -> dict:
    """
    Logistic regression on the last timestep's features only.

    Motivation: if the current LOB snapshot alone (no history) has
    predictive power, logistic regression will find it. It's the
    simplest possible causal model.

    Limitation: ignores all temporal dynamics — no memory of how the
    book evolved to reach this state.
    """
    logger.info("Running Logistic Regression baseline...")

    # Use only the last timestep of each sequence
    X_train = splits["X_train"][:, -1, :]  # (n, n_features)
    X_val   = splits["X_val"][:,   -1, :]
    X_test  = splits["X_test"][:,  -1, :]

    # Remap {-1, 0, 1} → {0, 1, 2}
    label_map = {-1: 0, 0: 1, 1: 2}
    y_train = np.vectorize(label_map.get)(splits["y_train"])
    y_val   = np.vectorize(label_map.get)(splits["y_val"])
    y_test  = np.vectorize(label_map.get)(splits["y_test"])

    # Standardize (LR is sensitive to feature scale)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)

    val_results  = evaluate("LogisticRegression [val]",  y_val,  model.predict(X_val))
    test_results = evaluate("LogisticRegression [test]", y_test, model.predict(X_test))

    return {"val": val_results, "test": test_results, "model_obj": model}


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

def run_mlp(splits: dict) -> dict:
    """
    MLP on flattened sequence features.

    Motivation: captures nonlinear feature interactions but treats
    the sequence as a flat bag of features — no notion of temporal order.
    If MLP >> LR, the nonlinearity matters. If DeepLOB >> MLP,
    the temporal structure matters.

    Architecture: 3 hidden layers, decreasing width, dropout-like
    regularization via alpha (L2).
    """
    logger.info("Running MLP baseline...")

    # Flatten sequence: (n, seq_len, n_features) → (n, seq_len * n_features)
    n_train, seq_len, n_feat = splits["X_train"].shape
    X_train = splits["X_train"].reshape(n_train, -1)
    X_val   = splits["X_val"].reshape(len(splits["X_val"]), -1)
    X_test  = splits["X_test"].reshape(len(splits["X_test"]), -1)

    label_map = {-1: 0, 0: 1, 1: 2}
    y_train = np.vectorize(label_map.get)(splits["y_train"])
    y_val   = np.vectorize(label_map.get)(splits["y_val"])
    y_test  = np.vectorize(label_map.get)(splits["y_test"])

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=100,
        alpha=1e-3,         # L2 regularization
        learning_rate_init=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    logger.info(f"MLP converged after {model.n_iter_} iterations")

    val_results  = evaluate("MLP [val]",  y_val,  model.predict(X_val))
    test_results = evaluate("MLP [test]", y_test, model.predict(X_test))

    return {"val": val_results, "test": test_results, "model_obj": model}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baselines(splits: dict, output_dir: str = "experiments") -> dict:
    """
    Run all baselines and save results to experiments/.

    Returns dict of all results for comparison with deep models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = {}

    lr_results  = run_logistic_regression(splits)
    mlp_results = run_mlp(splits)

    results["logistic_regression"] = {
        "val":  lr_results["val"],
        "test": lr_results["test"],
    }
    results["mlp"] = {
        "val":  mlp_results["val"],
        "test": mlp_results["test"],
    }

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path  = output_dir / f"baselines_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved → {out_path}")

    # Print comparison table
    logger.info("\n{'='*50}")
    logger.info("BASELINE COMPARISON (test set)")
    logger.info(f"{'Model':<25} {'Kappa':>8} {'MacroF1':>8} {'F1-down':>8} {'F1-up':>8}")
    logger.info("-" * 60)
    for name, res in results.items():
        t = res["test"]
        logger.info(
            f"{name:<25} {t['kappa']:>8.4f} {t['macro_f1']:>8.4f} "
            f"{t['f1_down']:>8.4f} {t['f1_up']:>8.4f}"
        )

    return results
