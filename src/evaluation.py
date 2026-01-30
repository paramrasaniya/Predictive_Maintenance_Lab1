
# src/evaluation.py`
# Calculates model performance (RMSE, MAE, R²) and saves results to `experiments/results.csv`.
# Generates proof graphs: scatter + regression line and residual plots (and other diagnostics if enabled).
# Validates that our regression model meaningfully **predicts time-to-failure** from sensor signals.

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def save_scatter_with_line(X, y, y_pred, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure()
    plt.scatter(X, y)
    # sort for a clean line
    idx = np.argsort(X)
    plt.plot(X[idx], y_pred[idx])
    plt.title(title)
    plt.xlabel("Predictor (X)")
    plt.ylabel("Target (y)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_residual_plot(y_true, y_pred, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = y_true - y_pred

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
