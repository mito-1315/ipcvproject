"""
evaluate.py — Evaluation, metrics, and report-artifact generation for Project Version Two.
Screenshot 7: Annotated bounding boxes with class labels and confidence scores.
Screenshot 8: Confusion matrix heatmap.
"""

import os
import time
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1 (macro), and inference time.

    Args:
        clf: Fitted classifier with predict().
        X_test: Test embedding array.
        y_test: True labels.

    Returns:
        Dict of metrics including 'y_pred'.
    """
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    inf_time = time.perf_counter() - t0

    # Mask unknown predictions (-1) as wrong for metric computation
    mask = y_pred != -1
    y_true_masked = y_test[mask]
    y_pred_masked = y_pred[mask]

    metrics = {
        "accuracy": accuracy_score(y_test, np.where(y_pred == -1, -999, y_pred)),
        "precision_macro": precision_score(y_true_masked, y_pred_masked,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_masked, y_pred_masked,
                                     average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_masked, y_pred_masked,
                             average="macro", zero_division=0),
        "inference_time_s": inf_time,
        "y_pred": y_pred,
    }
    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1_macro']:.4f} | "
        f"Inf: {inf_time*1000:.1f}ms"
    )
    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    num_classes: int = 40,
    dpi: int = 150,
) -> None:
    """
    Screenshot 8: 40×40 confusion matrix heatmap (seaborn.heatmap).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Destination PNG.
        num_classes: Number of classes.
        dpi: Output DPI.
    """
    labels = list(range(num_classes))
    y_pred_clean = np.where(y_pred == -1, -1, y_pred)
    cm = confusion_matrix(y_true, y_pred_clean, labels=labels)

    fig, ax = plt.subplots(figsize=(18, 15))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    sns.heatmap(
        cm, ax=ax, cmap="Blues",
        linewidths=0.3, linecolor="#313244",
        xticklabels=[f"s{i+1}" for i in labels],
        yticklabels=[f"s{i+1}" for i in labels],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Subject", color="#cdd6f4", fontsize=11)
    ax.set_ylabel("True Subject", color="#cdd6f4", fontsize=11)
    ax.set_title(
        "Screenshot 8: Confusion Matrix — 40-Class ArcFace Face Recognition",
        color="#cba6f7", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#cdd6f4", labelsize=7)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Confusion matrix → {output_path}")


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str
) -> None:
    """
    Write per-class classification report to .txt.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Destination .txt.
    """
    y_pred_clean = np.where(y_pred == -1, -1, y_pred)
    max_class = max(y_true.max(), y_pred_clean[y_pred_clean != -1].max() if (y_pred_clean != -1).any() else 0)
    target_names = [f"s{i+1}" for i in range(max_class + 1)]
    report = classification_report(
        y_true, y_pred_clean, target_names=target_names, zero_division=0
    )
    header = ("=" * 65 + "\n"
              "Classification Report — ArcFace + KNN (cosine) vs ORL Dataset\n"
              "=" * 65 + "\n\n")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(header + report)
    logger.info(f"Classification report → {output_path}")


def save_performance_metrics(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Save performance comparison CSV."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Performance metrics → {output_path}")


def save_before_after_tuning(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Save before/after tuning comparison CSV."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Before/after tuning → {output_path}")


def save_predictions_grid(
    test_images_bgr: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    max_show: int = 40,
    dpi: int = 150,
) -> None:
    """
    Screenshot 7: Annotated predictions grid with bounding boxes and labels.
    Green = correct, Red = incorrect.

    Args:
        test_images_bgr: BGR uint8 images for test subjects (N, H, W, 3).
        y_test: True labels.
        y_pred: Predicted labels.
        output_dir: Destination directory.
        max_show: Max images per grid.
        dpi: Output DPI.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _draw_grid(indices: np.ndarray, title: str, filename: str) -> None:
        n = min(len(indices), max_show)
        if n == 0:
            logger.warning(f"No images for {filename}")
            return
        ncols = 8
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.6))
        fig.patch.set_facecolor("#1e1e2e")
        fig.suptitle(title, color="#cba6f7", fontsize=13, y=1.01)
        axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

        for slot, ax in enumerate(axes_flat):
            ax.set_facecolor("#1e1e2e")
            if slot < n:
                idx = indices[slot]
                img_bgr = test_images_bgr[idx]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                correct = (y_test[idx] == y_pred[idx])
                ax.imshow(img_rgb)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#a6e3a1" if correct else "#f38ba8")
                    spine.set_linewidth(3)
                pred_label = y_pred[idx]
                pred_str = f"s{pred_label+1}" if pred_label != -1 else "unk"
                ax.set_title(
                    f"T:s{y_test[idx]+1}\nP:{pred_str}",
                    fontsize=6.5, color="#cdd6f4", pad=2,
                )
            ax.axis("off")

        plt.tight_layout()
        out = os.path.join(output_dir, filename)
        plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Prediction grid → {out}")

    all_idx = np.arange(len(y_test))
    correct_idx = np.where(y_test == y_pred)[0]
    incorrect_idx = np.where(y_test != y_pred)[0]

    _draw_grid(all_idx, "Screenshot 7: All Test Predictions (Green=Correct, Red=Incorrect)",
               "predictions_with_bboxes.png")
    _draw_grid(correct_idx, "Correct Predictions", "correct_predictions_grid.png")
    _draw_grid(incorrect_idx, "Incorrect Predictions", "incorrect_predictions_grid.png")


def save_annotated_predictions(
    test_images_bgr: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    output_path: str,
    max_show: int = 16,
    dpi: int = 150,
) -> None:
    """
    Screenshot 7 (detailed): Annotated grid with bounding boxes, labels, and confidence.

    Args:
        test_images_bgr: BGR uint8 test images (N, H, W, 3).
        y_test: True labels.
        y_pred: Predicted labels.
        confidences: Similarity scores for each prediction.
        output_path: Destination PNG.
        max_show: Max images to show.
        dpi: Output DPI.
    """
    n = min(len(y_test), max_show)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.5))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("Screenshot 7: Annotated Predictions — Bounding Boxes + Labels + Confidence",
                 color="#cba6f7", fontsize=12, y=1.01)
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for slot, ax in enumerate(axes_flat):
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        if slot < n:
            img_rgb = cv2.cvtColor(test_images_bgr[slot], cv2.COLOR_BGR2RGB)
            correct = (y_test[slot] == y_pred[slot])
            color = "#a6e3a1" if correct else "#f38ba8"

            ax.imshow(img_rgb)
            h, w = img_rgb.shape[:2]
            rect = mpatches.FancyBboxPatch(
                (0, 0), w - 1, h - 1,
                boxstyle="square,pad=0",
                linewidth=3, edgecolor=color, facecolor="none",
                transform=ax.transData,
            )
            ax.add_patch(rect)

            conf = confidences[slot] if confidences is not None else 0.0
            pred_str = f"s{y_pred[slot]+1}" if y_pred[slot] != -1 else "unk"
            ax.set_title(
                f"Pred: {pred_str} ({conf:.2f})\nTrue: s{y_test[slot]+1}",
                fontsize=8, color=color, pad=3,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Annotated predictions → {output_path}")
