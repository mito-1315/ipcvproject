"""
train.py — Training pipeline for Project Version Two.
Stratified split → fit KNN variants → save best model.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit

from src.classifier import build_knn, CosineThresholdMatcher

logger = logging.getLogger(__name__)


def split_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/test split on embedding arrays.

    Args:
        embeddings: float32 array (N, 512).
        labels: int32 array (N,).
        test_size: Fraction for test set.
        random_state: Seed.

    Returns:
        X_train, X_test, y_train, y_test
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(embeddings, labels))
    logger.info(
        f"Split: {len(train_idx)} train / {len(test_idx)} test "
        f"({len(np.unique(labels))} classes)"
    )
    return embeddings[train_idx], embeddings[test_idx], labels[train_idx], labels[test_idx]


def train_pipeline(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full training: split → KNN k=1/3/5 → cosine matcher → save models.

    Args:
        embeddings: float32 array (N, 512).
        labels: int32 array (N,).
        cfg: Full config dict.

    Returns:
        results dict with all fitted models, data splits, and timing.
    """
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    seed = cfg["random_seed"]
    knn_cfg = cfg["knn"]
    cm_cfg = cfg["cosine_matcher"]

    X_train, X_test, y_train, y_test = split_embeddings(
        embeddings, labels, cfg["split"]["test_size"], seed
    )

    timing = {}
    models = {}

    # KNN variants
    for k in knn_cfg["k_values"]:
        logger.info(f"Training KNN k={k} (cosine metric)…")
        clf = build_knn(k=k, metric=knn_cfg["metric"])
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        timing[f"knn_k{k}"] = time.perf_counter() - t0
        models[f"knn_k{k}"] = clf
        logger.info(f"  KNN k={k} trained in {timing[f'knn_k{k}']:.3f}s")

    # Cosine threshold matcher
    logger.info(f"Training CosineThresholdMatcher (threshold={cm_cfg['threshold']})…")
    matcher = CosineThresholdMatcher(threshold=cm_cfg["threshold"])
    t0 = time.perf_counter()
    matcher.fit(X_train, y_train)
    timing["cosine_matcher"] = time.perf_counter() - t0
    models["cosine_matcher"] = matcher

    # Save best classifier (KNN k=1 by default)
    best_clf = models["knn_k1"]
    clf_path = Path(models_dir) / "classifier.pkl"
    joblib.dump(best_clf, str(clf_path))
    logger.info(f"Best classifier (KNN k=1) saved → {clf_path}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "timing": timing,
    }
