"""
classifier.py — KNN (cosine) and Cosine-Threshold matchers for ArcFace embeddings.
"""

import logging
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def build_knn(k: int = 1, metric: str = "cosine") -> KNeighborsClassifier:
    """
    Construct a KNN classifier with cosine distance metric.

    Args:
        k: Number of neighbours.
        metric: Distance metric (default 'cosine' for ArcFace embeddings).

    Returns:
        Unfitted KNeighborsClassifier.
    """
    return KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)


class CosineThresholdMatcher:
    """
    1-NN cosine distance matcher with a configurable threshold.

    Classifies as 'unknown' if the nearest embedding cosine distance
    exceeds the threshold. Useful for open-set face verification.
    """

    def __init__(self, threshold: float = 0.4):
        """
        Args:
            threshold: Maximum cosine distance to accept a match.
        """
        self.threshold = threshold
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CosineThresholdMatcher":
        """Store training embeddings (L2-normalised)."""
        norms = np.linalg.norm(X_train, axis=1, keepdims=True)
        self.X_train = X_train / (norms + 1e-8)
        self.y_train = y_train
        logger.info(f"CosineThresholdMatcher fitted on {len(X_train)} samples.")
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels using nearest-neighbour cosine distance.

        Args:
            X_test: float array of shape (N, 512).

        Returns:
            Predicted labels array (same class as nearest train embedding,
            or -1 if above threshold).
        """
        norms = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_norm = X_test / (norms + 1e-8)
        # cosine similarity = dot product of unit vectors
        sims = X_norm @ self.X_train.T   # (N_test, N_train)
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(X_test)), best_idx]
        best_dist = 1.0 - best_sim       # cosine distance

        preds = self.y_train[best_idx].copy()
        preds[best_dist > self.threshold] = -1
        return preds

    def predict_with_confidence(self, X_test: np.ndarray):
        """Return (labels, confidence_scores)."""
        norms = np.linalg.norm(X_test, axis=1, keepdims=True)
        X_norm = X_test / (norms + 1e-8)
        sims = X_norm @ self.X_train.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(X_test)), best_idx]
        preds = self.y_train[best_idx].copy()
        return preds, best_sim.astype(np.float32)
