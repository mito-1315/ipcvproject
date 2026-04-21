"""
embeddings.py — ArcFace embedding extraction via InsightFace.
Initialises the FaceAnalysis pipeline (RetinaFace detector + ArcFace recognizer),
extracts 512-dim embeddings for every ORL image, and caches them to disk.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def init_insightface(model_name: str = "buffalo_l", providers: list = None, det_size: Tuple = (160, 160)):
    """
    Initialise the InsightFace FaceAnalysis pipeline.

    Args:
        model_name: InsightFace model pack (default 'buffalo_l' — includes
                    RetinaFace detector + ArcFace recognizer).
        providers: ONNX execution providers (default CPU).
        det_size: Face detection input size.

    Returns:
        Initialised FaceAnalysis app object.
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError(
            "InsightFace is not installed. Run: pip install insightface"
        )

    if providers is None:
        providers = ["CPUExecutionProvider"]

    logger.info(f"Initialising InsightFace model '{model_name}' (may download on first run)…")
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    logger.info("InsightFace ready.")
    return app


def extract_embedding_single(
    app,
    bgr_img: np.ndarray,
    det_score_thresh: float = 0.3,
) -> Tuple[np.ndarray, object]:
    """
    Extract a 512-dim ArcFace embedding from a single BGR image.
    Falls back to direct recognition inference if detection fails.

    Args:
        app: Initialised FaceAnalysis object.
        bgr_img: BGR uint8 image (e.g., 160×160×3).
        det_score_thresh: Minimum detection confidence to use detected face.

    Returns:
        embedding: float32 array of shape (512,)
        face_obj: InsightFace face object or None (for landmarks, bbox access)
    """
    faces = app.get(bgr_img)

    # Use highest-confidence detected face above threshold
    valid = [f for f in faces if f.det_score >= det_score_thresh]
    if valid:
        best = max(valid, key=lambda f: f.det_score)
        return best.normed_embedding.astype(np.float32), best

    # Fallback: pass full image directly to recognition model
    logger.debug("No face detected; using direct recognition fallback.")
    try:
        rec_model = app.models.get("recognition") or list(app.models.values())[-1]
        emb = rec_model.get_feat(bgr_img).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32), None
    except Exception as e:
        logger.warning(f"Recognition fallback failed: {e}. Returning zeros.")
        return np.zeros(512, dtype=np.float32), None


def extract_all_embeddings(
    app,
    images: np.ndarray,
    labels: np.ndarray,
    cfg: Dict[str, Any],
    models_dir: str,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract ArcFace embeddings for all images and cache to disk.

    Args:
        app: Initialised FaceAnalysis object.
        images: uint8 BGR array of shape (N, H, W, 3).
        labels: int32 label array of shape (N,).
        cfg: insightface config dict with det_score_thresh.
        models_dir: Directory to cache .npy files.

    Returns:
        embeddings: float32 array of shape (N, 512)
        labels: same labels passed in
        face_objects: list of InsightFace face objects (or None) per image
    """
    emb_path = Path(models_dir) / "arcface_embeddings.npy"
    lbl_path = Path(models_dir) / "labels.npy"

    if emb_path.exists() and lbl_path.exists():
        logger.info("Loading cached embeddings from disk…")
        embeddings = np.load(str(emb_path))
        labels_cached = np.load(str(lbl_path))
        if len(embeddings) == len(images):
            logger.info(f"Loaded {len(embeddings)} cached embeddings.")
            return embeddings, labels_cached, [None] * len(images)
        logger.info("Cache size mismatch — re-extracting…")

    thresh = cfg.get("det_score_thresh", 0.3)
    embeddings = np.zeros((len(images), 512), dtype=np.float32)
    face_objects = []
    det_count = 0

    logger.info(f"Extracting embeddings for {len(images)} images…")
    for i, img in enumerate(images):
        emb, face_obj = extract_embedding_single(app, img, det_score_thresh=thresh)
        embeddings[i] = emb
        face_objects.append(face_obj)
        if face_obj is not None:
            det_count += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(images)} done | detected: {det_count}")

    logger.info(f"Embedding extraction complete. Faces detected: {det_count}/{len(images)}")

    os.makedirs(models_dir, exist_ok=True)
    np.save(str(emb_path), embeddings)
    np.save(str(lbl_path), labels)
    logger.info(f"Embeddings cached → {emb_path}")

    return embeddings, labels, face_objects
