"""
predict.py — Single-image inference CLI for Project Version Two.
Loads saved ArcFace embeddings + KNN classifier, predicts subject,
draws bounding box with label and confidence score.

Usage:
    python src/predict.py --image ../Dataset/s5/8.pgm
    python src/predict.py --image ../Dataset/s10/3.pgm --output outputs/05_results/my_demo.png
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def load_models_and_embeddings(models_dir: str):
    """
    Load the trained KNN classifier and cached training embeddings.

    Args:
        models_dir: Path to models/ directory.

    Returns:
        clf: Trained KNN classifier.
        train_embeddings: float32 array (N_train, 512) — for confidence estimation.
        train_labels: int32 label array (N_train,).
    """
    clf_path = Path(models_dir) / "classifier.pkl"
    emb_path = Path(models_dir) / "arcface_embeddings.npy"
    lbl_path = Path(models_dir) / "labels.npy"

    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier not found at {clf_path}. Run main.py first.")

    clf = joblib.load(str(clf_path))
    embeddings = np.load(str(emb_path)) if emb_path.exists() else None
    labels = np.load(str(lbl_path)) if lbl_path.exists() else None
    logger.info("Classifier and embeddings loaded.")
    return clf, embeddings, labels


def preprocess_and_embed(image_path: str, cfg: dict):
    """
    Load image, convert to BGR, resize, extract ArcFace embedding.

    Args:
        image_path: Path to input image.
        cfg: Full config dict.

    Returns:
        embedding: float32 (512,)
        bgr_img: BGR uint8 image used for embedding
        face_obj: InsightFace face object or None
    """
    from src.embeddings import init_insightface, extract_embedding_single

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    W = cfg["dataset"]["resize_width"]
    H = cfg["dataset"]["resize_height"]
    bgr_resized = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    ins_cfg = cfg["insightface"]
    app = init_insightface(
        model_name=ins_cfg["model_name"],
        providers=ins_cfg["providers"],
        det_size=tuple(ins_cfg["det_size"]),
    )
    embedding, face_obj = extract_embedding_single(
        app, bgr_resized, det_score_thresh=ins_cfg["det_score_thresh"]
    )
    return embedding, bgr_resized, face_obj


def predict_subject(
    clf,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    embedding: np.ndarray,
) -> tuple:
    """
    Predict subject class and compute cosine similarity confidence.

    Args:
        clf: Fitted KNN classifier.
        train_embeddings: Training embedding matrix.
        train_labels: Training labels.
        embedding: Query embedding (512,).

    Returns:
        pred_label: int (0-indexed)
        confidence: float cosine similarity to nearest training example
    """
    t0 = time.perf_counter()
    pred_label = int(clf.predict(embedding.reshape(1, -1))[0])
    inf_time = time.perf_counter() - t0

    confidence = 0.0
    if train_embeddings is not None:
        norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
        X_norm = train_embeddings / (norms + 1e-8)
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        sims = X_norm @ emb_norm
        confidence = float(sims.max())

    logger.info(
        f"Predicted: s{pred_label+1} | Confidence: {confidence:.4f} | "
        f"Inference: {inf_time*1000:.1f}ms"
    )
    return pred_label, confidence


def annotate_and_save(
    bgr_img: np.ndarray,
    pred_label: int,
    confidence: float,
    face_obj,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Draw bounding box + label on the image and save PNG.

    Args:
        bgr_img: BGR uint8 face image.
        pred_label: 0-indexed predicted subject.
        confidence: Cosine similarity score.
        face_obj: InsightFace face object (for bbox) or None.
        output_path: Destination PNG path.
        dpi: Output DPI.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(5, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.imshow(rgb)

    # Use face bbox if available, else full image border
    if face_obj is not None and hasattr(face_obj, "bbox"):
        bbox = face_obj.bbox.astype(int)
        bx, by, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
        rect = mpatches.FancyBboxPatch(
            (bx, by), bx2 - bx, by2 - by,
            boxstyle="square,pad=0", linewidth=3,
            edgecolor="#a6e3a1", facecolor="none",
        )
    else:
        rect = mpatches.FancyBboxPatch(
            (2, 2), w - 4, h - 4,
            boxstyle="square,pad=0", linewidth=3,
            edgecolor="#a6e3a1", facecolor="none",
        )
    ax.add_patch(rect)

    label_str = f"Predicted: s{pred_label+1}  (conf: {confidence:.2f})"
    ax.text(
        w / 2, h - 4, label_str,
        color="white", fontsize=10, ha="center", va="bottom",
        bbox=dict(facecolor="#313244", edgecolor="#a6e3a1", boxstyle="round,pad=0.3"),
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    plt.title("ArcFace Inference Demo", color="#cba6f7", fontsize=12, pad=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Inference demo saved → {output_path}")


def run_inference(
    image_path: str,
    output_path: str,
    config_path: str = "config.yaml",
) -> tuple:
    """
    Full inference pipeline: load → embed → predict → annotate → save.

    Args:
        image_path: Path to input image.
        output_path: Where to save annotated PNG.
        config_path: YAML config path.

    Returns:
        (pred_label, confidence)
    """
    cfg = load_config(config_path)
    clf, train_embs, train_lbls = load_models_and_embeddings(cfg["paths"]["models_dir"])
    embedding, bgr_img, face_obj = preprocess_and_embed(image_path, cfg)
    pred_label, confidence = predict_subject(clf, train_embs, train_lbls, embedding)
    annotate_and_save(bgr_img, pred_label, confidence, face_obj, output_path)

    print(f"\n{'─'*50}")
    print(f"  Input       : {image_path}")
    print(f"  Predicted   : Subject s{pred_label+1}")
    print(f"  Confidence  : {confidence*100:.2f}%")
    print(f"  Saved to    : {output_path}")
    print(f"{'─'*50}\n")
    return pred_label, confidence


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="ArcFace Face Recognition — Inference")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output", default="outputs/05_results/inference_demo_single.png",
                        help="Output annotated image path.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    try:
        run_inference(args.image, args.output, args.config)
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
