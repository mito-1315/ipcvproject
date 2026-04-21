"""
data_loader.py — Dataset loading and exploration for Project Version Two.
Loads ORL PGM images, converts to BGR for InsightFace,
and generates all dataset exploration artifacts.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_dir: str,
    num_subjects: int = 40,
    images_per_subject: int = 10,
    resize_wh: Tuple[int, int] = (160, 160),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all PGM images from the ORL dataset.
    Converts grayscale → 3-channel BGR and resizes for InsightFace.

    Args:
        dataset_dir: Path to the Dataset/ folder (relative or absolute).
        num_subjects: Number of subjects (40).
        images_per_subject: Images per subject (10).
        resize_wh: (width, height) to resize to for InsightFace input.

    Returns:
        images: uint8 BGR array of shape (N, H, W, 3)
        labels: int32 array of shape (N,) — 0-indexed subject IDs
        paths: List of source file paths
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    images, labels, paths = [], [], []
    W, H = resize_wh

    for subj_idx in range(1, num_subjects + 1):
        subject_dir = dataset_path / f"s{subj_idx}"
        if not subject_dir.exists():
            logger.warning(f"Missing subject folder: {subject_dir}")
            continue
        for img_idx in range(1, images_per_subject + 1):
            img_path = subject_dir / f"{img_idx}.pgm"
            if not img_path.exists():
                logger.warning(f"Missing image: {img_path}")
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                logger.warning(f"Could not read: {img_path}")
                continue
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            bgr_resized = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
            images.append(bgr_resized)
            labels.append(subj_idx - 1)
            paths.append(str(img_path))

    images_arr = np.array(images, dtype=np.uint8)
    labels_arr = np.array(labels, dtype=np.int32)
    logger.info(f"Loaded {len(images_arr)} images | {num_subjects} subjects | shape {images_arr.shape}")
    return images_arr, labels_arr, paths


def load_images_original_size(
    dataset_dir: str,
    num_subjects: int = 40,
    images_per_subject: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images at their original 112×92 size (grayscale) for visualization.

    Returns:
        images: float32 array of shape (N, 112, 92) in [0,1]
        labels: int32 array of shape (N,)
    """
    dataset_path = Path(dataset_dir)
    images, labels = [], []
    for subj_idx in range(1, num_subjects + 1):
        subject_dir = dataset_path / f"s{subj_idx}"
        for img_idx in range(1, images_per_subject + 1):
            img_path = subject_dir / f"{img_idx}.pgm"
            if not img_path.exists():
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                images.append(gray.astype(np.float32) / 255.0)
                labels.append(subj_idx - 1)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def generate_folder_structure(dataset_dir: str, output_path: str) -> None:
    """
    Walk dataset directory and write a tree to a .txt file.

    Args:
        dataset_dir: Root Dataset/ path.
        output_path: Destination .txt file.
    """
    lines = []
    for root, dirs, files in os.walk(dataset_dir):
        dirs.sort(key=lambda d: int(d[1:]) if d.startswith("s") and d[1:].isdigit() else 0)
        depth = root.replace(str(dataset_dir), "").count(os.sep)
        indent = "│   " * depth + "├── "
        lines.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = "│   " * (depth + 1) + "├── "
        for f in sorted(files):
            lines.append(f"{sub_indent}{f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info(f"Folder structure → {output_path}")


def generate_folder_tree_image(structure_txt: str, output_path: str, dpi: int = 150) -> None:
    """
    Render the folder structure text as a stylized PNG image.

    Args:
        structure_txt: Path to folder_structure.txt.
        output_path: Destination PNG path.
        dpi: Output DPI.
    """
    with open(structure_txt, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    display_lines = [l.rstrip() for l in lines if l.strip().endswith("/")][:45]

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor("#1e1e2e")
    ax.axis("off")
    ax.set_facecolor("#1e1e2e")

    ax.text(
        0.02, 0.98, "\n".join(display_lines),
        transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#181825",
                  edgecolor="#89b4fa", linewidth=1.5),
        color="#cdd6f4",
    )
    plt.title("ORL/AT&T Dataset — Folder Structure", color="#cba6f7", fontsize=13, pad=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Folder tree image → {output_path}")


def generate_class_csv(dataset_dir: str, output_path: str) -> None:
    """
    Create Class/Subject_ID/Image_Count/Folder_Path CSV.

    Args:
        dataset_dir: Root Dataset/ path.
        output_path: Destination CSV.
    """
    rows = []
    for subj_idx in range(1, 41):
        subject_dir = Path(dataset_dir) / f"s{subj_idx}"
        count = len(list(subject_dir.glob("*.pgm"))) if subject_dir.exists() else 0
        rows.append({"Class": subj_idx, "Subject_ID": f"s{subj_idx}",
                     "Image_Count": count, "Folder_Path": str(subject_dir)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Class CSV → {output_path}")


def generate_sample_grid(
    dataset_dir: str,
    output_path: str,
    nrows: int = 8,
    ncols: int = 5,
    dpi: int = 150,
) -> None:
    """
    Create an 8×5 grid showing one sample image per class (40 subjects).

    Args:
        dataset_dir: Root Dataset/ path.
        output_path: Destination PNG.
        nrows: Grid rows.
        ncols: Grid columns.
        dpi: Output DPI.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.6))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("ORL/AT&T Dataset — Sample Faces per Subject",
                 fontsize=14, color="#cba6f7", y=1.01)

    subj = 1
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row][col]
            ax.set_facecolor("#1e1e2e")
            if subj <= 40:
                img_path = Path(dataset_dir) / f"s{subj}" / "1.pgm"
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    ax.imshow(gray, cmap="gray")
                ax.set_title(f"s{subj}", fontsize=8, color="#cdd6f4", pad=2)
            ax.axis("off")
            subj += 1

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Sample grid → {output_path}")
