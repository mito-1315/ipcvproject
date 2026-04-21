"""
visualization.py — Feature visualizations for Project Version Two.
Generates t-SNE, PCA 2D projection, embedding histogram, HOG visualization,
pixel intensity histogram, and pipeline diagram.
"""

import os
import logging
from typing import Tuple, List, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import pandas as pd

logger = logging.getLogger(__name__)


def save_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    dpi: int = 150,
) -> None:
    """
    2D t-SNE projection of 512-dim ArcFace embeddings, colored by subject class.

    Args:
        embeddings: float32 array (N, 512).
        labels: int32 array (N,).
        output_path: Destination PNG.
        perplexity: t-SNE perplexity parameter.
        n_iter: t-SNE iterations.
        random_state: Seed.
        dpi: Output DPI.
    """
    from sklearn.manifold import TSNE
    logger.info("Running t-SNE (this may take ~30s)…")
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                    random_state=random_state)
    except TypeError:
        # sklearn < 1.5 uses n_iter
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=random_state)

    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    cmap = plt.cm.get_cmap("tab20", 40)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap=cmap, s=30, alpha=0.85,
        linewidths=0.3, edgecolors="white",
    )
    cb = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cb.set_label("Subject (0-indexed)", color="#cdd6f4")
    cb.ax.yaxis.set_tick_params(color="#cdd6f4")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#cdd6f4")

    ax.set_xlabel("t-SNE Dim 1", color="#cdd6f4")
    ax.set_ylabel("t-SNE Dim 2", color="#cdd6f4")
    ax.set_title("t-SNE Projection of ArcFace 512-D Embeddings (40 Subjects)",
                 color="#cba6f7", fontsize=13)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(alpha=0.15, color="#45475a")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"t-SNE plot → {output_path}")


def save_pca2d_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    random_state: int = 42,
    dpi: int = 150,
) -> None:
    """
    2D PCA projection of ArcFace embeddings, colored by subject class.

    Args:
        embeddings: float32 array (N, 512).
        labels: int32 array (N,).
        output_path: Destination PNG.
        random_state: Seed.
        dpi: Output DPI.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(embeddings)
    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")

    cmap = plt.cm.get_cmap("tab20", 40)
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap=cmap, s=30, alpha=0.85,
        linewidths=0.3, edgecolors="white",
    )
    cb = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cb.set_label("Subject (0-indexed)", color="#cdd6f4")
    cb.ax.yaxis.set_tick_params(color="#cdd6f4")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#cdd6f4")

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", color="#cdd6f4")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", color="#cdd6f4")
    ax.set_title("PCA 2D Projection of ArcFace Embeddings (40 Subjects)",
                 color="#cba6f7", fontsize=13)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(alpha=0.15, color="#45475a")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"PCA 2D plot → {output_path}")


def save_embedding_histogram(
    embeddings: np.ndarray, output_path: str, dpi: int = 150
) -> None:
    """
    Histogram of all ArcFace embedding values across the dataset.

    Args:
        embeddings: float32 array (N, 512).
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    values = embeddings.flatten()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")
    ax.hist(values, bins=120, color="#cba6f7", alpha=0.85, edgecolor="#1e1e2e")
    ax.set_xlabel("Embedding Value", color="#cdd6f4")
    ax.set_ylabel("Frequency", color="#cdd6f4")
    ax.set_title(
        f"ArcFace Embedding Value Distribution\n({len(embeddings)} images × 512 dims)",
        color="#cba6f7", fontsize=13,
    )
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(alpha=0.2, color="#45475a")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Embedding histogram → {output_path}")


def save_hog_visualization(
    sample_gray: np.ndarray,
    output_path: str,
    pixels_per_cell: Tuple[int,int] = (8, 8),
    cells_per_block: Tuple[int,int] = (2, 2),
    orientations: int = 9,
    dpi: int = 150,
) -> None:
    """
    HOG feature visualization on a sample face (for comparison discussion).

    Args:
        sample_gray: Grayscale uint8 face image.
        output_path: Destination PNG.
        pixels_per_cell: HOG cell size.
        cells_per_block: HOG block size.
        orientations: Gradient orientation bins.
        dpi: Output DPI.
    """
    from skimage.feature import hog
    from skimage import exposure as skexp

    img_float = sample_gray.astype(np.float32) / 255.0
    fd, hog_img = hog(
        img_float,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        orientations=orientations,
        visualize=True,
    )
    hog_img_rescaled = skexp.rescale_intensity(hog_img, in_range=(0, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax1.imshow(sample_gray, cmap="gray")
    ax1.set_title("Original Face", color="#cdd6f4", fontsize=12)
    ax1.axis("off"); ax1.set_facecolor("#1e1e2e")
    ax2.imshow(hog_img_rescaled, cmap="inferno")
    ax2.set_title(f"HOG Features\n(classical — dim={len(fd)})", color="#cdd6f4", fontsize=12)
    ax2.axis("off"); ax2.set_facecolor("#1e1e2e")
    plt.suptitle("HOG (Comparison) vs ArcFace 512-D Embeddings", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"HOG visualization → {output_path}")


def save_pixel_histogram(images_flat: np.ndarray, output_path: str, dpi: int = 150) -> None:
    """
    Pixel intensity distribution across the whole dataset.

    Args:
        images_flat: float32 array (N, D) in [0,1] or uint8 (N, H, W).
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    pixels = images_flat.flatten()
    if pixels.max() > 1.0:
        pixels = pixels / 255.0

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")
    ax.hist(pixels, bins=100, color="#89b4fa", alpha=0.85, edgecolor="#1e1e2e")
    ax.set_xlabel("Pixel Intensity (normalised)", color="#cdd6f4")
    ax.set_ylabel("Frequency", color="#cdd6f4")
    ax.set_title(f"Pixel Intensity Distribution — ORL Dataset",
                 color="#cba6f7", fontsize=13)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#45475a")
    ax.grid(alpha=0.2, color="#45475a")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Pixel histogram → {output_path}")


def save_feature_comparison_table(rows: list, output_path: str) -> None:
    """Save feature method comparison CSV."""
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Feature comparison table → {output_path}")


def generate_pipeline_diagram(output_path: str, dpi: int = 150) -> None:
    """
    Auto-generate a 7-node pipeline diagram for ArcFace face recognition.

    Nodes: Dataset → Load → Preprocess → ArcFace Embedding → Classifier → Evaluation → Prediction

    Args:
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.set_xlim(0, 18); ax.set_ylim(0, 6); ax.axis("off")

    nodes = [
        ("[DATASET]\n400 images\n40 subjects", 1.5, 3.0, "#313244", "#89b4fa"),
        ("[LOAD]\nExplore", 3.8, 3.0, "#313244", "#cba6f7"),
        ("[PREPROCESS]\nHist EQ, Blur,\nMedian, CLAHE", 6.3, 3.0, "#313244", "#f9e2af"),
        ("[ARCFACE]\nEmbedding\n512-dim", 9.0, 3.0, "#1a1a2e", "#f38ba8"),
        ("[KNN]\nClassifier\ncosine dist", 11.7, 3.0, "#313244", "#fab387"),
        ("[EVALUATE]\nMetrics &\nVisualizations", 14.4, 3.0, "#313244", "#89dceb"),
        ("[PREDICT]\nInference\nDemo", 16.8, 3.0, "#313244", "#a6e3a1"),
    ]

    box_w, box_h = 2.1, 1.9
    for label, cx, cy, facecolor, edgecolor in nodes:
        fancy = mpatches.FancyBboxPatch(
            (cx - box_w/2, cy - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.12",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2.0, zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=8.5, color="#cdd6f4", fontweight="bold",
                zorder=4, multialignment="center")

    arrow_kwargs = dict(arrowstyle="-|>", color="#585b70", lw=2.0,
                        mutation_scale=18, zorder=2)
    for i in range(len(nodes) - 1):
        x_start = nodes[i][1] + box_w/2
        x_end = nodes[i+1][1] - box_w/2
        y = nodes[i][2]
        ax.add_patch(FancyArrowPatch((x_start, y), (x_end, y), **arrow_kwargs))

    outputs = [
        "folder_structure.txt\nsamples_per_class.png",
        "filter PNGs\nCLAHE, Canny, FFT",
        "arcface_embeddings.npy\n512-dim per image",
        "classifier.pkl\nKNN cosine",
        "confusion_matrix.png\nperformance_metrics.csv",
        "inference_demo_single.png",
    ]
    for idx, (ox, out) in enumerate(zip([n[1] for n in nodes[1:]], outputs)):
        ax.annotate(out, xy=(ox, nodes[0][2] - box_h/2), xytext=(ox, 0.75),
                    fontsize=6.5, color="#6c7086", ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color="#45475a", lw=0.8),
                    multialignment="center")

    plt.title("Face Recognition v2 Pipeline — ArcFace + InsightFace",
              color="#cba6f7", fontsize=14, pad=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Pipeline diagram → {output_path}")
