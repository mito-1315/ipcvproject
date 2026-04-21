"""
preprocessing.py — OpenCV image filters + extra IPCV outputs.
Applies 4 standard filters and generates additional report-ready
visualizations: CLAHE, Canny edge map, Fourier Transform, and
RetinaFace landmark overlay.
"""

import os
import logging
import textwrap
from typing import Tuple, Dict, Any

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Standard 4 Filters
# ────────────────────────────────────────────────────────────

def apply_histogram_eq(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply OpenCV histogram equalization to a grayscale image."""
    return gray, cv2.equalizeHist(gray)


def apply_gaussian(gray: np.ndarray, ksize: Tuple[int,int] = (5,5), sigma: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Gaussian blur to reduce sensor noise."""
    return gray, cv2.GaussianBlur(gray, ksize, sigma)


def apply_median(gray: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Apply median filter to remove salt-and-pepper noise."""
    return gray, cv2.medianBlur(gray, ksize)


def apply_sharpen(
    gray: np.ndarray,
    blur_ksize: Tuple[int,int] = (9,9),
    sigma: float = 10.0,
    alpha: float = 1.5,
    beta: float = -0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply unsharp masking (sharpening) to enhance facial edges."""
    blurred = cv2.GaussianBlur(gray, blur_ksize, sigma)
    sharpened = cv2.addWeighted(gray, alpha, blurred, beta, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return gray, sharpened


def save_before_after(
    before: np.ndarray,
    after: np.ndarray,
    filter_name: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Save a side-by-side before/after grayscale comparison PNG.

    Args:
        before: Original grayscale image (uint8).
        after: Filtered grayscale image (uint8).
        filter_name: Title label for the filter.
        output_path: Destination PNG path.
        dpi: Output DPI.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor("#1e1e2e")
    for ax, img, title in zip(axes, [before, after], ["Before", f"After — {filter_name}"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="#cdd6f4", fontsize=11)
        ax.axis("off")
        ax.set_facecolor("#1e1e2e")
    plt.suptitle(filter_name, color="#cba6f7", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Filter comparison → {output_path}")


def apply_all_filters_and_save(sample_gray: np.ndarray, output_dir: str, cfg: Dict[str,Any]) -> None:
    """
    Apply all 4 filters to a sample face and save before/after PNGs.

    Args:
        sample_gray: Grayscale uint8 face image.
        output_dir: Destination directory for filter PNGs.
        cfg: preprocessing section of config.yaml.
    """
    filters_map = {
        "Histogram Equalization": (
            apply_histogram_eq(sample_gray),
            "before_after_histogram_eq.png",
        ),
        "Gaussian Blur": (
            apply_gaussian(sample_gray, tuple(cfg["gaussian_ksize"]), cfg["gaussian_sigma"]),
            "before_after_gaussian.png",
        ),
        "Median Filter": (
            apply_median(sample_gray, cfg["median_ksize"]),
            "before_after_median.png",
        ),
        "Unsharp Masking (Sharpen)": (
            apply_sharpen(
                sample_gray,
                tuple(cfg["sharpen_blur_ksize"]),
                cfg["sharpen_sigma"],
                cfg["sharpen_alpha"],
                cfg["sharpen_beta"],
            ),
            "before_after_sharpen.png",
        ),
    }
    for name, ((before, after), fname) in filters_map.items():
        save_before_after(before, after, name, os.path.join(output_dir, fname))


def save_filter_explanations(output_path: str) -> None:
    """
    Write filter code snippets and rationale to a .txt file.

    Args:
        output_path: Destination .txt file.
    """
    content = textwrap.dedent("""\
    ============================================================
    Filter Explanations — Project Version Two (OpenCV)
    ORL/AT&T Face Recognition using ArcFace + InsightFace
    ============================================================

    1. HISTOGRAM EQUALIZATION  (cv2.equalizeHist)
    ──────────────────────────────────────────────
    Rationale:
      ORL images were captured under varying lighting across sessions.
      Histogram equalization stretches the intensity histogram to span
      0-255, normalising brightness differences between subjects.
      This reduces lighting-induced variability before ArcFace embedding.

    Code:
      gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
      eq   = cv2.equalizeHist(gray)

    ──────────────────────────────────────────────────────────────

    2. GAUSSIAN BLUR  (cv2.GaussianBlur, 5×5, sigma=0)
    ────────────────────────────────────────────────────
    Rationale:
      Gaussian noise from CCD sensors is attenuated by a low-pass
      Gaussian kernel. The 5×5 kernel removes fine noise while keeping
      macro facial structure (shape, shadow gradients) for ArcFace's
      convolutional layers intact.

    Code:
      blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ──────────────────────────────────────────────────────────────

    3. MEDIAN FILTER  (cv2.medianBlur, kernel=3)
    ─────────────────────────────────────────────
    Rationale:
      Median filtering removes salt-and-pepper (impulse) noise robustly
      because the median is insensitive to extreme outliers. Unlike mean
      filters it preserves sharp facial feature edges (eye corners, lip
      lines) critical for ArcFace's edge-sensitive early layers.

    Code:
      med = cv2.medianBlur(gray, 3)

    ──────────────────────────────────────────────────────────────

    4. UNSHARP MASKING / SHARPENING  (cv2.addWeighted)
    ────────────────────────────────────────────────────
    Rationale:
      Boosting high-frequency edges enhances facial landmarks (eyelid
      folds, nasal ridges) that contribute most to ArcFace's 512-dim
      embedding. Subtracting a heavy Gaussian blur from the original
      creates a "residual edge map" added back with gain 1.5.

    Code:
      blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)
      sharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    ──────────────────────────────────────────────────────────────
    """)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info(f"Filter explanations → {output_path}")


# ────────────────────────────────────────────────────────────
# Extra IPCV Outputs (Screenshots 1–6)
# ────────────────────────────────────────────────────────────

def save_clahe_comparison(gray: np.ndarray, output_path: str, cfg: Dict[str,Any], dpi: int = 150) -> None:
    """
    Screenshot 1 & 2: Raw input vs CLAHE contrast-enhanced comparison.

    Args:
        gray: Original grayscale face (uint8).
        output_path: Destination PNG.
        cfg: preprocessing config dict.
        dpi: Output DPI.
    """
    clip = cfg.get("clahe_clip_limit", 2.0)
    tile = tuple(cfg.get("clahe_tile_grid", [8, 8]))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    enhanced = clahe.apply(gray)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    for ax, img, title in zip(
        axes,
        [gray, enhanced],
        ["Screenshot 1: Raw Input", "Screenshot 2: After CLAHE Enhancement"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="#cdd6f4", fontsize=10)
        ax.axis("off")
        ax.set_facecolor("#1e1e2e")
    plt.suptitle("CLAHE Contrast Enhancement — Side by Side", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"CLAHE comparison → {output_path}")

    return enhanced  # return for histogram step


def save_canny_edge_map(gray: np.ndarray, output_path: str, cfg: Dict[str,Any], dpi: int = 150) -> None:
    """
    Screenshot 3: Canny edge detection map showing facial structure boundaries.

    Args:
        gray: Grayscale face (uint8).
        output_path: Destination PNG.
        cfg: canny config with threshold1, threshold2.
        dpi: Output DPI.
    """
    t1 = cfg.get("threshold1", 50)
    t2 = cfg.get("threshold2", 150)
    edges = cv2.Canny(gray, t1, t2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax1.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax1.set_title("Screenshot 3a: Original", color="#cdd6f4", fontsize=10)
    ax1.axis("off"); ax1.set_facecolor("#1e1e2e")
    ax2.imshow(edges, cmap="hot")
    ax2.set_title(f"Screenshot 3b: Canny Edge Map\n(t1={t1}, t2={t2})", color="#cdd6f4", fontsize=10)
    ax2.axis("off"); ax2.set_facecolor("#1e1e2e")
    plt.suptitle("Canny Edge Detection — Facial Feature Boundaries", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Canny edge map → {output_path}")


def save_histogram_clahe_comparison(
    gray: np.ndarray, output_path: str, cfg: Dict[str,Any], dpi: int = 150
) -> None:
    """
    Screenshot 4: Pixel intensity histogram before and after CLAHE.

    Args:
        gray: Original grayscale image (uint8).
        output_path: Destination PNG.
        cfg: preprocessing config.
        dpi: Output DPI.
    """
    clip = cfg.get("clahe_clip_limit", 2.0)
    tile = tuple(cfg.get("clahe_tile_grid", [8, 8]))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    enhanced = clahe.apply(gray)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    for ax, img, title, color in zip(
        [ax1, ax2],
        [gray, enhanced],
        ["Before CLAHE", "After CLAHE"],
        ["#89b4fa", "#a6e3a1"],
    ):
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        ax.plot(hist, color=color, linewidth=1.5)
        ax.fill_between(range(256), hist, alpha=0.3, color=color)
        ax.set_title(f"Screenshot 4 — {title}", color="#cdd6f4", fontsize=10)
        ax.set_xlabel("Pixel Intensity", color="#cdd6f4")
        ax.set_ylabel("Frequency", color="#cdd6f4")
        ax.set_facecolor("#181825")
        ax.tick_params(colors="#cdd6f4")
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        ax.grid(alpha=0.2, color="#45475a")

    plt.suptitle("Pixel Intensity Histogram Before vs After CLAHE", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Histogram CLAHE comparison → {output_path}")


def save_fourier_spectrum(gray: np.ndarray, output_path: str, dpi: int = 150) -> None:
    """
    Screenshot 5: Fourier Transform magnitude spectrum of a face image.

    Args:
        gray: Grayscale face (uint8).
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    gray_float = np.float32(gray)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft[:, :, 0] + 1j * dft[:, :, 1])
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Screenshot 5a: Original Face", color="#cdd6f4", fontsize=10)
    ax1.axis("off"); ax1.set_facecolor("#1e1e2e")

    ax2.imshow(magnitude, cmap="inferno")
    ax2.set_title("Screenshot 5b: Fourier Magnitude Spectrum", color="#cdd6f4", fontsize=10)
    ax2.axis("off"); ax2.set_facecolor("#1e1e2e")

    plt.suptitle("Fourier Transform — Frequency Domain Analysis", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Fourier spectrum → {output_path}")


def save_retinaface_landmarks(
    bgr_img: np.ndarray,
    face_info: Dict,
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    Screenshot 6: RetinaFace detection output with 5 facial landmarks.

    Args:
        bgr_img: BGR uint8 image (resized to 160x160).
        face_info: InsightFace face object with .kps (5 keypoints) and .bbox.
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    annotated = bgr_img.copy()
    landmark_names = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
    landmark_colors = [
        (255, 128, 0), (0, 128, 255), (0, 255, 128),
        (255, 0, 128), (128, 0, 255),
    ]

    if face_info is not None and hasattr(face_info, 'kps') and face_info.kps is not None:
        for (x, y), name, color in zip(face_info.kps.astype(int), landmark_names, landmark_colors):
            cv2.circle(annotated, (x, y), 4, color, -1)
            cv2.putText(annotated, name, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        bbox = face_info.bbox.astype(int)
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 100), 2)

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax1.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Screenshot 6a: Input Face", color="#cdd6f4", fontsize=10)
    ax1.axis("off"); ax1.set_facecolor("#1e1e2e")
    ax2.imshow(rgb)
    ax2.set_title("Screenshot 6b: RetinaFace — 5 Landmarks", color="#cdd6f4", fontsize=10)
    ax2.axis("off"); ax2.set_facecolor("#1e1e2e")

    plt.suptitle("RetinaFace Detection with 5 Facial Landmarks", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"RetinaFace landmarks → {output_path}")


def save_retinaface_no_detection_fallback(
    bgr_img: np.ndarray, output_path: str, dpi: int = 150
) -> None:
    """
    Fallback for Screenshot 6 when no face is detected: show image border as bbox.

    Args:
        bgr_img: BGR uint8 image.
        output_path: Destination PNG.
        dpi: Output DPI.
    """
    annotated = bgr_img.copy()
    h, w = annotated.shape[:2]
    cv2.rectangle(annotated, (3, 3), (w - 3, h - 3), (0, 255, 100), 2)
    landmark_positions = [
        (w // 3, h // 3), (2 * w // 3, h // 3),
        (w // 2, h // 2),
        (w // 3, 2 * h // 3), (2 * w // 3, 2 * h // 3),
    ]
    for pos in landmark_positions:
        cv2.circle(annotated, pos, 4, (255, 165, 0), -1)

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax1.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Screenshot 6a: Input Face (resized)", color="#cdd6f4", fontsize=10)
    ax1.axis("off"); ax1.set_facecolor("#1e1e2e")
    ax2.imshow(rgb)
    ax2.set_title("Screenshot 6b: Estimated Landmark Overlay\n(fallback — detection below threshold)", color="#cdd6f4", fontsize=9)
    ax2.axis("off"); ax2.set_facecolor("#1e1e2e")
    plt.suptitle("RetinaFace Landmark Detection Output", color="#cba6f7", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"RetinaFace fallback landmarks → {output_path}")
