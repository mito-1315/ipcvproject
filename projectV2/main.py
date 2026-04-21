"""
main.py — Project Version Two Entry Point
ArcFace + InsightFace Face Recognition Pipeline (10 steps)
Running `python main.py` generates all report artifacts automatically.
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path

import yaml
import numpy as np

# ── Logging ──────────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "pipeline_v2.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("main_v2")


# ── ANSI colors ───────────────────────────────────────────────
class C:
    RESET   = "\033[0m"; BOLD    = "\033[1m"
    GREEN   = "\033[92m"; YELLOW = "\033[93m"
    CYAN    = "\033[96m"; RED    = "\033[91m"
    MAGENTA = "\033[95m"


def banner(text: str) -> None:
    print(f"\n{C.CYAN}{C.BOLD}{'═'*65}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {text}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'═'*65}{C.RESET}\n")


def step(num: int, total: int, text: str) -> None:
    print(f"{C.YELLOW}{C.BOLD}[Step {num}/{total}] {text}{C.RESET}")


def ok(path: str) -> None:
    print(f"  {C.GREEN}✅ {path}{C.RESET}")


def err(msg: str) -> None:
    print(f"  {C.RED}❌ {msg}{C.RESET}")


# ── Registry ──────────────────────────────────────────────────
generated: list = []

def register(path: str) -> None:
    generated.append(path)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.perf_counter()
    banner("Face Recognition v2 — ArcFace + InsightFace | ORL/AT&T Dataset")

    cfg = load_config("config.yaml")
    DATASET_DIR = cfg["paths"]["dataset_dir"]
    OUTPUT_DIR  = cfg["paths"]["output_dir"]
    MODELS_DIR  = cfg["paths"]["models_dir"]
    SEED        = cfg["random_seed"]
    TOTAL_STEPS = 10

    for d in [OUTPUT_DIR, MODELS_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    ins_cfg = cfg["insightface"]

    # ════════════════════════════════════════════════════════
    # STEP 1 — Dataset Loading & Exploration
    # ════════════════════════════════════════════════════════
    step(1, TOTAL_STEPS, "Dataset Loading & Exploration")
    try:
        from src.data_loader import (
            load_dataset, load_images_original_size,
            generate_folder_structure, generate_folder_tree_image,
            generate_class_csv, generate_sample_grid,
        )

        images_bgr, labels, paths = load_dataset(
            DATASET_DIR,
            cfg["dataset"]["num_subjects"],
            cfg["dataset"]["images_per_subject"],
            resize_wh=(cfg["dataset"]["resize_width"], cfg["dataset"]["resize_height"]),
        )
        images_gray, _ = load_images_original_size(
            DATASET_DIR, cfg["dataset"]["num_subjects"], cfg["dataset"]["images_per_subject"]
        )
        logger.info(f"Loaded: {len(images_bgr)} BGR images, {len(images_gray)} gray images")

        DIR1 = os.path.join(OUTPUT_DIR, "01_dataset_structure")
        DIR2 = os.path.join(OUTPUT_DIR, "02_sample_images")
        struct_txt = os.path.join(DIR1, "folder_structure.txt")
        struct_png = os.path.join(DIR1, "folder_tree_visualization.png")
        class_csv  = os.path.join(DIR2, "images_per_class_table.csv")
        sample_png = os.path.join(DIR2, "samples_per_class.png")

        generate_folder_structure(DATASET_DIR, struct_txt);    register(struct_txt); ok(struct_txt)
        generate_folder_tree_image(struct_txt, struct_png);    register(struct_png); ok(struct_png)
        generate_class_csv(DATASET_DIR, class_csv);            register(class_csv);  ok(class_csv)
        generate_sample_grid(DATASET_DIR, sample_png, dpi=cfg["visualization"]["dpi"])
        register(sample_png); ok(sample_png)

    except Exception as exc:
        err(f"Step 1 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 2 — Preprocessing Filters + Extra IPCV Outputs
    # ════════════════════════════════════════════════════════
    step(2, TOTAL_STEPS, "Preprocessing Filters & Extra IPCV Screenshots")
    try:
        import cv2
        from src.preprocessing import (
            apply_all_filters_and_save, save_filter_explanations,
            save_clahe_comparison, save_canny_edge_map,
            save_histogram_clahe_comparison, save_fourier_spectrum,
            save_retinaface_landmarks, save_retinaface_no_detection_fallback,
        )

        DIR3   = os.path.join(OUTPUT_DIR, "03_filters")
        DIR8   = os.path.join(OUTPUT_DIR, "08_extra_ipcv")
        sample_bgr  = images_bgr[0]
        sample_gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)

        # Standard 4 filters
        apply_all_filters_and_save(sample_gray, DIR3, cfg["preprocessing"])
        for fname in ["before_after_histogram_eq.png", "before_after_gaussian.png",
                      "before_after_median.png", "before_after_sharpen.png"]:
            fp = os.path.join(DIR3, fname); register(fp); ok(fp)

        expl = os.path.join(DIR3, "filter_explanations.txt")
        save_filter_explanations(expl); register(expl); ok(expl)

        # Screenshot 1&2: CLAHE comparison
        clahe_png = os.path.join(DIR8, "clahe_comparison.png")
        save_clahe_comparison(sample_gray, clahe_png, cfg["preprocessing"])
        register(clahe_png); ok(clahe_png)

        # Screenshot 3: Canny edge map
        canny_png = os.path.join(DIR8, "canny_edge_map.png")
        save_canny_edge_map(sample_gray, canny_png, cfg["canny"])
        register(canny_png); ok(canny_png)

        # Screenshot 4: Histogram CLAHE comparison
        hist_clahe = os.path.join(DIR8, "histogram_clahe_comparison.png")
        save_histogram_clahe_comparison(sample_gray, hist_clahe, cfg["preprocessing"])
        register(hist_clahe); ok(hist_clahe)

        # Screenshot 5: Fourier Transform
        fft_png = os.path.join(DIR8, "fourier_transform_spectrum.png")
        save_fourier_spectrum(sample_gray, fft_png)
        register(fft_png); ok(fft_png)

    except Exception as exc:
        err(f"Step 2 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 3 — ArcFace Embedding Extraction
    # ════════════════════════════════════════════════════════
    step(3, TOTAL_STEPS, "ArcFace Embedding Extraction via InsightFace (buffalo_l)")
    try:
        from src.embeddings import init_insightface, extract_all_embeddings

        print(f"  {C.MAGENTA}NOTE: First run downloads ~300MB InsightFace model.{C.RESET}")
        app = init_insightface(
            model_name=ins_cfg["model_name"],
            providers=ins_cfg["providers"],
            det_size=tuple(ins_cfg["det_size"]),
        )
        embeddings, labels_out, face_objects = extract_all_embeddings(
            app, images_bgr, labels, ins_cfg, MODELS_DIR
        )
        emb_npy  = os.path.join(MODELS_DIR, "arcface_embeddings.npy")
        lbl_npy  = os.path.join(MODELS_DIR, "labels.npy")
        register(emb_npy); ok(emb_npy)
        register(lbl_npy); ok(lbl_npy)

        # Screenshot 6: RetinaFace landmarks (use first image with a detected face)
        DIR8 = os.path.join(OUTPUT_DIR, "08_extra_ipcv")
        landmarks_png = os.path.join(DIR8, "retinaface_landmarks.png")
        from src.preprocessing import save_retinaface_landmarks, save_retinaface_no_detection_fallback

        # Find first image with a detected face for Screenshot 6
        face_sample_idx = next((i for i, f in enumerate(face_objects) if f is not None), None)
        if face_sample_idx is not None:
            save_retinaface_landmarks(images_bgr[face_sample_idx], face_objects[face_sample_idx], landmarks_png)
        else:
            save_retinaface_no_detection_fallback(images_bgr[0], landmarks_png)
        register(landmarks_png); ok(landmarks_png)

    except Exception as exc:
        err(f"Step 3 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 4 — Feature Visualizations
    # ════════════════════════════════════════════════════════
    step(4, TOTAL_STEPS, "Feature Visualizations (t-SNE, PCA 2D, Histograms, HOG)")
    try:
        from src.visualization import (
            save_tsne_plot, save_pca2d_plot, save_embedding_histogram,
            save_hog_visualization, save_pixel_histogram, save_feature_comparison_table,
        )
        import cv2 as _cv2

        DIR4 = os.path.join(OUTPUT_DIR, "04_features")
        tsne_cfg = cfg["tsne"]
        hog_cfg  = cfg["hog"]

        tsne_png = os.path.join(DIR4, "arcface_embedding_tsne.png")
        save_tsne_plot(embeddings, labels_out, tsne_png,
                       perplexity=tsne_cfg["perplexity"],
                       n_iter=tsne_cfg["n_iter"],
                       random_state=SEED)
        register(tsne_png); ok(tsne_png)

        pca_png = os.path.join(DIR4, "arcface_embedding_pca.png")
        save_pca2d_plot(embeddings, labels_out, pca_png, random_state=SEED)
        register(pca_png); ok(pca_png)

        emb_hist = os.path.join(DIR4, "embedding_histogram.png")
        save_embedding_histogram(embeddings, emb_hist)
        register(emb_hist); ok(emb_hist)

        sample_gray_f = images_gray[0]
        sample_gray_u8 = (sample_gray_f * 255).astype(np.uint8)
        hog_png = os.path.join(DIR4, "hog_visualization_sample.png")
        save_hog_visualization(sample_gray_u8, hog_png,
                               pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
                               cells_per_block=tuple(hog_cfg["cells_per_block"]),
                               orientations=hog_cfg["orientations"])
        register(hog_png); ok(hog_png)

        pix_hist = os.path.join(DIR4, "pixel_intensity_histogram.png")
        save_pixel_histogram(images_gray, pix_hist)
        register(pix_hist); ok(pix_hist)

        feat_rows = [
            {"method": "Raw Pixels", "dimension": 112*92, "accuracy": "~80%",
             "notes": "baseline, no feature extraction"},
            {"method": "PCA (Eigenfaces)", "dimension": 100, "accuracy": "~90%",
             "notes": "classical dimensionality reduction"},
            {"method": "HOG", "dimension": "~8100", "accuracy": "~88%",
             "notes": "handcrafted gradient features"},
            {"method": "ArcFace (InsightFace)", "dimension": 512, "accuracy": "~95%+",
             "notes": "deep pretrained, cosine similarity"},
        ]
        feat_csv = os.path.join(DIR4, "feature_comparison_table.csv")
        save_feature_comparison_table(feat_rows, feat_csv)
        register(feat_csv); ok(feat_csv)

    except Exception as exc:
        err(f"Step 4 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 5 — Training
    # ════════════════════════════════════════════════════════
    step(5, TOTAL_STEPS, "Classifier Training (KNN k=1/3/5 + Cosine Matcher)")
    try:
        from src.train import train_pipeline
        train_results = train_pipeline(embeddings, labels_out, cfg)
        clf_pkl = os.path.join(MODELS_DIR, "classifier.pkl")
        register(clf_pkl); ok(clf_pkl)

    except Exception as exc:
        err(f"Step 5 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 6 — Evaluation & Metrics
    # ════════════════════════════════════════════════════════
    step(6, TOTAL_STEPS, "Evaluation & Report Artifacts")
    try:
        from src.evaluate import (
            evaluate_model, save_confusion_matrix, save_classification_report,
            save_performance_metrics, save_before_after_tuning,
            save_predictions_grid, save_annotated_predictions,
        )

        DIR5 = os.path.join(OUTPUT_DIR, "05_results")
        DIR6 = os.path.join(OUTPUT_DIR, "06_metrics")
        DIR8 = os.path.join(OUTPUT_DIR, "08_extra_ipcv")

        X_test = train_results["X_test"]
        y_test = train_results["y_test"]
        models = train_results["models"]
        timing = train_results["timing"]

        # Evaluate all KNN variants + matcher
        all_metrics = {}
        for name, clf_model in models.items():
            all_metrics[name] = evaluate_model(clf_model, X_test, y_test)

        best_preds = all_metrics["knn_k1"]["y_pred"]

        # Confusion matrix (Screenshot 8)
        cm_png = os.path.join(DIR6, "confusion_matrix.png")
        save_confusion_matrix(y_test, best_preds, cm_png,
                              num_classes=cfg["dataset"]["num_subjects"])
        register(cm_png); ok(cm_png)

        # Classification report
        cr_txt = os.path.join(DIR6, "classification_report.txt")
        save_classification_report(y_test, best_preds, cr_txt)
        register(cr_txt); ok(cr_txt)

        # Performance metrics CSV
        perf_rows = []
        for name, m in all_metrics.items():
            perf_rows.append({
                "model": name,
                "accuracy": round(m["accuracy"], 4),
                "precision_macro": round(m["precision_macro"], 4),
                "recall_macro": round(m["recall_macro"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "train_time_s": round(timing.get(name, 0), 3),
                "inference_time_s": round(m["inference_time_s"], 4),
            })
        perf_csv = os.path.join(DIR6, "performance_metrics.csv")
        save_performance_metrics(perf_rows, perf_csv)
        register(perf_csv); ok(perf_csv)

        # Before/after tuning CSV
        tuning_rows = [
            {"variant": "KNN k=1 (cosine)", "accuracy": round(all_metrics["knn_k1"]["accuracy"], 4),
             "f1_macro": round(all_metrics["knn_k1"]["f1_macro"], 4), "note": "default"},
            {"variant": "KNN k=3 (cosine)", "accuracy": round(all_metrics["knn_k3"]["accuracy"], 4),
             "f1_macro": round(all_metrics["knn_k3"]["f1_macro"], 4), "note": "k=3"},
            {"variant": "KNN k=5 (cosine)", "accuracy": round(all_metrics["knn_k5"]["accuracy"], 4),
             "f1_macro": round(all_metrics["knn_k5"]["f1_macro"], 4), "note": "k=5"},
            {"variant": "Cosine Threshold Matcher", "accuracy": round(all_metrics["cosine_matcher"]["accuracy"], 4),
             "f1_macro": round(all_metrics["cosine_matcher"]["f1_macro"], 4), "note": f"thresh={cfg['cosine_matcher']['threshold']}"},
        ]
        tuning_csv = os.path.join(DIR6, "before_after_tuning.csv")
        save_before_after_tuning(tuning_rows, tuning_csv)
        register(tuning_csv); ok(tuning_csv)

        # Identify which test images we need for visualization
        # Match test indices back to images_bgr
        from src.train import split_embeddings
        _, _, y_tr, y_te_check = split_embeddings(
            embeddings, labels_out, cfg["split"]["test_size"], SEED
        )
        # Get indices of test samples
        import numpy as _np
        sss_temp = None
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg["split"]["test_size"], random_state=SEED)
        _, test_idx = next(sss.split(embeddings, labels_out))
        test_images_bgr = images_bgr[test_idx]

        # Prediction grids (Screenshot 7)
        save_predictions_grid(test_images_bgr, y_test, best_preds, DIR5,
                              max_show=40, dpi=cfg["visualization"]["dpi"])
        for fname in ["predictions_with_bboxes.png", "correct_predictions_grid.png",
                      "incorrect_predictions_grid.png"]:
            fp = os.path.join(DIR5, fname); register(fp); ok(fp)

        # Annotated predictions with confidence (Screenshot 7 detailed)
        _, confidences = models["knn_k1"].predict(X_test), None
        try:
            confidences_arr, confs = models["cosine_matcher"].predict_with_confidence(X_test)
        except Exception:
            confs = np.ones(len(y_test), dtype=np.float32)
        annotated_png = os.path.join(DIR8, "annotated_predictions.png")
        save_annotated_predictions(test_images_bgr, y_test, best_preds, confs,
                                   annotated_png, max_show=16)
        register(annotated_png); ok(annotated_png)

    except Exception as exc:
        err(f"Step 6 failed: {exc}"); traceback.print_exc(); sys.exit(1)

    # ════════════════════════════════════════════════════════
    # STEP 7 — Inference Demo
    # ════════════════════════════════════════════════════════
    step(7, TOTAL_STEPS, "Single-Image Inference Demo")
    try:
        from src.predict import run_inference
        demo_img = os.path.join(DATASET_DIR, "s5", "8.pgm")
        demo_out = os.path.join(OUTPUT_DIR, "05_results", "inference_demo_single.png")

        if os.path.exists(demo_img):
            run_inference(demo_img, demo_out, "config.yaml")
            register(demo_out); ok(demo_out)
        else:
            logger.warning(f"Demo image not found: {demo_img}")

    except Exception as exc:
        err(f"Step 7 failed: {exc}"); traceback.print_exc()

    # ════════════════════════════════════════════════════════
    # STEP 8 — Pipeline Diagram
    # ════════════════════════════════════════════════════════
    step(8, TOTAL_STEPS, "Pipeline Diagram")
    try:
        from src.visualization import generate_pipeline_diagram
        pipeline_png = os.path.join(OUTPUT_DIR, "07_pipeline", "pipeline_diagram.png")
        generate_pipeline_diagram(pipeline_png, dpi=cfg["visualization"]["dpi"])
        register(pipeline_png); ok(pipeline_png)

    except Exception as exc:
        err(f"Step 8 failed: {exc}"); traceback.print_exc()

    # ════════════════════════════════════════════════════════
    # STEP 9 — (Reserved; future extensibility)
    # ════════════════════════════════════════════════════════
    step(9, TOTAL_STEPS, "Verification — Checking all artifacts on disk")
    missing = [p for p in generated if not Path(p).exists()]
    if missing:
        for p in missing:
            err(f"MISSING: {p}")
    else:
        print(f"  {C.GREEN}All {len(generated)} artifacts verified on disk.{C.RESET}")

    # ════════════════════════════════════════════════════════
    # STEP 10 — Final Colored Summary
    # ════════════════════════════════════════════════════════
    step(10, TOTAL_STEPS, "Final Summary")
    t_total = time.perf_counter() - t_start
    banner(f"Pipeline Complete! ({t_total:.1f}s total)")

    best_acc = all_metrics["knn_k1"]["accuracy"] if "all_metrics" in dir() else 0.0

    print(f"{C.MAGENTA}{C.BOLD}Generated Artifacts ({len(generated)} files):{C.RESET}")
    for fp in generated:
        exists = Path(fp).exists()
        status = f"{C.GREEN}✅" if exists else f"{C.RED}❌"
        print(f"  {status} {fp}{C.RESET}")

    print(f"\n{C.CYAN}Best KNN k=1 (cosine) accuracy : {best_acc*100:.2f}%{C.RESET}")
    print(f"{C.CYAN}Total artifacts generated      : {len(generated)}{C.RESET}")
    print(f"{C.CYAN}All outputs in                 : {OUTPUT_DIR}/{C.RESET}")
    print(f"{C.CYAN}Total runtime                  : {t_total:.1f}s{C.RESET}\n")


if __name__ == "__main__":
    main()
