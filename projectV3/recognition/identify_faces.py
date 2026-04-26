import sys
import os
import numpy as np
import cv2
import joblib

# Ensure projectV2 is in path to import InsightFace wrappers
from config import PROJECT_V2_DIR, PROJECT_V2_MODELS_DIR
if PROJECT_V2_DIR not in sys.path:
    sys.path.append(PROJECT_V2_DIR)

from src.embeddings import init_insightface, extract_embedding_single

class FaceIdentifier:
    def __init__(self):
        self.clf = None
        self.train_embs = None
        self.train_labels = None
        self.app = None
        
        self.load_models()
        self.init_insightface_app()

    def load_models(self):
        clf_path = os.path.join(PROJECT_V2_MODELS_DIR, "classifier.pkl")
        emb_path = os.path.join(PROJECT_V2_MODELS_DIR, "arcface_embeddings.npy")
        lbl_path = os.path.join(PROJECT_V2_MODELS_DIR, "labels.npy")
        
        self.clf = joblib.load(clf_path)
        if os.path.exists(emb_path):
            self.train_embs = np.load(emb_path)
        if os.path.exists(lbl_path):
            self.train_labels = np.load(lbl_path)

    def init_insightface_app(self):
        self.app = init_insightface(model_name="buffalo_l", providers=["CPUExecutionProvider"], det_size=(160, 160))

    def identify_face(self, bgr_face_img: np.ndarray, thresh_conf: float = 0.3) -> str:
        """
        Extract embedding for the face crop and run KNN to identify identity.
        """
        # We pass the face crop directly. det_score_thresh = 0.0 allows fallback to recognition when cropped face is small
        embedding, _ = extract_embedding_single(self.app, bgr_face_img, det_score_thresh=0.0)
        
        pred_label_index = int(self.clf.predict(embedding.reshape(1, -1))[0])
        
        confidence = 0.0
        if self.train_embs is not None:
            norms = np.linalg.norm(self.train_embs, axis=1, keepdims=True)
            X_norm = self.train_embs / (norms + 1e-8)
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            sims = X_norm @ emb_norm
            confidence = float(sims.max())
            
        if confidence < thresh_conf:
            return "Unknown"
            
        return f"s{pred_label_index+1}"
