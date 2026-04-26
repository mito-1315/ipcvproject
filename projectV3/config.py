import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# YOLO model path
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8n-face.pt")
# Fallback if it's currently at the root
if not os.path.exists(YOLO_MODEL_PATH):
    fallback_path = os.path.join(PROJECT_ROOT, "yolov8n-face.pt")
    if os.path.exists(fallback_path):
        YOLO_MODEL_PATH = fallback_path

# projectV2 paths
PROJECT_V2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "projectV2"))
PROJECT_V2_MODELS_DIR = os.path.join(PROJECT_V2_DIR, "models")
PROJECT_V2_SRC_DIR = os.path.join(PROJECT_V2_DIR, "src")

# Output directory root
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")

# File path for liveoutput.txt
LIVEOUTPUT_TXT_PATH = os.path.join(PROJECT_ROOT, "liveoutput.txt")
