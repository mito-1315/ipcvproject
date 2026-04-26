import os
import argparse
import logging
import cv2

from config import YOLO_MODEL_PATH, OUTPUT_ROOT, LIVEOUTPUT_TXT_PATH
from detection.detect_faces import detect_faces
from recognition.identify_faces import FaceIdentifier
from utils.file_manager import create_run_directory
from utils.image_utils import crop_faces, draw_boxes_with_labels
from utils.mapping_utils import save_mapping, update_liveoutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_args():
    parser = argparse.ArgumentParser(description="Face Detection and Identification Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    return parser.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.image):
        logging.error(f"Image not found: {args.image}")
        return

    if not os.path.exists(YOLO_MODEL_PATH):
        logging.error(f"YOLO model not found at {YOLO_MODEL_PATH}")
        return

    logging.info(f"Setting up output directories for {args.image}...")
    paths = create_run_directory(OUTPUT_ROOT, args.image)
    
    # Step 2: Face Detection
    logging.info("Detecting faces...")
    image, boxes, temp_labels = detect_faces(args.image, YOLO_MODEL_PATH)
    
    if not boxes:
        logging.warning("No faces detected in the image.")
        return

    logging.info(f"Detected {len(boxes)} faces.")
    
    # Step 4: Save Face Detection Output
    detected_img_path = os.path.join(paths["face-detected"], "detected.jpg")
    draw_boxes_with_labels(image, boxes, temp_labels, detected_img_path)
    
    # Step 5: Face Cropping
    logging.info("Cropping faces...")
    crop_paths = crop_faces(image, boxes, paths["face-cropped"])
    
    # Initialize the Face Identifier
    logging.info("Initializing Face Identification model...")
    identifier = FaceIdentifier()
    
    # Step 6 & 7: Face Identification and mapping
    logging.info("Identifying faces...")
    mapping_dict = {}
    identified_names = []
    
    for label, crop_path in crop_paths.items():
        crop_img = cv2.imread(crop_path)
        if crop_img is None:
            continue
            
        identity = identifier.identify_face(crop_img)
        mapping_dict[label] = identity
        identified_names.append(identity)
        logging.info(f"Identified {label} -> {identity}")
        
    mapping_json_path = os.path.join(paths["face-identified"], "mapping.json")
    save_mapping(mapping_dict, mapping_json_path)
    
    # Step 8 & 9: Replace Labels with Names and Save Final Output
    logging.info("Generating final annotated image...")
    final_labels = [mapping_dict[label] for label in temp_labels]
    
    final_img_path = os.path.join(paths["face-name-labelled"], "final.jpg")
    detected_img = cv2.imread(detected_img_path) # Or just use original image again
    # We will use original image so we don't double-draw
    draw_boxes_with_labels(image, boxes, final_labels, final_img_path)
    
    # Step 10: Update Live Output File
    logging.info("Updating liveoutput.txt...")
    update_liveoutput(identified_names, LIVEOUTPUT_TXT_PATH)
    
    logging.info("Pipeline completed successfully!")
    logging.info(f"Check results in {paths['root']}")

if __name__ == "__main__":
    main()
