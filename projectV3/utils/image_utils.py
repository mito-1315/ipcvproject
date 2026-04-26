import cv2
import os

def crop_faces(image, boxes, output_dir: str):
    """
    Given an image and bounding boxes, crops and saves them as Face_1.jpg, Face_2.jpg...
    boxes format: list of (x1, y1, x2, y2)
    Returns a dictionary mapping label like "Face_1" to its path.
    """
    cropped_paths = {}
    h, w = image.shape[:2]
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label = f"Face_{i+1}"
        # Ensure boxes are within image boundaries
        x1_crop = max(0, int(x1))
        y1_crop = max(0, int(y1))
        x2_crop = min(w, int(x2))
        y2_crop = min(h, int(y2))
        
        crop_img = image[y1_crop:y2_crop, x1_crop:x2_crop]
        if crop_img.size == 0:
            continue
            
        crop_path = os.path.join(output_dir, f"{label}.jpg")
        cv2.imwrite(crop_path, crop_img)
        cropped_paths[label] = crop_path
        
    return cropped_paths

def draw_boxes_with_labels(image, boxes, labels, output_path: str):
    """
    Draw bounding boxes with labels and save image.
    """
    img_copy = image.copy()
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        pt1 = (x1, y1)
        pt2 = (x1 + text_width, y1 - text_height - baseline)
        
        cv2.rectangle(img_copy, pt1, pt2, (0, 255, 0), cv2.FILLED)
        cv2.putText(img_copy, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    cv2.imwrite(output_path, img_copy)
    return img_copy
