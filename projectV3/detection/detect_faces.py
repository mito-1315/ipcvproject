import cv2
from ultralytics import YOLO

def detect_faces(image_path: str, model_path: str):
    """
    Detects faces using YOLOv8.
    Returns: image (numpy array), boxes (list of [x1, y1, x2, y2]), labels (list of str)
    """
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    results = model(image, verbose=False)
    
    boxes = []
    if len(results) > 0 and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            boxes.append(box.tolist())
            
    # Sort boxes from left to right to assign Face_1, Face_2 consistently
    boxes = sorted(boxes, key=lambda x: x[0])
    
    labels = [f"Face_{i+1}" for i in range(len(boxes))]
    return image, boxes, labels
