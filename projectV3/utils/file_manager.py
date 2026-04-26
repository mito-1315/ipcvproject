import os
import datetime

def create_run_directory(base_output_dir: str, image_path: str) -> dict:
    """
    Creates a timestamped run directory and its required subdirectories.
    
    Returns a dictionary of paths.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{image_name}_{timestamp}"
    
    run_dir = os.path.join(base_output_dir, run_folder_name)
    
    paths = {
        "root": run_dir,
        "face-detected": os.path.join(run_dir, "face-detected"),
        "face-cropped": os.path.join(run_dir, "face-cropped"),
        "face-identified": os.path.join(run_dir, "face-identified"),
        "face-name-labelled": os.path.join(run_dir, "face-name-labelled"),
        "logs": os.path.join(run_dir, "logs")
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths
