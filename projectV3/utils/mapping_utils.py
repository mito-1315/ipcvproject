import os
import cv2
import json

def save_mapping(mapping_dict: dict, output_path: str):
    """
    Saves the identification mapping as a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(mapping_dict, f, indent=4)

def update_liveoutput(names: list, output_path: str):
    """
    Overwrites the liveoutput.txt file with the newly detected names,
    keeping only the new set of names. One name per line, matching detection order.
    """
    with open(output_path, "w") as f:
        for name in names:
            f.write(f"{name}\n")
