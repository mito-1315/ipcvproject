import os
import random
import cv2
import numpy as np

def create_crowd_collage(input_dir, output_dir, num_samples=50, canvas_size=(800, 800)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_images = [f for f in os.listdir(input_dir) if f.endswith('.pgm')]
    if not all_images:
        print("No .pgm images found in", input_dir)
        return

    for sample_idx in range(num_samples):
        # Create a blank white canvas
        canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255

        # Choose random number of people (1 to 10)
        num_people = random.randint(1, 10)
        selected_images = random.sample(all_images, min(num_people, len(all_images)))

        print(f"Creating crowd sample {sample_idx+1} with {num_people} people...")

        for img_name in selected_images:
            img_path = os.path.join(input_dir, img_name)
            person_img = cv2.imread(img_path)
            
            if person_img is None:
                continue

            # Get dimensions
            h, w = person_img.shape[:2]

            # Random position (allow some part to go slightly off screen, but keep center within canvas)
            x_offset = random.randint(0, max(1, canvas_size[0] - w))
            y_offset = random.randint(0, max(1, canvas_size[1] - h))

            # Overlay
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = person_img

        output_path = os.path.join(output_dir, f"crowd_sample_{sample_idx+1}.jpg")
        cv2.imwrite(output_path, canvas)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_directory = r"e:\Workspace\ipcvproject\Images\allImages"
    output_directory = r"e:\Workspace\ipcvproject\Images\CrowdSamples"
    create_crowd_collage(input_directory, output_directory, num_samples=50)
