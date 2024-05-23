import cv2
import os
import numpy as np

# Path to the directory containing collected images
input_dir = 'dataset'
output_dir = 'preprocessed_dataset'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Resize images to a consistent size and normalize pixel values
image_size = (100, 100)  # Set the desired image size

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is not None:
        # Resize image
        resized_img = cv2.resize(img, image_size)
        
        # Normalize pixel values
        normalized_img = resized_img / 255.0
        
        # Save preprocessed image
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, normalized_img * 255)  # Save normalized image
        
        print(f"Image {img_name} preprocessed and saved.")

print("Preprocessing complete.")