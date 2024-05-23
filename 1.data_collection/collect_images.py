import cv2
import os

# Create a directory to store the collected images
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or specify the camera index

# Set the number of images to collect per individual
num_images_per_person = 10

# Loop to capture images
for person_id in range(1, 6):  # Assuming 5 individuals
    print(f"Collecting images for person {person_id}")
    
    for img_num in range(num_images_per_person):
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to capture image")
            continue
        
        # Save the captured image
        img_path = os.path.join(output_dir, f'person_{person_id}_{img_num}.jpg')
        cv2.imwrite(img_path, frame)
        
        print(f"Image {img_num+1} captured for person {person_id}")
        
    print(f"Images captured for person {person_id}")

# Release the camera
cap.release()
