import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

# Define the paths
original_dataset_dir = 'C:/Users/Administrator/Desktop/pytoremoji/gestures'
base_dir = 'C:/Users/Administrator/Desktop/pytoremoji/splitgest'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

#split ratios here
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Track total number of images
total_train = 0
total_val = 0
total_test = 0

# Splitting the data
for class_folder in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_folder)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        train, temp = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
        val, test = train_test_split(temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

        # Creating class directories in train, val, and test directories
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_folder), exist_ok=True)

        #here wee are moving the images to the respective directories
        for img in train:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_folder, img))
        for img in val:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_folder, img))
        for img in test:
            shutil.move(os.path.join(class_path, img), os.path.join(test_dir, class_folder, img))

        # Update counts
        total_train += len(train)
        total_val += len(val)
        total_test += len(test)

print(f"Total training images: {total_train}")
print(f"Total validation images: {total_val}")
print(f"Total testing images: {total_test}")
print("Dataset split into train, validation, and test sets.")

# Function to process images and convert them to a DataFrame
def images_to_csv(image_dir, output_csv):
    data = []
    labels = []

    for class_folder in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_folder)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (50, 50))  # Resize to 50x50 if necessary
                    img_flat = img.flatten()  # Flatten the image to 1D array
                    data.append(img_flat)
                    labels.append(class_folder)
    
    # Creating a DataFrame
    df = pd.DataFrame(data)
    df['label'] = labels
    
    # Save to CSV status
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

# Convert images to CSV files
images_to_csv(train_dir, 'train_data.csv')
images_to_csv(val_dir, 'val_data.csv')
images_to_csv(test_dir, 'test_data.csv')
