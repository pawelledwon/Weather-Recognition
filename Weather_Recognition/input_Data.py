import os
import cv2
import numpy as np


def read_data_set(weather_names, root_dir, image_size):

    images = []
    labels = []
    problematic_files = []

    for label, weatherName in enumerate(weather_names):
        weather_dir = os.path.join(root_dir, weatherName)
        for filename in os.listdir(weather_dir):
            img_path = os.path.join(weather_dir, filename)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                problematic_files.append(img_path)

    # Convert the lists of images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    if problematic_files:
        print("List of problematic files:")
        for file_path in problematic_files:
            print(file_path)

    return labels, images