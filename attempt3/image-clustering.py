# image_clustering.py
"""
Animal Image Clustering with PCA and KMeans
Author: Jordan Freyman
Description: This script processes animal image data, extracts features using a pre-trained VGG16 model, 
applies PCA for dimensionality reduction, and clusters the data using KMeans. Visualizations are provided for analysis.
Credit attributed to ChatGPT for assistance in script development.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Real-time progress function
def print_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Set paths to your dataset directories
train_dir = r"C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
val_dir = r"C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val"

# Validate paths
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("Ensure your train and validation directories are correctly set.")

# Initialize ImageDataGenerators for training
print_progress("Initializing image data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Shuffle is False for consistent feature extraction
)

# Load the VGG16 model without the top layer
print_progress("Loading VGG16 model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Build feature extraction model
print_progress("Building feature extraction model...")
feature_extractor = models.Model(
    inputs=base_model.input, 
    outputs=layers.GlobalAveragePooling2D()(base_model.output)
)

# Feature extraction
print_progress("Extracting features...")
features = feature_extractor.predict(
    train_generator, 
    steps=train_generator.samples // train_generator.batch_size, 
    verbose=1
)
features = features.reshape(features.shape[0], -1)

# Apply PCA for dimensionality reduction
print_progress("Applying PCA...")
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Apply KMeans clustering
print_progress("Clustering with KMeans...")
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Define animal class names based on the dataset
animal_classes = ["Cat", "Dog", "Elephant", "Lion", "Horse"]

# Visualization of clustered data
print_progress("Visualizing clusters...")
sample_indices = np.random.choice(len(reduced_features), size=500, replace=False)
sampled_features = reduced_features[sample_indices]
sampled_labels = cluster_labels[sample_indices]

plt.figure(figsize=(10, 8))
for i, animal in enumerate(animal_classes):
    indices = [j for j, label in enumerate(sampled_labels) if label == i]
    plt.scatter(sampled_features[indices, 0], sampled_features[indices, 1], label=animal, alpha=0.6)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Animal Image Clusters")
plt.legend()
plt.show()

print_progress("Clustering and visualization completed.")
