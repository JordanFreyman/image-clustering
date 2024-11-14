import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
val_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val"

# Initialize ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)

# Create data generator for loading a smaller sample of the images to speed up testing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,  # Reduce batch size to make initial runs faster
    class_mode='categorical',
    shuffle=False  # Disable shuffling to make cluster visualization easier
)

# Load a pre-trained VGG16 model for feature extraction, excluding the final classification layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze layers

# Define a model to output features
feature_extractor = models.Model(inputs=base_model.input, outputs=layers.GlobalAveragePooling2D()(base_model.output))

# Extract features from images
features = feature_extractor.predict(train_generator, steps=train_generator.samples // train_generator.batch_size, verbose=1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Perform K-Means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(reduced_features)
cluster_labels = kmeans.labels_
print("Cluster labels:", cluster_labels)

# Visualize a sample of images from each cluster
def visualize_clusters(train_dir, cluster_labels, n_clusters=5, num_images=5):
    fig, axs = plt.subplots(n_clusters, num_images, figsize=(15, 8))
    fig.suptitle("Sample Images from Each Cluster")
    
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        sample_indices = random.sample(list(cluster_indices), min(num_images, len(cluster_indices)))
        
        for i, img_index in enumerate(sample_indices):
            img_path = train_generator.filepaths[img_index]
            img = Image.open(img_path).resize((128, 128))
            axs[cluster, i].imshow(img)
            axs[cluster, i].axis("off")
            axs[cluster, i].set_title(f"Cluster {cluster}")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

visualize_clusters(train_dir, cluster_labels)
