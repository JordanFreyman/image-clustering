import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
val_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"

# Initialize ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Set shuffle to False for consistent feature extraction
)

# Load the VGG16 model without the top layer (use pre-trained weights)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model layers

# Feature extraction model
feature_extractor = models.Model(inputs=base_model.input, outputs=layers.GlobalAveragePooling2D()(base_model.output))

# Extract features for clustering
features = feature_extractor.predict(train_generator, steps=train_generator.samples // train_generator.batch_size)
features = features.reshape(features.shape[0], -1)  # Flatten if needed

# Apply PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Apply KMeans clustering directly to the 2D PCA-reduced features
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_features)

# Define animal class names based on the order in your dataset
animal_classes = ["Cat", "Dog", "Elephant", "Lion", "Horse"]

# Select a subset of samples to visualize (e.g., first 500 samples for clarity)
sample_indices = np.random.choice(len(reduced_features), size=500, replace=False)
sampled_features = reduced_features[sample_indices]
sampled_labels = cluster_labels[sample_indices]
sampled_classes = [animal_classes[label] for label in sampled_labels]

# Plot the clustered data with labels
plt.figure(figsize=(10, 8))
for i, animal in enumerate(animal_classes):
    indices = [j for j, label in enumerate(sampled_labels) if label == i]
    plt.scatter(sampled_features[indices, 0], sampled_features[indices, 1], label=animal, alpha=0.6)

# Plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')

# Add legend and titles
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Animal Image Clusters (Labeled by Class)")
plt.legend()
plt.show()