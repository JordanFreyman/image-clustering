import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def print_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def cache_data(path, data=None):
    if data is not None:
        np.save(path, data)
    elif os.path.exists(path + ".npy"):
        return np.load(path + ".npy")
    return None

train_dir = r"C:/Users/jfrey/Documents/GitHub/wildlife-monitoring/snapshot-serengeti"
feature_cache_path = "./cached_features.npy"

if not os.path.exists(train_dir):
    raise FileNotFoundError("Ensure your training directory is correctly set.")

print_progress("Initializing image data generators...")
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,  
    subset='training'  
)

print_progress("Loading VGG16 model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

print_progress("Building feature extraction model...")
feature_extractor = models.Model(
    inputs=base_model.input,
    outputs=layers.GlobalAveragePooling2D()(base_model.output)
)

print_progress("Checking feature cache...")
features = cache_data(feature_cache_path)

if features is None:
    print_progress("Extracting features...")

    def extract_features(generator, feature_extractor_model):
        features_list = []
        num_batches = len(generator)
        for i, batch in enumerate(generator):
            if i >= num_batches:
                break
            print_progress(f"Processing batch {i + 1}/{num_batches}...")
            features = feature_extractor_model.predict(batch)
            features_list.append(features)
        return np.vstack(features_list)

    features = extract_features(train_generator, feature_extractor)

    print_progress("Saving features to cache...")
    cache_data(feature_cache_path, features)

print_progress("Applying Incremental PCA...")
n_components = 50  # Increase to capture more features
ipca = IncrementalPCA(n_components=n_components, batch_size=256)
reduced_features = ipca.fit_transform(features)

print_progress("Clustering with MiniBatchKMeans...")
n_clusters = 10  # Experiment with different numbers of clusters
mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
cluster_labels = mb_kmeans.fit_predict(reduced_features)

print_progress("Visualizing cluster distributions...")
plt.figure(figsize=(10, 8))
for cluster_id in range(n_clusters):
    indices = [j for j, cl_id in enumerate(cluster_labels) if cl_id == cluster_id]
    plt.scatter(
        reduced_features[indices, 0],
        reduced_features[indices, 1],
        label=f"Cluster {cluster_id}",
        alpha=0.6
    )

centers = mb_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Wildlife Clusters (Unlabeled)")
plt.legend()
plt.show()

print_progress("Displaying representative images for each cluster...")
filepaths = train_generator.filepaths

def visualize_cluster_samples(cluster_labels, filepaths, cluster_id, num_samples=5):
    indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    selected_indices = np.random.choice(indices, size=min(len(indices), num_samples), replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(selected_indices):
        img = plt.imread(filepaths[idx])
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Sample Images from Cluster {cluster_id}")
    plt.show()

for cluster_id in range(n_clusters):
    visualize_cluster_samples(cluster_labels, filepaths, cluster_id)

print_progress("Clustering analysis completed.")
