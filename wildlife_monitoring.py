import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans

# Real-time progress function
def print_progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Caching function for intermediate results
def cache_data(path, data=None):
    if data is not None:
        np.save(path, data)
    elif os.path.exists(path + ".npy"):
        return np.load(path + ".npy")
    return None

# Set paths to your Snapshot Serengeti dataset directories
train_dir = r"C:/Users/jfrey/Documents/GitHub/wildlife-monitoring/snapshot-serengeti"
feature_cache_path_vgg16 = "./cached_features_vgg16.npy"
feature_cache_path_resnet50 = "./cached_features_resnet50.npy"
feature_cache_path_effnetb0 = "./cached_features_effnetb0.npy"

# Validate paths
if not os.path.exists(train_dir):
    raise FileNotFoundError("Ensure your training directory is correctly set.")

# Initialize ImageDataGenerators for training
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
    validation_split=0.2  # Split 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,  # No labels since it's unsupervised learning
    subset='training'  # Training subset
)

# Function to load feature extraction model (either VGG16 or ResNet50)
def load_feature_extractor(model_name='VGG16'):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    base_model.trainable = False  # We freeze the layers since we're using transfer learning
    
    feature_extractor = models.Model(
        inputs=base_model.input,
        outputs=layers.GlobalAveragePooling2D()(base_model.output)
    )
    return feature_extractor

# Extract features from the generator using the specified model
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

# Function to apply PCA and clustering (MiniBatchKMeans)
def apply_pca_and_clustering(features, n_components=50, n_clusters=10):
    # Apply Incremental PCA for dimensionality reduction
    print_progress("Applying Incremental PCA...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=256)
    reduced_features = ipca.fit_transform(features)

    # Apply MiniBatchKMeans clustering
    print_progress("Clustering with MiniBatchKMeans...")
    mb_kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    cluster_labels = mb_kmeans.fit_predict(reduced_features)

    return reduced_features, cluster_labels, mb_kmeans.cluster_centers_

# Function to visualize the clusters
def plot_clusters(reduced_features, cluster_labels, n_clusters, model_name="Model"):
    plt.figure(figsize=(10, 8))
    for cluster_id in range(n_clusters):
        indices = [j for j, cl_id in enumerate(cluster_labels) if cl_id == cluster_id]
        plt.scatter(
            reduced_features[indices, 0],
            reduced_features[indices, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.6
        )
    
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{model_name} Clusters")
    plt.legend()
    plt.show()

# Function to visualize sample images from each cluster
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

# Compare VGG16 and ResNet50
for model_name, feature_cache_path in zip(['VGG16', 'ResNet50', 'EfficientNetB0'], [feature_cache_path_vgg16, feature_cache_path_resnet50, feature_cache_path_effnetb0]):
    # Load model and check for cached features
    feature_extractor = load_feature_extractor(model_name)
    features = cache_data(feature_cache_path)
    
    if features is None:
        print_progress(f"Extracting features with {model_name}...")
        features = extract_features(train_generator, feature_extractor)

        # Save features to cache
        print_progress(f"Saving {model_name} features to cache...")
        cache_data(feature_cache_path, features)

    # Apply PCA and clustering
    reduced_features, cluster_labels, cluster_centers = apply_pca_and_clustering(features)

    # Visualize clusters for each model
    plot_clusters(reduced_features, cluster_labels, n_clusters=10, model_name=model_name)

    # Display sample images for each cluster
    print_progress(f"Displaying sample images for {model_name} clusters...")
    filepaths = train_generator.filepaths
    for cluster_id in range(10):
        visualize_cluster_samples(cluster_labels, filepaths, cluster_id)

print_progress("Clustering analysis completed for all three models.")