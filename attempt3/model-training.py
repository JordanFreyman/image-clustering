from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
val_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val"

# Initialize ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Load VGG16 model without top layers for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=5,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Visualization
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Feature Extraction and Dimensionality Reduction
feature_extractor = models.Model(inputs=base_model.input, outputs=base_model.output)
train_features = feature_extractor.predict(train_generator, steps=train_generator.samples // train_generator.batch_size)
train_features = train_features.reshape(train_features.shape[0], -1)

# Apply PCA
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(train_features)

# Apply Clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters based on your needs
cluster_labels = kmeans.fit_predict(reduced_features)

# Output cluster labels
print("Cluster labels:", cluster_labels)
