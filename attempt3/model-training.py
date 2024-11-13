import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
val_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val"

# Ensure directories exist
assert os.path.exists(train_dir), f"Train directory does not exist: {train_dir}"
assert os.path.exists(val_dir), f"Validation directory does not exist: {val_dir}"

# Initialize ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for loading the images in batches
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode='categorical')

# Load the MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model layers

# Build the model by adding custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training and validation generators
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, epochs=3,
                    validation_data=val_generator, validation_steps=val_generator.samples // val_generator.batch_size)

# Save the trained model
model.save('animal_classifier_model.keras')

# Optionally, visualize training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Feature Extraction for Clustering
# Use base_model directly for feature extraction
feature_extractor = models.Model(inputs=base_model.input, outputs=base_model.output)
features = feature_extractor.predict(train_generator, steps=train_generator.samples // train_generator.batch_size)
features = features.reshape(features.shape[0], -1)  # Flatten the features if necessary

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Now you can proceed to clustering with reduced features
