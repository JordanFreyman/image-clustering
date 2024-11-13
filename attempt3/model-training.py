import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/Stuff I've made/cis4930 data mining/attempt3/animals/train"
val_dir = "C:/Users/jfrey/Documents/Stuff I've made/cis4930 data mining/attempt3/animals/val"

# Initialize ImageDataGenerators with fewer augmentations to speed up
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Increase batch size and decrease target image resolution for faster processing
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  # Lower resolution (e.g., 64x64) to speed up
    batch_size=64,  # Larger batch size to reduce steps per epoch
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),
    batch_size=64,
    class_mode='categorical'
)

# Load a smaller model (VGG16 with fewer layers) or use VGG16 and freeze more layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

# Build and compile the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Fewer neurons in the dense layer
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with fewer epochs
# Train the model using the training and validation generators
history = model.fit(
    train_generator,
    steps_per_epoch=None,  # Automatically determine steps from data
    epochs=5,
    validation_data=val_generator,
    validation_steps=None  # Automatically determine steps from data
)


# Save the trained model
model.save('animal_classifier_model_reduced.h5')

# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Feature Extraction and PCA for Clustering
# Create a feature extraction model after training by specifying inputs and desired output layer
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features from the training and validation sets
train_features = feature_extractor.predict(train_generator, verbose=1)
val_features = feature_extractor.predict(val_generator, verbose=1)

# Flatten features for PCA
train_features_flat = train_features.reshape(train_features.shape[0], -1)
val_features_flat = val_features.reshape(val_features.shape[0], -1)

# Apply PCA with fewer components
pca = PCA(n_components=20)  # Fewer components for a simpler, faster PCA
reduced_train_features = pca.fit_transform(train_features_flat)
reduced_val_features = pca.transform(val_features_flat)

# Save the reduced features
np.save('reduced_train_features.npy', reduced_train_features)
np.save('reduced_val_features.npy', reduced_val_features)
