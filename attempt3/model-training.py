import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Set paths to your dataset directories
train_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"  # Change to your actual path
val_dir = "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val"  # Change to your actual path

# Initialize ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for loading the images in batches
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to match input size of VGG16
    batch_size=32,  # You can adjust this based on available memory
    class_mode='categorical'  # Use categorical labels (multi-class classification)
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),  # Resize images to match input size of VGG16
    batch_size=32,
    class_mode='categorical'
)

# Load the VGG16 model without the top layer (use pre-trained weights)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model layers

# Build the model by adding custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Pooling layer to reduce dimensions
    layers.Dense(512, activation='relu'),  # Fully connected layer
    layers.Dense(5, activation='softmax')  # Output layer for 5 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Call the model with a sample input to initialize it
sample_input = tf.random.normal([1, 128, 128, 3])  # Sample input with the correct shape
model(sample_input)  # This will "build" the model with the input shape

# Train the model using the training and validation generators
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Batches per epoch
    epochs=5,  # Adjust the number of epochs as needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size  # Validation batches
)

# Save the trained model
model.save('animal_classifier_model.keras')

# Optionally, visualize training history
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Feature Extraction for Clustering
feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)

# Extract features for the training set
features = feature_extractor.predict(train_generator, steps=train_generator.samples // train_generator.batch_size)
features = features.reshape(features.shape[0], -1)  # Flatten the features if necessary

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)  # Adjust components as needed
reduced_features = pca.fit_transform(features)

# Now you can proceed to clustering with reduced features (step 5)
