from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize image pixel values to [0, 1]
    rotation_range=40,  # Random rotations for augmentation
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Random shearing transformations
    zoom_range=0.2,  # Random zoom transformations
    horizontal_flip=True,  # Random horizontal flipping
    fill_mode='nearest'  # Fill mode for missing pixels
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only normalization for validation data

# Load and iterate training and validation datasets
train_generator = train_datagen.flow_from_directory(
    "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train",  # Replace with your actual train directory path
    target_size=(128, 128),  # Resize all images to 128x128 pixels
    batch_size=32,  # Batch size
    class_mode='categorical'  # Multi-class classification
)

val_generator = val_datagen.flow_from_directory(
    "C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/val",  # Replace with your actual validation directory path
    target_size=(128, 128),  # Resize all images to 128x128 pixels
    batch_size=32,  # Batch size
    class_mode='categorical'  # Multi-class classification
)
