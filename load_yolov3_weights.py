import numpy as np
import tensorflow as tf

# Function to load YOLOv3 weights from a file
def load_yolov3_weights(model, weights_file):
    """
    Load the YOLOv3 weights into the model.
    
    Args:
    model: TensorFlow/Keras model object.
    weights_file: Path to the YOLOv3 weights file (binary format).
    
    Returns:
    model: Model with weights loaded.
    """
    # Open the weights file
    with open(weights_file, 'rb') as f:
        # Read the header information
        major, minor, revision = np.fromfile(f, dtype=np.int32, count=3)
        seen = np.fromfile(f, dtype=np.int64, count=1)[0]  # Number of images seen during training
        
        print(f"Loaded YOLOv3 weights file: {weights_file}")
        print(f"Version: {major}.{minor}.{revision}, Images seen: {seen}")
        
        # Iterate over each layer and load weights
        layer_index = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                # Load weights for Conv2D layers
                filters = layer.filters
                kernel_size = layer.kernel_size
                num_filters = filters[0] * filters[1] * layer.input_shape[3] if isinstance(filters, tuple) else filters
                filter_shape = (num_filters, filters[0], filters[1], layer.input_shape[3])
                
                # Read the filter weights
                filters = np.fromfile(f, dtype=np.float32, count=np.prod(filter_shape)).reshape(filter_shape)
                # Read the bias values
                biases = np.fromfile(f, dtype=np.float32, count=filters.shape[0])
                
                # Set the weights and biases to the model layer
                layer.set_weights([filters, biases])
                
                print(f"Loaded weights for layer {layer_index}: Conv2D with {filters.shape[0]} filters")
                layer_index += 1
            elif isinstance(layer, tf.keras.layers.Dense):
                # Load weights for Dense layers
                units = layer.units
                input_dim = layer.input_shape[1]
                
                # Read the weights and biases
                weights = np.fromfile(f, dtype=np.float32, count=units * input_dim).reshape(input_dim, units)
                biases = np.fromfile(f, dtype=np.float32, count=units)
                
                # Set the weights and biases to the model layer
                layer.set_weights([weights, biases])
                
                print(f"Loaded weights for layer {layer_index}: Dense with {units} units")
                layer_index += 1
            else:
                # Skip other layers (e.g., BatchNormalization, Activation)
                continue
    
    print("YOLOv3 weights loaded successfully.")
    return model

# Example usage:
# Assuming `model` is a pre-defined YOLOv3 model built in Keras/TensorFlow
# weights_file = 'path/to/yolov3.weights'
# model = load_yolov3_weights(model, weights_file)
