import unittest
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import numpy as np

class TestFeatureExtraction(unittest.TestCase):
    def test_feature_extraction(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        feature_extractor = models.Model(inputs=base_model.input, outputs=layers.GlobalAveragePooling2D()(base_model.output))
        
        dummy_data = np.random.rand(10, 128, 128, 3)  # 10 random images
        features = feature_extractor.predict(dummy_data)
        
        self.assertEqual(features.shape, (10, 512))  # 512 is the expected output dimension

if __name__ == '__main__':
    unittest.main()
