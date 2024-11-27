import unittest
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TestDataPipeline(unittest.TestCase):
    def test_data_pipeline(self):
        train_dir = r"C:/Users/jfrey/Documents/GitHub/image-clustering/attempt3/animals/train"
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical'
        )
        self.assertIsNotNone(generator)
        self.assertGreater(generator.samples, 0)  # Ensure data exists

if __name__ == '__main__':
    unittest.main()
