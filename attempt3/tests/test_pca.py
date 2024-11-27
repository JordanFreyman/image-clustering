import unittest
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

class TestPCA(unittest.TestCase):
    def test_pca_reduction(self):
        # Load a real dataset, like the Iris dataset
        iris = load_iris()
        dummy_features = iris.data  # Use real data

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(dummy_features)
        
        # Check that the number of components is 2
        self.assertEqual(reduced_features.shape[1], 2)
        
        # Ensure that PCA retains significant variance (in this case, typically > 90%)
        self.assertGreaterEqual(sum(pca.explained_variance_ratio_), 0.9)  # Retain significant variance

if __name__ == '__main__':
    unittest.main()
