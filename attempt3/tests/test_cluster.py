import unittest
from sklearn.cluster import KMeans
import numpy as np

class TestKMeans(unittest.TestCase):
    def test_kmeans_clustering(self):
        dummy_features = np.random.rand(100, 2)  # 100 samples in 2D
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans.fit(dummy_features)
        
        self.assertEqual(len(kmeans.cluster_centers_), 5)  # Check number of clusters
        self.assertEqual(len(kmeans.labels_), 100)  # Check labels for all points

if __name__ == '__main__':
    unittest.main()
