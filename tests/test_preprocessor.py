import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.sample_data = pd.DataFrame({
            'user_id': [1, 2, 1, 3],
            'item_id': [101, 102, 103, 101],
            'rating': [4, 3, 5, 2],
            'age': [25, 30, 25, 35],
            'gender': ['M', 'F', 'M', 'M'],
            'occupation': ['student', 'engineer', 'student', 'doctor'],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie A'],
            'year': [2000, 2001, 2002, 2000],
            'genres': ["['Action']", "['Comedy']", "['Drama']", "['Action']"]
        })
        
    def test_feature_preparation(self):
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(self.sample_data, is_training=True)
        
        # Check that features are created
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(self.sample_data))
        
        # Check that statistical features are calculated
        self.assertTrue(hasattr(preprocessor, 'global_avg_rating'))
        self.assertTrue(hasattr(preprocessor, 'user_avg_ratings'))
        self.assertTrue(hasattr(preprocessor, 'item_avg_ratings'))
        
    def test_rating_constraints(self):
        # Test that ratings are properly constrained
        self.assertTrue(all(self.sample_data['rating'] >= 1))
        self.assertTrue(all(self.sample_data['rating'] <= 5))

if __name__ == '__main__':
    unittest.main()