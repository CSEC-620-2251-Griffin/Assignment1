"""
k-Nearest Neighbors classifier implementation.

This module contains the KNNClassifier class for SMS spam detection
without using third-party libraries.
"""

import numpy as np
from typing import List, Tuple, Union
from collections import Counter


class KNNClassifier:
    """
    k-Nearest Neighbors classifier for SMS spam detection.
    
    Implements k-NN algorithm from scratch without third-party dependencies.
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
        """
        Initialize the k-NN classifier.
        
        Args:
            k: Number of nearest neighbors to consider
            distance_metric: Distance metric to use ('euclidean' or 'manhattan')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate distance between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Distance between the vectors
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _find_k_nearest_neighbors(self, x: np.ndarray) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors for a given sample.
        
        Args:
            x: Input sample
            
        Returns:
            List of (index, distance) tuples for k nearest neighbors
        """
        distances = []
        
        for i, train_sample in enumerate(self.X_train):
            distance = self._calculate_distance(x, train_sample)
            distances.append((i, distance))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:self.k]
    
    def _predict_single(self, x: np.ndarray) -> str:
        """
        Predict label for a single sample.
        
        Args:
            x: Input sample
            
        Returns:
            Predicted label
        """
        # Find k nearest neighbors
        neighbors = self._find_k_nearest_neighbors(x)
        
        # Get labels of neighbors
        neighbor_labels = [self.y_train[idx] for idx, _ in neighbors]
        
        # Majority vote
        label_counts = Counter(neighbor_labels)
        predicted_label = label_counts.most_common(1)[0][0]
        
        return predicted_label
    
    def fit(self, X: np.ndarray, y: List[str]):
        """
        Fit the k-NN classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must match")
        
        if self.k > len(X):
            raise ValueError(f"k ({self.k}) cannot be greater than number of training samples ({len(X)})")
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> List[str]:
        """
        Predict labels for test samples.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            List of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Number of features in X must match training data")
        
        predictions = []
        for sample in X:
            prediction = self._predict_single(sample)
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test samples.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        # Get unique classes
        classes = sorted(list(set(self.y_train)))
        n_classes = len(classes)
        n_samples = X.shape[0]
        
        probabilities = np.zeros((n_samples, n_classes))
        
        for i, sample in enumerate(X):
            # Find k nearest neighbors
            neighbors = self._find_k_nearest_neighbors(sample)
            
            # Get labels of neighbors
            neighbor_labels = [self.y_train[idx] for idx, _ in neighbors]
            
            # Calculate probabilities based on neighbor counts
            label_counts = Counter(neighbor_labels)
            total_neighbors = len(neighbor_labels)
            
            for j, class_label in enumerate(classes):
                count = label_counts.get(class_label, 0)
                probabilities[i, j] = count / total_neighbors
        
        return probabilities
    
    def get_params(self) -> dict:
        """
        Get classifier parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'k': self.k,
            'distance_metric': self.distance_metric
        }
    
    def set_params(self, **params):
        """
        Set classifier parameters.
        
        Args:
            **params: Parameters to set
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown parameter: {param}")
        
        # Reset fitted state if parameters changed
        self.is_fitted = False
