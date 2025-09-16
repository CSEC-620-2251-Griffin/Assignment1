"""
Multinomial Naive Bayes classifier implementation.

This module contains the NaiveBayesClassifier class for SMS spam detection
"""

import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import numpy as np


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes classifier for SMS spam detection.
    
    Implements Naive Bayes algorithm from scratch
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Naive Bayes classifier.
        
        Args:
            alpha: Smoothing parameter (Laplace smoothing)
        """
        self.alpha = alpha
        self.class_priors = {}
        self.word_likelihoods = {}
        self.vocabulary = set()
        self.classes = set()
        self.is_fitted = False
    
    def _extract_vocabulary(self, documents: List[Dict]) -> set:
        """
        Extract vocabulary from training documents.
        
        Args:
            documents: List of documents with 'tokens' and 'label' keys
            
        Returns:
            Set of unique words in the vocabulary
        """
        vocabulary = set()
        for doc in documents:
            vocabulary.update(doc['tokens'])
        return vocabulary
    
    def _calculate_class_priors(self, documents: List[Dict]) -> Dict[str, float]:
        """
        Calculate prior probabilities for each class.
        
        Args:
            documents: List of documents with 'tokens' and 'label' keys
            
        Returns:
            Dictionary mapping class labels to prior probabilities
        """
        class_counts = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            class_counts[doc['label']] += 1
        
        priors = {}
        for class_label, count in class_counts.items():
            priors[class_label] = count / total_docs
        
        return priors
    
    def _calculate_word_likelihoods(self, documents: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate likelihood probabilities for each word given each class.
        
        Args:
            documents: List of documents with 'tokens' and 'label' keys
            
        Returns:
            Nested dictionary: {class_label: {word: likelihood}}
        """
        # Count words per class
        class_word_counts = defaultdict(Counter)
        class_total_words = Counter()
        
        for doc in documents:
            class_label = doc['label']
            tokens = doc['tokens']
            
            for token in tokens:
                class_word_counts[class_label][token] += 1
                class_total_words[class_label] += 1
        
        # Calculate likelihoods with Laplace smoothing
        likelihoods = {}
        vocabulary_size = len(self.vocabulary)
        
        for class_label in self.classes:
            likelihoods[class_label] = {}
            total_words_in_class = class_total_words[class_label]
            
            for word in self.vocabulary:
                word_count = class_word_counts[class_label][word]
                # Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                likelihood = (word_count + self.alpha) / (total_words_in_class + self.alpha * vocabulary_size)
                likelihoods[class_label][word] = likelihood
        
        return likelihoods
    
    def fit(self, documents: List[Dict]):
        """
        Fit the Naive Bayes classifier.
        
        Args:
            documents: List of documents with 'tokens' and 'label' keys
        """
        if not documents:
            raise ValueError("Training documents cannot be empty")
        
        # Extract classes and vocabulary
        self.classes = set(doc['label'] for doc in documents)
        self.vocabulary = self._extract_vocabulary(documents)
        
        # Calculate class priors
        self.class_priors = self._calculate_class_priors(documents)
        
        # Calculate word likelihoods
        self.word_likelihoods = self._calculate_word_likelihoods(documents)
        
        self.is_fitted = True
    
    def _calculate_log_probability(self, tokens: List[str], class_label: str) -> float:
        """
        Calculate log probability of a document given a class.
        
        Args:
            tokens: Document tokens
            class_label: Class label
            
        Returns:
            Log probability
        """
        # Start with log prior
        log_prob = math.log(self.class_priors[class_label])
        
        # Add log likelihoods for each word
        for token in tokens:
            if token in self.word_likelihoods[class_label]:
                log_prob += math.log(self.word_likelihoods[class_label][token])
        
        return log_prob
    
    def _predict_single(self, tokens: List[str]) -> str:
        """
        Predict class for a single document.
        
        Args:
            tokens: Document tokens
            
        Returns:
            Predicted class label
        """
        best_class = None
        best_log_prob = float('-inf')
        
        for class_label in self.classes:
            log_prob = self._calculate_log_probability(tokens, class_label)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = class_label
        
        return best_class
    
    def predict(self, documents: List[Dict]) -> List[str]:
        """
        Predict classes for test documents.
        
        Args:
            documents: List of test documents with 'tokens' key
            
        Returns:
            List of predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        predictions = []
        for doc in documents:
            prediction = self._predict_single(doc['tokens'])
            predictions.append(prediction)
        
        return predictions
    
    def predict_proba(self, documents: List[Dict]) -> np.ndarray:
        """
        Predict class probabilities for test documents.
        
        Args:
            documents: List of test documents with 'tokens' key
            
        Returns:
            Array of shape (n_documents, n_classes) with class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        n_docs = len(documents)
        n_classes = len(self.classes)
        classes_list = sorted(list(self.classes))
        
        probabilities = np.zeros((n_docs, n_classes))
        
        for i, doc in enumerate(documents):
            tokens = doc['tokens']
            
            # Calculate log probabilities for each class
            log_probs = []
            for class_label in classes_list:
                log_prob = self._calculate_log_probability(tokens, class_label)
                log_probs.append(log_prob)
            
            # Convert to probabilities using log-sum-exp trick for numerical stability
            max_log_prob = max(log_probs)
            exp_log_probs = [math.exp(log_prob - max_log_prob) for log_prob in log_probs]
            sum_exp = sum(exp_log_probs)
            
            for j, exp_prob in enumerate(exp_log_probs):
                probabilities[i, j] = exp_prob / sum_exp
        
        return probabilities
    
    def get_feature_importance(self, class_label: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get most important features (words) for a given class.
        
        Args:
            class_label: Class to analyze
            top_n: Number of top features to return
            
        Returns:
            List of (word, likelihood) tuples sorted by importance
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before getting feature importance")
        
        if class_label not in self.classes:
            raise ValueError(f"Unknown class: {class_label}")
        
        word_likelihoods = self.word_likelihoods[class_label]
        sorted_words = sorted(word_likelihoods.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_n]
    
    def get_params(self) -> dict:
        """
        Get classifier parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'alpha': self.alpha
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
