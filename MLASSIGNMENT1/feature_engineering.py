"""
Feature engineering utilities for SMS spam classification.

This module contains the TFIDFVectorizer class for converting text
to numerical features without using third-party libraries.
"""

import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import numpy as np


class TFIDFVectorizer:
    """
    TF-IDF vectorizer for converting text documents to numerical features.
    
    Implements Term Frequency-Inverse Document Frequency without
    third-party dependencies.
    """
    
    def __init__(self, max_features: int = None, min_df: int = 1, max_df: float = 1.0):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to keep
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term (as fraction)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}
        self.idf_ = {}
        self.feature_names_ = []
    
    def _build_vocabulary(self, documents: List[Dict]) -> Dict[str, int]:
        """
        Build vocabulary from training documents.
        
        Args:
            documents: List of documents with 'tokens' key
            
        Returns:
            Dictionary mapping terms to feature indices
        """
        # Count document frequencies
        doc_freq = Counter()
        term_counts = Counter()
        
        for doc in documents:
            tokens = doc['tokens']
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freq[term] += 1
            
            for term in tokens:
                term_counts[term] += 1
        
        # Filter terms based on document frequency
        n_docs = len(documents)
        min_docs = max(1, int(self.min_df * n_docs)) if self.min_df < 1 else self.min_df
        max_docs = int(self.max_df * n_docs) if self.max_df < 1 else n_docs
        
        # Select terms that meet criteria
        valid_terms = []
        for term, df in doc_freq.items():
            if min_docs <= df <= max_docs:
                valid_terms.append((term, term_counts[term]))
        
        # Sort by frequency and limit features
        valid_terms.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            valid_terms = valid_terms[:self.max_features]
        
        # Create vocabulary
        vocabulary = {term: idx for idx, (term, _) in enumerate(valid_terms)}
        return vocabulary
    
    def _calculate_idf(self, documents: List[Dict]) -> Dict[str, float]:
        """
        Calculate Inverse Document Frequency for each term.
        
        Args:
            documents: List of documents with 'tokens' key
            
        Returns:
            Dictionary mapping terms to IDF values
        """
        n_docs = len(documents)
        doc_freq = Counter()
        
        # Count document frequencies
        for doc in documents:
            tokens = doc['tokens']
            unique_terms = set(tokens)
            for term in unique_terms:
                if term in self.vocabulary_:
                    doc_freq[term] += 1
        
        # Calculate IDF
        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log(n_docs / df)
        
        return idf
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calculate Term Frequency for a document.
        
        Args:
            tokens: Document tokens
            
        Returns:
            Dictionary mapping terms to TF values
        """
        term_counts = Counter(tokens)
        doc_length = len(tokens)
        
        # Calculate TF (normalized by document length)
        tf = {}
        for term, count in term_counts.items():
            if term in self.vocabulary_:
                tf[term] = count / doc_length
        
        return tf
    
    def fit_transform(self, train_docs: List[Dict], test_docs: List[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the vectorizer and transform documents to TF-IDF matrix.
        
        Args:
            train_docs: Training documents (list of dicts with 'tokens' key)
            test_docs: Test documents (optional)
            
        Returns:
            Tuple of (X_train, X_test) matrices
        """
        # Build vocabulary from training data
        self.vocabulary_ = self._build_vocabulary(train_docs)
        self.feature_names_ = [term for term, _ in sorted(self.vocabulary_.items(), key=lambda x: x[1])]
        
        # Calculate IDF from training data
        self.idf_ = self._calculate_idf(train_docs)
        
        # Transform training documents
        X_train = self._transform_documents(train_docs)
        
        # Transform test documents if provided
        if test_docs is not None:
            X_test = self._transform_documents(test_docs)
            return X_train, X_test
        else:
            return X_train, None
    
    def _transform_documents(self, documents: List[Dict]) -> np.ndarray:
        """
        Transform documents to TF-IDF matrix.
        
        Args:
            documents: List of documents with 'tokens' key
            
        Returns:
            TF-IDF matrix
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        # Initialize matrix
        matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = doc['tokens']
            # Calculate TF for this document
            tf = self._calculate_tf(tokens)
            
            # Calculate TF-IDF for each term
            for term, tf_val in tf.items():
                if term in self.idf_:
                    feature_idx = self.vocabulary_[term]
                    matrix[doc_idx, feature_idx] = tf_val * self.idf_[term]
        
        return matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names_.copy()
    
    def get_vocabulary_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Number of features
        """
        return len(self.vocabulary_)
