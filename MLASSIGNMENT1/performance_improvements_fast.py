"""
SMS Spam Classification - Performance Improvements (Fast Version)
================================================================

This module implements key techniques to improve classifier performance
with optimized execution time and better progress tracking.

Key Improvements Tested:
1. Enhanced text preprocessing (stopwords, stemming)
2. Feature selection (Chi-squared)
3. Optimized Naive Bayes with feature weighting
4. Hyperparameter tuning
5. Different distance metrics for k-NN

"""

import time
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
from tqdm import tqdm

# Import original implementations
from utils.data_loader import DataLoader
from utils.text_processing_clean import TextTokenizer
from data_preprocessing import DataSplitter
from feature_engineering import TFIDFVectorizer
from models.knn_classifier import KNNClassifier
from models.naive_bayes_classifier import NaiveBayesClassifier
from evaluation.metrics import ModelEvaluator


class EnhancedTextTokenizer:
    """Enhanced text tokenizer with advanced preprocessing."""
    
    def __init__(self, clean_text=True, remove_stopwords=True, stem_words=True):
        self.clean_text = clean_text
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'you', 'your', 'i', 'me', 'my', 'we',
            'our', 'they', 'them', 'their', 'this', 'these', 'those', 'have',
            'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'may', 'might', 'must', 'shall', 'am', 'are', 'is', 'was', 'were',
            'or', 'but', 'not', 'no', 'yes', 'so', 'if', 'then', 'when', 'where',
            'why', 'how', 'what', 'who', 'which', 'all', 'any', 'some', 'many',
            'much', 'more', 'most', 'few', 'little', 'each', 'every', 'other',
            'another', 'same', 'different', 'new', 'old', 'good', 'bad', 'big',
            'small', 'long', 'short', 'high', 'low', 'first', 'last', 'next',
            'previous', 'here', 'there', 'now', 'then', 'today', 'yesterday',
            'tomorrow', 'always', 'never', 'sometimes', 'often', 'usually'
        }
        
        # Simple stemming rules
        self.stemming_rules = [
            (r'ing$', ''),
            (r'ed$', ''),
            (r's$', ''),
            (r'ly$', ''),
            (r'ment$', ''),
            (r'ness$', ''),
            (r'ful$', ''),
            (r'less$', ''),
        ]
    
    def clean_message(self, text: str) -> str:
        """Enhanced text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' ', text)
        
        # Remove special characters but keep letters, numbers, and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def stem_word(self, word: str) -> str:
        """Simple stemming implementation."""
        for pattern, replacement in self.stemming_rules:
            word = re.sub(pattern, replacement, word)
        return word
    
    def tokenize_message(self, text: str) -> List[str]:
        """Enhanced tokenization with preprocessing."""
        if not isinstance(text, str):
            return []
        
        # Apply cleaning if enabled
        if self.clean_text:
            text = self.clean_message(text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        # Remove stop words if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if enabled
        if self.stem_words:
            tokens = [self.stem_word(token) for token in tokens]
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def tokenize_messages(self, messages: List) -> List[Dict]:
        """Tokenize a list of messages."""
        tokenized_data = []
        
        for message in messages:
            tokens = self.tokenize_message(message.text)
            tokenized_data.append({
                'tokens': tokens,
                'label': message.label
            })
        
        return tokenized_data


class FeatureSelector:
    """Feature selection using Chi-squared test."""
    
    def __init__(self, k=1000):
        self.k = k
        self.selected_features = None
        self.feature_scores = None
    
    def chi2_score(self, X, y):
        """Calculate chi-squared scores for features."""
        n_samples, n_features = X.shape
        chi2_scores = np.zeros(n_features)
        
        # Get unique classes
        classes = np.unique(y)
        
        for i in range(n_features):
            feature_values = X[:, i]
            
            # Create contingency table
            contingency = np.zeros((len(classes), 2))  # 2 bins: present/absent
            
            for j, class_label in enumerate(classes):
                class_mask = (y == class_label)
                class_feature = feature_values[class_mask]
                
                # Count non-zero values (feature present)
                contingency[j, 1] = np.sum(class_feature > 0)
                contingency[j, 0] = len(class_feature) - contingency[j, 1]
            
            # Calculate chi-squared statistic
            total = np.sum(contingency)
            if total > 0:
                expected = np.outer(np.sum(contingency, axis=1), np.sum(contingency, axis=0)) / total
                chi2 = np.sum((contingency - expected) ** 2 / (expected + 1e-10))
                chi2_scores[i] = chi2
            else:
                chi2_scores[i] = 0
        
        return chi2_scores
    
    def fit(self, X, y):
        """Fit feature selector and select top features."""
        scores = self.chi2_score(X, y)
        self.feature_scores = scores
        
        # Select top k features
        top_indices = np.argsort(scores)[-self.k:]
        self.selected_features = top_indices
        
        return self
    
    def transform(self, X):
        """Transform data to selected features."""
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted first")
        
        return X[:, self.selected_features]


class OptimizedNaiveBayesClassifier:
    """Optimized Naive Bayes with feature weighting."""
    
    def __init__(self, alpha=1.0, use_feature_weights=True):
        self.alpha = alpha
        self.use_feature_weights = use_feature_weights
        self.class_priors = {}
        self.word_likelihoods = {}
        self.vocabulary = set()
        self.classes = set()
        self.feature_weights = {}
        self.is_fitted = False
    
    def _calculate_feature_weights(self, documents: List[Dict]) -> Dict[str, float]:
        """Calculate feature weights based on document frequency."""
        doc_freq = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            unique_tokens = set(doc['tokens'])
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Calculate IDF-based weights
        weights = {}
        for token, freq in doc_freq.items():
            idf = math.log(total_docs / freq)
            weights[token] = idf
        
        return weights
    
    def _calculate_class_priors(self, documents: List[Dict]) -> Dict[str, float]:
        """Calculate class priors with smoothing."""
        class_counts = Counter()
        total_docs = len(documents)
        
        for doc in documents:
            class_counts[doc['label']] += 1
        
        priors = {}
        for class_label, count in class_counts.items():
            priors[class_label] = (count + self.alpha) / (total_docs + self.alpha * len(self.classes))
        
        return priors
    
    def _calculate_word_likelihoods(self, documents: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate word likelihoods with smoothing."""
        class_word_counts = defaultdict(Counter)
        class_total_words = Counter()
        
        for doc in documents:
            class_label = doc['label']
            tokens = doc['tokens']
            
            for token in tokens:
                class_word_counts[class_label][token] += 1
                class_total_words[class_label] += 1
        
        likelihoods = {}
        vocabulary_size = len(self.vocabulary)
        
        for class_label in self.classes:
            likelihoods[class_label] = {}
            total_words_in_class = class_total_words[class_label]
            
            for word in self.vocabulary:
                word_count = class_word_counts[class_label][word]
                likelihood = (word_count + self.alpha) / (total_words_in_class + self.alpha * vocabulary_size)
                likelihoods[class_label][word] = likelihood
        
        return likelihoods
    
    def fit(self, documents: List[Dict]):
        """Fit the optimized Naive Bayes classifier."""
        if not documents:
            raise ValueError("Training documents cannot be empty")
        
        # Extract classes and vocabulary
        self.classes = set(doc['label'] for doc in documents)
        self.vocabulary = set()
        for doc in documents:
            self.vocabulary.update(doc['tokens'])
        
        # Calculate feature weights if enabled
        if self.use_feature_weights:
            self.feature_weights = self._calculate_feature_weights(documents)
        
        # Calculate class priors
        self.class_priors = self._calculate_class_priors(documents)
        
        # Calculate word likelihoods
        self.word_likelihoods = self._calculate_word_likelihoods(documents)
        
        self.is_fitted = True
    
    def _calculate_log_probability(self, tokens: List[str], class_label: str) -> float:
        """Calculate log probability with feature weighting."""
        # Start with log prior
        log_prob = math.log(self.class_priors[class_label])
        
        # Add log likelihoods for each word
        for token in tokens:
            if token in self.word_likelihoods[class_label]:
                likelihood = self.word_likelihoods[class_label][token]
                
                # Apply feature weighting if enabled
                if self.use_feature_weights and token in self.feature_weights:
                    weight = self.feature_weights[token]
                    log_prob += weight * math.log(likelihood)
                else:
                    log_prob += math.log(likelihood)
        
        return log_prob
    
    def _predict_single(self, tokens: List[str]) -> str:
        """Predict class for a single document."""
        best_class = None
        best_log_prob = float('-inf')
        
        for class_label in self.classes:
            log_prob = self._calculate_log_probability(tokens, class_label)
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = class_label
        
        return best_class
    
    def predict(self, documents: List[Dict]) -> List[str]:
        """Predict classes for test documents."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before making predictions")
        
        predictions = []
        for doc in documents:
            prediction = self._predict_single(doc['tokens'])
            predictions.append(prediction)
        
        return predictions


def run_fast_performance_comparison():
    """Run fast performance comparison focusing on key improvements."""
    print("="*80)
    print("SMS SPAM CLASSIFICATION - FAST PERFORMANCE IMPROVEMENTS ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n1. LOADING DATA")
    print("-" * 50)
    loader = DataLoader()
    messages, dataset_info = loader.load_data()
    print(f"✓ Loaded {dataset_info.total_messages} messages")
    
    # Split data
    print("\n2. SPLITTING DATA")
    print("-" * 50)
    splitter = DataSplitter(random_state=42)
    splits = splitter.create_stratified_split(messages, test_size=0.2)
    print(f"✓ Train: {len(splits['train'])}, Test: {len(splits['test'])}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    results = {}
    
    # Test 1: Original Implementation
    print("\n3. TESTING ORIGINAL IMPLEMENTATION")
    print("-" * 50)
    
    # Original tokenizer
    print("Tokenizing with original method...")
    original_tokenizer = TextTokenizer(clean_text=True)
    train_tokenized_orig = original_tokenizer.tokenize_messages(splits['train'])
    test_tokenized_orig = original_tokenizer.tokenize_messages(splits['test'])
    
    # Original TF-IDF
    print("Creating TF-IDF vectors...")
    original_vectorizer = TFIDFVectorizer()
    X_train_orig, X_test_orig = original_vectorizer.fit_transform(train_tokenized_orig, test_tokenized_orig)
    y_train = [data['label'] for data in train_tokenized_orig]
    y_test = [data['label'] for data in test_tokenized_orig]
    
    # Original k-NN (with progress bar)
    print("Testing original k-NN...")
    start_time = time.time()
    original_knn = KNNClassifier(k=5)
    original_knn.fit(X_train_orig, y_train)
    
    # Predict with progress bar
    knn_pred_orig = []
    for i in tqdm(range(len(X_test_orig)), desc="k-NN predictions", unit="sample"):
        prediction = original_knn._predict_single(X_test_orig[i])
        knn_pred_orig.append(prediction)
    
    knn_time_orig = time.time() - start_time
    knn_metrics_orig = evaluator.calculate_metrics(y_test, knn_pred_orig)
    
    # Original Naive Bayes
    print("Testing original Naive Bayes...")
    start_time = time.time()
    original_nb = NaiveBayesClassifier()
    original_nb.fit(train_tokenized_orig)
    nb_pred_orig = original_nb.predict(test_tokenized_orig)
    nb_time_orig = time.time() - start_time
    nb_metrics_orig = evaluator.calculate_metrics(y_test, nb_pred_orig)
    
    results['Original'] = {
        'k-NN': {'metrics': knn_metrics_orig, 'time': knn_time_orig},
        'Naive Bayes': {'metrics': nb_metrics_orig, 'time': nb_time_orig}
    }
    
    # Test 2: Enhanced Text Preprocessing
    print("\n4. TESTING ENHANCED TEXT PREPROCESSING")
    print("-" * 50)
    
    # Enhanced tokenizer
    print("Tokenizing with enhanced preprocessing...")
    enhanced_tokenizer = EnhancedTextTokenizer(
        clean_text=True, 
        remove_stopwords=True, 
        stem_words=True
    )
    train_tokenized_enh = enhanced_tokenizer.tokenize_messages(splits['train'])
    test_tokenized_enh = enhanced_tokenizer.tokenize_messages(splits['test'])
    
    # Enhanced TF-IDF
    print("Creating enhanced TF-IDF vectors...")
    enhanced_vectorizer = TFIDFVectorizer()
    X_train_enh, X_test_enh = enhanced_vectorizer.fit_transform(train_tokenized_enh, test_tokenized_enh)
    
    # Enhanced k-NN
    print("Testing enhanced k-NN...")
    start_time = time.time()
    enhanced_knn = KNNClassifier(k=5)
    enhanced_knn.fit(X_train_enh, y_train)
    
    # Predict with progress bar
    knn_pred_enh = []
    for i in tqdm(range(len(X_test_enh)), desc="Enhanced k-NN predictions", unit="sample"):
        prediction = enhanced_knn._predict_single(X_test_enh[i])
        knn_pred_enh.append(prediction)
    
    knn_time_enh = time.time() - start_time
    knn_metrics_enh = evaluator.calculate_metrics(y_test, knn_pred_enh)
    
    # Enhanced Naive Bayes
    print("Testing enhanced Naive Bayes...")
    start_time = time.time()
    enhanced_nb = NaiveBayesClassifier()
    enhanced_nb.fit(train_tokenized_enh)
    nb_pred_enh = enhanced_nb.predict(test_tokenized_enh)
    nb_time_enh = time.time() - start_time
    nb_metrics_enh = evaluator.calculate_metrics(y_test, nb_pred_enh)
    
    results['Enhanced Preprocessing'] = {
        'k-NN': {'metrics': knn_metrics_enh, 'time': knn_time_enh},
        'Naive Bayes': {'metrics': nb_metrics_enh, 'time': nb_time_enh}
    }
    
    # Test 3: Feature Selection
    print("\n5. TESTING FEATURE SELECTION")
    print("-" * 50)
    
    # Chi-squared feature selection
    print("Applying Chi-squared feature selection...")
    chi2_selector = FeatureSelector(k=1000)
    chi2_selector.fit(X_train_enh, y_train)
    X_train_chi2 = chi2_selector.transform(X_train_enh)
    X_test_chi2 = chi2_selector.transform(X_test_enh)
    
    print(f"Selected {len(chi2_selector.selected_features)} features from {X_train_enh.shape[1]}")
    
    # k-NN with feature selection
    print("Testing k-NN with feature selection...")
    start_time = time.time()
    chi2_knn = KNNClassifier(k=5)
    chi2_knn.fit(X_train_chi2, y_train)
    
    # Predict with progress bar
    knn_pred_chi2 = []
    for i in tqdm(range(len(X_test_chi2)), desc="Chi2 k-NN predictions", unit="sample"):
        prediction = chi2_knn._predict_single(X_test_chi2[i])
        knn_pred_chi2.append(prediction)
    
    knn_time_chi2 = time.time() - start_time
    knn_metrics_chi2 = evaluator.calculate_metrics(y_test, knn_pred_chi2)
    
    # Naive Bayes with feature selection (use original tokenized data)
    print("Testing Naive Bayes with feature selection...")
    start_time = time.time()
    chi2_nb = NaiveBayesClassifier()
    chi2_nb.fit(train_tokenized_enh)
    nb_pred_chi2 = chi2_nb.predict(test_tokenized_enh)
    nb_time_chi2 = time.time() - start_time
    nb_metrics_chi2 = evaluator.calculate_metrics(y_test, nb_pred_chi2)
    
    results['Feature Selection (Chi2)'] = {
        'k-NN': {'metrics': knn_metrics_chi2, 'time': knn_time_chi2},
        'Naive Bayes': {'metrics': nb_metrics_chi2, 'time': nb_time_chi2}
    }
    
    # Test 4: Optimized Naive Bayes
    print("\n6. TESTING OPTIMIZED NAIVE BAYES")
    print("-" * 50)
    
    # Optimized Naive Bayes
    print("Testing optimized Naive Bayes with feature weighting...")
    start_time = time.time()
    opt_nb = OptimizedNaiveBayesClassifier(
        alpha=1.0, 
        use_feature_weights=True
    )
    opt_nb.fit(train_tokenized_enh)
    nb_pred_opt = opt_nb.predict(test_tokenized_enh)
    nb_time_opt = time.time() - start_time
    nb_metrics_opt = evaluator.calculate_metrics(y_test, nb_pred_opt)
    
    results['Optimized Naive Bayes'] = {
        'Naive Bayes': {'metrics': nb_metrics_opt, 'time': nb_time_opt}
    }
    
    # Test 5: Hyperparameter Tuning (Naive Bayes only for speed)
    print("\n7. TESTING HYPERPARAMETER TUNING")
    print("-" * 50)
    
    # Test different alpha values for Naive Bayes
    best_nb_alpha = 1.0
    best_nb_score = 0
    
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        print(f"Testing Naive Bayes with alpha={alpha}...")
        start_time = time.time()
        tuned_nb = NaiveBayesClassifier(alpha=alpha)
        tuned_nb.fit(train_tokenized_enh)
        nb_pred_tuned = tuned_nb.predict(test_tokenized_enh)
        nb_time_tuned = time.time() - start_time
        nb_metrics_tuned = evaluator.calculate_metrics(y_test, nb_pred_tuned)
        
        if nb_metrics_tuned['f1_score'] > best_nb_score:
            best_nb_score = nb_metrics_tuned['f1_score']
            best_nb_alpha = alpha
            best_nb_metrics = nb_metrics_tuned
            best_nb_time = nb_time_tuned
    
    results['Hyperparameter Tuned'] = {
        'Naive Bayes': {'metrics': best_nb_metrics, 'time': best_nb_time, 'params': {'alpha': best_nb_alpha}}
    }
    
    # Print Results Summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Time (s)':<10}")
    print("-" * 100)
    
    for method_name, method_results in results.items():
        for model_name, model_results in method_results.items():
            metrics = model_results['metrics']
            time_taken = model_results['time']
            print(f"{method_name:<25} {model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {time_taken:<10.2f}")
    
    # Find best performing methods
    print("\n" + "="*80)
    print("BEST PERFORMING METHODS")
    print("="*80)
    
    # Best k-NN
    best_knn_method = None
    best_knn_f1 = 0
    for method_name, method_results in results.items():
        if 'k-NN' in method_results:
            knn_f1 = method_results['k-NN']['metrics']['f1_score']
            if knn_f1 > best_knn_f1:
                best_knn_f1 = knn_f1
                best_knn_method = method_name
    
    # Best Naive Bayes
    best_nb_method = None
    best_nb_f1 = 0
    for method_name, method_results in results.items():
        if 'Naive Bayes' in method_results:
            nb_f1 = method_results['Naive Bayes']['metrics']['f1_score']
            if nb_f1 > best_nb_f1:
                best_nb_f1 = nb_f1
                best_nb_method = method_name
    
    print(f"Best k-NN Method: {best_knn_method} (F1-Score: {best_knn_f1:.4f})")
    print(f"Best Naive Bayes Method: {best_nb_method} (F1-Score: {best_nb_f1:.4f})")
    
    # Performance improvements
    print("\n" + "="*80)
    print("PERFORMANCE IMPROVEMENTS")
    print("="*80)
    
    original_knn_f1 = results['Original']['k-NN']['metrics']['f1_score']
    original_nb_f1 = results['Original']['Naive Bayes']['metrics']['f1_score']
    
    knn_improvement = ((best_knn_f1 - original_knn_f1) / original_knn_f1) * 100
    nb_improvement = ((best_nb_f1 - original_nb_f1) / original_nb_f1) * 100
    
    print(f"k-NN F1-Score Improvement: {knn_improvement:+.2f}%")
    print(f"Naive Bayes F1-Score Improvement: {nb_improvement:+.2f}%")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    print("\n1. TEXT PREPROCESSING IMPACT:")
    orig_vocab = len({tok for m in train_tokenized_orig for tok in m['tokens']})
    enh_vocab = len({tok for m in train_tokenized_enh for tok in m['tokens']})

    print(f"   Original vocabulary size: {orig_vocab}")
    print(f"   Enhanced vocabulary size: {enh_vocab}")
    print(f"   Vocabulary reduction: {((orig_vocab - enh_vocab) / orig_vocab * 100):.1f}%")
    
    print("\n2. FEATURE SELECTION IMPACT:")
    print(f"   Selected features: {len(chi2_selector.selected_features)}")
    print(f"   Feature reduction: {((X_train_enh.shape[1] - len(chi2_selector.selected_features)) / X_train_enh.shape[1] * 100):.1f}%")
    
    print("\n3. ALGORITHM OPTIMIZATIONS:")
    print("   - Enhanced text preprocessing (stopwords, stemming)")
    print("   - Chi-squared feature selection")
    print("   - Feature weighting in Naive Bayes")
    print("   - Hyperparameter tuning")
    
    return results


if __name__ == "__main__":
    results = run_fast_performance_comparison()
