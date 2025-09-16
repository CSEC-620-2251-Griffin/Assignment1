"""
Test script for SMS spam classification implementation.

This script tests all components to ensure they work correctly
without third-party dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.text_processing_clean import TextTokenizer
from data_preprocessing import DataSplitter
from feature_engineering import TFIDFVectorizer
from models.knn_classifier import KNNClassifier
from models.naive_bayes_classifier import NaiveBayesClassifier
from evaluation.metrics import ModelEvaluator


def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    try:
        loader = DataLoader()
        messages, dataset_info = loader.load_data()
        print(f"✓ Loaded {len(messages)} messages")
        print(f"✓ Ham: {dataset_info.ham_count}, Spam: {dataset_info.spam_count}")
        return messages, dataset_info
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None, None


def test_tokenization(messages):
    """Test text tokenization."""
    print("\nTesting tokenization...")
    try:
        tokenizer = TextTokenizer(clean_text=True)
        tokenized_data = tokenizer.tokenize_messages(messages[:100])  # Test with first 100 messages
        vocabulary = tokenizer.get_vocabulary(tokenized_data)
        stats = tokenizer.get_vocabulary_stats(tokenized_data)
        
        print(f"✓ Tokenized {len(tokenized_data)} messages")
        print(f"✓ Vocabulary size: {len(vocabulary)}")
        print(f"✓ Average tokens per message: {stats['avg_tokens_per_message']:.2f}")
        return tokenized_data
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return None


def test_data_splitting(messages):
    """Test data splitting."""
    print("\nTesting data splitting...")
    try:
        splitter = DataSplitter(random_state=42)
        splits = splitter.create_stratified_split(messages[:1000], test_size=0.2)  # Test with first 1000 messages
        splitter.print_split_analysis(splits)
        return splits
    except Exception as e:
        print(f"❌ Data splitting failed: {e}")
        return None


def test_tfidf_vectorization(splits):
    """Test TF-IDF vectorization."""
    print("\nTesting TF-IDF vectorization...")
    try:
        tokenizer = TextTokenizer(clean_text=True)
        train_tokenized = tokenizer.tokenize_messages(splits['train'])
        test_tokenized = tokenizer.tokenize_messages(splits['test'])
        
        vectorizer = TFIDFVectorizer(max_features=1000)
        X_train, X_test = vectorizer.fit_transform(train_tokenized, test_tokenized)
        
        print(f"✓ Training features shape: {X_train.shape}")
        print(f"✓ Test features shape: {X_test.shape}")
        print(f"✓ Vocabulary size: {vectorizer.get_vocabulary_size()}")
        return X_train, X_test, train_tokenized, test_tokenized
    except Exception as e:
        print(f"❌ TF-IDF vectorization failed: {e}")
        return None, None, None, None


def test_knn_classifier(X_train, X_test, splits):
    """Test k-NN classifier."""
    print("\nTesting k-NN classifier...")
    try:
        # Prepare labels
        y_train = [msg.label for msg in splits['train']]
        y_test = [msg.label for msg in splits['test']]
        
        # Train classifier
        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        
        # Make predictions
        predictions = knn.predict(X_test)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, predictions)
        
        print(f"✓ k-NN Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ k-NN F1-Score: {metrics['f1_score']:.4f}")
        return metrics
    except Exception as e:
        print(f"❌ k-NN classifier failed: {e}")
        return None


def test_naive_bayes_classifier(train_tokenized, test_tokenized):
    """Test Naive Bayes classifier."""
    print("\nTesting Naive Bayes classifier...")
    try:
        # Train classifier
        nb = NaiveBayesClassifier(alpha=1.0)
        nb.fit(train_tokenized)
        
        # Make predictions
        predictions = nb.predict(test_tokenized)
        
        # Prepare true labels
        y_test = [doc['label'] for doc in test_tokenized]
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, predictions)
        
        print(f"✓ Naive Bayes Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ Naive Bayes F1-Score: {metrics['f1_score']:.4f}")
        return metrics
    except Exception as e:
        print(f"❌ Naive Bayes classifier failed: {e}")
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("SMS SPAM CLASSIFICATION - IMPLEMENTATION TEST")
    print("=" * 60)
    
    # Test data loading
    messages, dataset_info = test_data_loading()
    if messages is None:
        print("❌ Cannot proceed without data. Exiting.")
        return
    
    # Test tokenization
    tokenized_data = test_tokenization(messages)
    if tokenized_data is None:
        print("❌ Cannot proceed without tokenization. Exiting.")
        return
    
    # Test data splitting
    splits = test_data_splitting(messages)
    if splits is None:
        print("❌ Cannot proceed without data splitting. Exiting.")
        return
    
    # Test TF-IDF vectorization
    X_train, X_test, train_tokenized, test_tokenized = test_tfidf_vectorization(splits)
    if X_train is None:
        print("❌ Cannot proceed without feature engineering. Exiting.")
        return
    
    # Test classifiers
    knn_metrics = test_knn_classifier(X_train, X_test, splits)
    nb_metrics = test_naive_bayes_classifier(train_tokenized, test_tokenized)
    
    # Compare results
    if knn_metrics and nb_metrics:
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        
        evaluator = ModelEvaluator()
        evaluator.compare_models({
            'k-NN': knn_metrics,
            'Naive Bayes': nb_metrics
        })
        
        print("✅ All tests completed successfully!")
        print("✅ Implementation is working correctly without third-party libraries!")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
