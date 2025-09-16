"""
SMS Spam Classification Project
==============================

Main implementation file for SMS spam detection using k-Nearest Neighbors 
and Naive Bayes classifiers with progress tracking and time estimation.

"""

import time
from tqdm import tqdm
from utils.data_loader import DataLoader
from utils.text_processing_clean import TextTokenizer
from data_preprocessing import DataSplitter
from feature_engineering import TFIDFVectorizer
from models.knn_classifier import KNNClassifier
from models.naive_bayes_classifier import NaiveBayesClassifier
from evaluation.metrics import ModelEvaluator
from evaluation.visualization import ResultsVisualizer


def format_time(seconds):
    """Format time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def main():
    """Main function to run the complete SMS spam classification pipeline."""
    start_time = time.time()
    
    print("="*80)
    print("SMS SPAM CLASSIFICATION PROJECT")
    print("="*80)
    
    try:
        # Step 1: Load and preprocess data
        step_start = time.time()
        print("\n1. LOADING AND PREPROCESSING DATA")
        print("-" * 50)
        
        loader = DataLoader()
        messages, dataset_info = loader.load_data()
        
        step_time = time.time() - step_start
        print(f"✓ Loaded {dataset_info.total_messages} messages")
        print(f"✓ Ham: {dataset_info.ham_count} ({dataset_info.ham_percentage:.1f}%)")
        print(f"✓ Spam: {dataset_info.spam_count} ({dataset_info.spam_percentage:.1f}%)")
        print(f"⏱️  Data loading completed in {format_time(step_time)}")
        
        # Step 2: Tokenize text
        step_start = time.time()
        print("\n2. TOKENIZING TEXT")
        print("-" * 50)
        
        tokenizer = TextTokenizer(clean_text=True)
        
        # Add progress bar for tokenization
        print("Tokenizing messages...")
        tokenized_data = []
        for message in tqdm(messages, desc="Tokenizing", unit="msg"):
            tokens = tokenizer.tokenize_message(message.text)
            tokenized_data.append({
                'tokens': tokens,
                'label': message.label
            })
        
        vocabulary = tokenizer.get_vocabulary(tokenized_data)
        
        step_time = time.time() - step_start
        print(f"✓ Tokenized {len(tokenized_data)} messages")
        print(f"✓ Vocabulary size: {len(vocabulary)}")
        print(f"⏱️  Tokenization completed in {format_time(step_time)}")
        
        # Step 3: Split data
        step_start = time.time()
        print("\n3. SPLITTING DATA")
        print("-" * 50)
        
        # Split data into train and test sets Randomly using random_state=42
        splitter = DataSplitter(random_state=42)
        splits = splitter.create_stratified_split(messages, test_size=0.2)
        splitter.print_split_analysis(splits)
        
        step_time = time.time() - step_start
        print(f"⏱️  Data splitting completed in {format_time(step_time)}")
        
        # Step 4: Feature engineering
        step_start = time.time()
        print("\n4. FEATURE ENGINEERING")
        print("-" * 50)
        
        # Tokenize splits with progress bar
        print("Tokenizing training set...")
        train_tokenized = []
        for message in tqdm(splits['train'], desc="Train tokens", unit="msg"):
            tokens = tokenizer.tokenize_message(message.text)
            train_tokenized.append({
                'tokens': tokens,
                'label': message.label
            })
        
        print("Tokenizing test set...")
        test_tokenized = []
        for message in tqdm(splits['test'], desc="Test tokens", unit="msg"):
            tokens = tokenizer.tokenize_message(message.text)
            test_tokenized.append({
                'tokens': tokens,
                'label': message.label
            })
        
        # Create TF-IDF vectors with progress tracking
        print("Creating TF-IDF vectors...")
        vectorizer = TFIDFVectorizer()
        X_train, X_test = vectorizer.fit_transform(train_tokenized, test_tokenized)
        y_train = [data['label'] for data in train_tokenized]
        y_test = [data['label'] for data in test_tokenized]
        
        step_time = time.time() - step_start
        print(f"✓ Created TF-IDF vectors")
        print(f"✓ Training features: {X_train.shape}")
        print(f"✓ Test features: {X_test.shape}")
        print(f"⏱️  Feature engineering completed in {format_time(step_time)}")
        
        # Step 5: Train models
        step_start = time.time()
        print("\n5. TRAINING MODELS")
        print("-" * 50)
        
        # Train k-NN classifier with progress tracking
        print("Training k-NN classifier...")
        knn_start = time.time()
        knn = KNNClassifier(k=5)
        knn.fit(X_train, y_train)
        
        # Predict with progress bar
        print("Making k-NN predictions...")
        knn_predictions = []
        for i in tqdm(range(len(X_test)), desc="k-NN predictions", unit="sample"):
            prediction = knn._predict_single(X_test[i])
            knn_predictions.append(prediction)
        
        knn_time = time.time() - knn_start
        print(f"✓ k-NN classifier trained and predictions made in {format_time(knn_time)}")
        
        # Train Naive Bayes classifier with progress tracking
        print("Training Naive Bayes classifier...")
        nb_start = time.time()
        nb = NaiveBayesClassifier()
        nb.fit(train_tokenized)  # Naive Bayes works with tokenized data directly
        
        # Predict with progress bar
        print("Making Naive Bayes predictions...")
        nb_predictions = []
        for doc in tqdm(test_tokenized, desc="NB predictions", unit="sample"):
            prediction = nb._predict_single(doc['tokens'])
            nb_predictions.append(prediction)
        
        nb_time = time.time() - nb_start
        print(f"✓ Naive Bayes classifier trained and predictions made in {format_time(nb_time)}")
        
        step_time = time.time() - step_start
        print(f"⏱️  Model training completed in {format_time(step_time)}")
        
        # Step 6: Evaluate models
        step_start = time.time()
        print("\n6. EVALUATING MODELS")
        print("-" * 50)
        
        evaluator = ModelEvaluator()
        
        # Evaluate k-NN
        knn_metrics = evaluator.calculate_metrics(y_test, knn_predictions)
        print("\nk-NN Results:")
        evaluator.print_metrics(knn_metrics)
        
        # Evaluate Naive Bayes
        nb_metrics = evaluator.calculate_metrics(y_test, nb_predictions)
        print("\nNaive Bayes Results:")
        evaluator.print_metrics(nb_metrics)
        
        step_time = time.time() - step_start
        print(f"⏱️  Model evaluation completed in {format_time(step_time)}")
        
        # Step 7: Compare models
        step_start = time.time()
        print("\n7. MODEL COMPARISON")
        print("-" * 50)
        
        evaluator.compare_models({
            'k-NN': knn_metrics,
            'Naive Bayes': nb_metrics
        })
        
        step_time = time.time() - step_start
        print(f"⏱️  Model comparison completed in {format_time(step_time)}")
        
        # Step 8: Visualize results
        step_start = time.time()
        print("\n8. CREATING VISUALIZATIONS")
        print("-" * 50)
        
        visualizer = ResultsVisualizer()
        visualizer.plot_confusion_matrices({
        
        # Create metrics comparison chart
        print("Creating metrics comparison chart...")
        visualizer.plot_metrics_comparison({
            'k-NN': knn_metrics,
            'Naive Bayes': nb_metrics
        })
        
        # Create class distribution chart
        print("Creating class distribution chart...")
        visualizer.plot_class_distribution(y_test, "Test Set Class Distribution")
        
            'k-NN': (y_test, knn_predictions),
            'Naive Bayes': (y_test, nb_predictions)
        })
        
        step_time = time.time() - step_start
        print(f"⏱️  Visualization completed in {format_time(step_time)}")
        
        # Final timing summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️  TOTAL EXECUTION TIME: {format_time(total_time)}")
        print("="*80)
        
        return {
            'dataset_info': dataset_info,
            'splits': splits,
            'knn_metrics': knn_metrics,
            'nb_metrics': nb_metrics,
            'timing': {
                'total_time': total_time,
                'knn_time': knn_time,
                'nb_time': nb_time
            }
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Error in pipeline after {format_time(total_time)}: {e}")
        return None


if __name__ == "__main__":
    results = main()
