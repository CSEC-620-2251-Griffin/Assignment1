"""
Model evaluation metrics for SMS spam classification.

This module contains the ModelEvaluator class for calculating performance
metrics without using third-party libraries.
"""

from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class ModelEvaluator:
    """
    Model evaluator for calculating performance metrics.
    
    Implements evaluation metrics from scratch without third-party dependencies.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        pass
    
    def _calculate_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as nested dictionary
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of true and predicted labels must match")
        
        # Get unique classes
        classes = sorted(list(set(y_true + y_pred)))
        
        # Initialize confusion matrix
        cm = {actual: {predicted: 0 for predicted in classes} for actual in classes}
        
        # Fill confusion matrix
        for true_label, pred_label in zip(y_true, y_pred):
            cm[true_label][pred_label] += 1
        
        return cm
    
    def _extract_binary_metrics(self, y_true: List[str], y_pred: List[str], positive_class: str = 'spam') -> Dict[str, int]:
        """
        Extract binary classification metrics from confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Class to treat as positive (default: 'spam')
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        cm = self._calculate_confusion_matrix(y_true, y_pred)
        
        # For binary classification, we need to identify the negative class
        classes = list(cm.keys())
        negative_class = [c for c in classes if c != positive_class][0]
        
        tp = cm[positive_class][positive_class]
        tn = cm[negative_class][negative_class]
        fp = cm[negative_class][positive_class]
        fn = cm[positive_class][negative_class]
        
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
    
    def calculate_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of true and predicted labels must match")
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total = len(y_true)
        
        return correct / total if total > 0 else 0.0
    
    def calculate_precision(self, y_true: List[str], y_pred: List[str], positive_class: str = 'spam') -> float:
        """
        Calculate precision for the positive class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Class to treat as positive
            
        Returns:
            Precision score
        """
        metrics = self._extract_binary_metrics(y_true, y_pred, positive_class)
        tp = metrics['TP']
        fp = metrics['FP']
        
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def calculate_recall(self, y_true: List[str], y_pred: List[str], positive_class: str = 'spam') -> float:
        """
        Calculate recall for the positive class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Class to treat as positive
            
        Returns:
            Recall score
        """
        metrics = self._extract_binary_metrics(y_true, y_pred, positive_class)
        tp = metrics['TP']
        fn = metrics['FN']
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def calculate_f1_score(self, y_true: List[str], y_pred: List[str], positive_class: str = 'spam') -> float:
        """
        Calculate F1-score for the positive class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Class to treat as positive
            
        Returns:
            F1-score
        """
        precision = self.calculate_precision(y_true, y_pred, positive_class)
        recall = self.calculate_recall(y_true, y_pred, positive_class)
        
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def calculate_metrics(self, y_true: List[str], y_pred: List[str], positive_class: str = 'spam') -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            positive_class: Class to treat as positive
            
        Returns:
            Dictionary with all metrics
        """
        accuracy = self.calculate_accuracy(y_true, y_pred)
        precision = self.calculate_precision(y_true, y_pred, positive_class)
        recall = self.calculate_recall(y_true, y_pred, positive_class)
        f1_score = self.calculate_f1_score(y_true, y_pred, positive_class)
        
        # Get confusion matrix components
        metrics = self._extract_binary_metrics(y_true, y_pred, positive_class)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'TP': metrics['TP'],
            'TN': metrics['TN'],
            'FP': metrics['FP'],
            'FN': metrics['FN']
        }
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print formatted metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("Performance Metrics:")
        print("-" * 20)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print()
        print("Confusion Matrix Components:")
        print(f"True Positives:  {metrics['TP']}")
        print(f"True Negatives:  {metrics['TN']}")
        print(f"False Positives: {metrics['FP']}")
        print(f"False Negatives: {metrics['FN']}")
        print()
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]):
        """
        Compare multiple models.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
        """
        print("Model Comparison:")
        print("=" * 60)
        
        # Print header
        print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        # Print results for each model
        for model_name, metrics in model_results.items():
            print(f"{model_name:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        print()
        
        # Find best model for each metric
        best_accuracy = max(model_results.items(), key=lambda x: x[1]['accuracy'])
        best_precision = max(model_results.items(), key=lambda x: x[1]['precision'])
        best_recall = max(model_results.items(), key=lambda x: x[1]['recall'])
        best_f1 = max(model_results.items(), key=lambda x: x[1]['f1_score'])
        
        print("Best Performers:")
        print(f"Best Accuracy:  {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
        print(f"Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
        print(f"Best Recall:    {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        print(f"Best F1-Score:  {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        print()
    
    def get_classification_report(self, y_true: List[str], y_pred: List[str]) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report
        """
        classes = sorted(list(set(y_true + y_pred)))
        report = []
        
        report.append("Classification Report:")
        report.append("=" * 50)
        report.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report.append("-" * 50)
        
        for class_label in classes:
            precision = self.calculate_precision(y_true, y_pred, class_label)
            recall = self.calculate_recall(y_true, y_pred, class_label)
            f1 = self.calculate_f1_score(y_true, y_pred, class_label)
            support = sum(1 for label in y_true if label == class_label)
            
            report.append(f"{class_label:<10} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        
        # Add macro averages
        macro_precision = np.mean([self.calculate_precision(y_true, y_pred, c) for c in classes])
        macro_recall = np.mean([self.calculate_recall(y_true, y_pred, c) for c in classes])
        macro_f1 = np.mean([self.calculate_f1_score(y_true, y_pred, c) for c in classes])
        total_support = len(y_true)
        
        report.append("-" * 50)
        report.append(f"{'Macro Avg':<10} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f} {total_support:<10}")
        
        return "\n".join(report)
