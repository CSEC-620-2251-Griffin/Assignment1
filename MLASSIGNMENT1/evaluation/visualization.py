"""
Visualization utilities for SMS spam classification results.

This module contains the ResultsVisualizer class for creating plots
without using third-party libraries.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from evaluation.metrics import ModelEvaluator


class ResultsVisualizer:
    """
    Results visualizer for SMS spam classification.
    
    Creates visualizations using matplotlib (which is allowed for plotting).
    """
    
    def __init__(self):
        """Initialize the results visualizer."""
        self.evaluator = ModelEvaluator()
    
    def plot_confusion_matrices(self, model_results: Dict[str, Tuple[List[str], List[str]]]):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            model_results: Dictionary mapping model names to (y_true, y_pred) tuples
        """
        n_models = len(model_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, (y_true, y_pred)) in enumerate(model_results.items()):
            # Calculate confusion matrix
            cm = self.evaluator._calculate_confusion_matrix(y_true, y_pred)
            
            # Convert to numpy array for plotting
            classes = sorted(list(set(y_true + y_pred)))
            cm_array = np.array([[cm[actual][predicted] for predicted in classes] for actual in classes])
            
            # Plot confusion matrix
            im = axes[i].imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].figure.colorbar(im, ax=axes[i])
            
            # Set labels
            axes[i].set(xticks=np.arange(cm_array.shape[1]),
                       yticks=np.arange(cm_array.shape[0]),
                       xticklabels=classes, yticklabels=classes,
                       title=f'{model_name} Confusion Matrix',
                       ylabel='True Label',
                       xlabel='Predicted Label')
            
            # Add text annotations
            thresh = cm_array.max() / 2.
            for row in range(cm_array.shape[0]):
                for col in range(cm_array.shape[1]):
                    axes[i].text(col, row, format(cm_array[row, col], 'd'),
                               ha="center", va="center",
                               color="white" if cm_array[row, col] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrices saved as 'confusion_matrices.png'")
    
    def plot_metrics_comparison(self, model_metrics: Dict[str, Dict[str, float]]):
        """
        Plot metrics comparison bar chart.
        
        Args:
            model_metrics: Dictionary mapping model names to their metrics
        """
        models = list(model_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, metric in enumerate(metrics):
            values = [model_metrics[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            values = [model_metrics[model][metric] for model in models]
            for j, v in enumerate(values):
                ax.text(j + i * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Metrics comparison saved as 'metrics_comparison.png'")
    
    def plot_class_distribution(self, y_true: List[str], title: str = "Class Distribution"):
        """
        Plot class distribution pie chart.
        
        Args:
            y_true: True labels
            title: Plot title
        """
        class_counts = {}
        for label in y_true:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        labels = list(class_counts.keys())
        sizes = list(class_counts.values())
        colors = ['lightblue', 'lightcoral']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Class distribution saved as 'class_distribution.png'")
