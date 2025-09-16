"""
Data preprocessing utilities for SMS spam classification.

This module contains the DataSplitter class for creating train-test splits
without using third-party libraries.
"""

import random
from typing import List, Dict
from collections import defaultdict
from utils.data_loader import SMSMessage


class DataSplitter:
    """
    Train-test splitting system for SMS spam classification.
    
    Features:
    - Stratified splitting to maintain class distribution
    - No third-party dependencies
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
    
    def create_stratified_split(self, 
                              messages: List[SMSMessage], 
                              test_size: float = 0.2) -> Dict[str, List[SMSMessage]]:
        """
        Create stratified train-test split maintaining class distribution.
        
        Args:
            messages: List of SMS messages
            test_size: Proportion of data for testing (0.0 to 1.0)
            
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        # Group messages by label
        label_groups = defaultdict(list)
        for message in messages:
            label_groups[message.label].append(message)
        
        train_messages = []
        test_messages = []
        
        # Split each class separately to maintain distribution
        for label, class_messages in label_groups.items():
            # Shuffle the messages for this class
            shuffled_messages = class_messages.copy()
            random.shuffle(shuffled_messages)
            
            # Calculate split point
            n_test = int(len(shuffled_messages) * test_size)
            
            # Split the messages
            test_messages.extend(shuffled_messages[:n_test])
            train_messages.extend(shuffled_messages[n_test:])
        
        # Shuffle the final splits
        random.shuffle(train_messages)
        random.shuffle(test_messages)
        
        return {
            'train': train_messages,
            'test': test_messages
        }
    
    def print_split_analysis(self, splits: Dict[str, List[SMSMessage]]):
        """
        Print analysis of the data splits.
        
        Args:
            splits: Dictionary containing train and test splits
        """
        print("Data Split Analysis:")
        print("-" * 30)
        
        for split_name, messages in splits.items():
            total = len(messages)
            ham_count = sum(1 for msg in messages if msg.label == 'ham')
            spam_count = total - ham_count
            
            ham_pct = (ham_count / total * 100) if total > 0 else 0
            spam_pct = (spam_count / total * 100) if total > 0 else 0
            
            print(f"{split_name.capitalize()} Set:")
            print(f"  Total: {total}")
            print(f"  Ham: {ham_count} ({ham_pct:.1f}%)")
            print(f"  Spam: {spam_count} ({spam_pct:.1f}%)")
            print()
