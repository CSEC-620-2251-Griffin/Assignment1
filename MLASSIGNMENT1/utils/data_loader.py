"""
Data loading utilities for SMS spam classification.

This module contains the DataLoader class and related data structures
for loading and validating SMS message data.
"""

import os
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SMSMessage:
    """Data class for SMS message."""
    label: str
    text: str
    
    def __post_init__(self):
        """Validate message data after initialization."""
        if self.label not in ['ham', 'spam']:
            raise ValueError(f"Invalid label: {self.label}. Must be 'ham' or 'spam'")
        
        if not isinstance(self.text, str) or len(self.text.strip()) == 0:
            raise ValueError("Message text cannot be empty")


@dataclass
class DatasetInfo:
    """Container for dataset statistics and metadata."""
    total_messages: int
    ham_count: int
    spam_count: int
    ham_percentage: float
    spam_percentage: float
    file_path: str


class DataLoader:
    """
    Data loader for SMS spam dataset.
    """
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data file
        """
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "SMSSpamCollection")
    
    def load_data(self) -> Tuple[List[SMSMessage], DatasetInfo]:
        """
        Load SMS data from file.
        
        Returns:
            Tuple of (messages_list, dataset_info)
        """
        print(f"Loading data from: {self.data_file}")
        
        messages = []
        ham_count = 0
        spam_count = 0
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split on first tab only (in case message contains tabs)
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        print(f"Warning: Invalid format at line {line_number}: {line[:50]}...")
                        continue
                    
                    label, text = parts
                    
                    # Validate and clean
                    label = label.lower().strip()
                    if label not in ['ham', 'spam']:
                        print(f"Warning: Invalid label '{label}' at line {line_number}")
                        continue
                    
                    text = text.strip()
                    if not text:
                        print(f"Warning: Empty text at line {line_number}")
                        continue
                    
                    # Create message object
                    message = SMSMessage(label=label, text=text)
                    messages.append(message)
                    
                    # Count classes
                    if label == 'ham':
                        ham_count += 1
                    else:
                        spam_count += 1
            
            # Validate loaded data
            if not messages:
                raise ValueError("No valid messages found in file")
            
            total_messages = len(messages)
            ham_percentage = (ham_count / total_messages) * 100
            spam_percentage = (spam_count / total_messages) * 100
            
            # Create dataset info
            dataset_info = DatasetInfo(
                total_messages=total_messages,
                ham_count=ham_count,
                spam_count=spam_count,
                ham_percentage=ham_percentage,
                spam_percentage=spam_percentage,
                file_path=self.data_file
            )
            
            print(f"Successfully loaded {total_messages} messages")
            print(f"Ham: {ham_count} ({ham_percentage:.1f}%), Spam: {spam_count} ({spam_percentage:.1f}%)")
            
            return messages, dataset_info
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
