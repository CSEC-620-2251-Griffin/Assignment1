"""
Text processing utilities for SMS spam classification.

This module contains the TextTokenizer class for preprocessing text
without using third-party libraries.
"""

import re
from typing import List, Dict, Set
from utils.data_loader import SMSMessage


class TextTokenizer:
    """
    Text tokenizer for SMS messages.
    
    Implements basic tokenization strategy without third-party dependencies.
    """
    
    def __init__(self, clean_text: bool = True):
        """
        Initialize the tokenizer.
        
        Args:
            clean_text: Whether to apply basic text cleaning
        """
        self.clean_text = clean_text
    
    def clean_message(self, text: str) -> str:
        """
        Apply basic text cleaning to improve token quality.
        
        Args:
            text: Raw message text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove punctuation (keep only letters, numbers, and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize_message(self, text: str) -> List[str]:
        """
        Tokenize a single message into words.
        
        Args:
            text: Message text
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
        
        # Apply cleaning if enabled
        if self.clean_text:
            text = self.clean_message(text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def tokenize_messages(self, messages: List[SMSMessage]) -> List[Dict]:
        """
        Tokenize a list of messages.
        
        Args:
            messages: List of SMS messages
            
        Returns:
            List of dictionaries with 'tokens' and 'label' keys
        """
        tokenized_data = []
        
        for message in messages:
            tokens = self.tokenize_message(message.text)
            tokenized_data.append({
                'tokens': tokens,
                'label': message.label
            })
        
        return tokenized_data
    
    def get_vocabulary(self, tokenized_data: List[Dict]) -> Set[str]:
        """
        Extract vocabulary from tokenized data.
        
        Args:
            tokenized_data: List of tokenized documents
            
        Returns:
            Set of unique tokens
        """
        vocabulary = set()
        
        for doc in tokenized_data:
            vocabulary.update(doc['tokens'])
        
        return vocabulary
    
    def get_vocabulary_stats(self, tokenized_data: List[Dict]) -> Dict:
        """
        Get vocabulary statistics.
        
        Args:
            tokenized_data: List of tokenized documents
            
        Returns:
            Dictionary with vocabulary statistics
        """
        vocabulary = self.get_vocabulary(tokenized_data)
        
        # Count total tokens
        total_tokens = sum(len(doc['tokens']) for doc in tokenized_data)
        
        # Count tokens per class
        ham_tokens = 0
        spam_tokens = 0
        
        for doc in tokenized_data:
            if doc['label'] == 'ham':
                ham_tokens += len(doc['tokens'])
            else:
                spam_tokens += len(doc['tokens'])
        
        return {
            'vocabulary_size': len(vocabulary),
            'total_tokens': total_tokens,
            'ham_tokens': ham_tokens,
            'spam_tokens': spam_tokens,
            'avg_tokens_per_message': total_tokens / len(tokenized_data) if tokenized_data else 0
        }
