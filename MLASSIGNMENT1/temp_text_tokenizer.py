class TextTokenizer:
    """
    Simple text tokenizer for SMS messages.
    
    Implements basic tokenization strategy:
    - Split messages into space-separated words
    - Basic text cleaning to improve token quality
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
            text: Message text to tokenize
            
        Returns:
            List of tokens (words)
        """
        if self.clean_text:
            text = self.clean_message(text)
        
        # Split into space-separated words
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def tokenize_messages(self, messages: List[SMSMessage]) -> List[Dict[str, any]]:
        """
        Tokenize a list of SMS messages.
        
        Args:
            messages: List of SMSMessage objects
            
        Returns:
            List of dictionaries with tokenized data
        """
        tokenized_data = []
        
        for message in messages:
            tokens = self.tokenize_message(message.text)
            
            tokenized_data.append({
                'label': message.label,
                'original_text': message.text,
                'tokens': tokens,
                'token_count': len(tokens)
            })
        
        return tokenized_data
    
    def get_vocabulary(self, tokenized_data: List[Dict[str, any]]) -> List[str]:
        """
        Extract unique vocabulary from tokenized data.
        
        Args:
            tokenized_data: List of tokenized message dictionaries
            
        Returns:
            Sorted list of unique words
        """
        vocabulary = set()
        
        for data in tokenized_data:
            vocabulary.update(data['tokens'])
        
        return sorted(list(vocabulary))
    
    def print_tokenization_summary(self, tokenized_data: List[Dict[str, any]]):
        """
        Print a summary of the tokenization results.
        
        Args:
            tokenized_data: List of tokenized message dictionaries
        """
        vocabulary = self.get_vocabulary(tokenized_data)
        
        # Count total tokens
        total_tokens = sum(data['token_count'] for data in tokenized_data)
        
        # Count tokens per message
        token_counts = [data['token_count'] for data in tokenized_data]
        
        print("\n" + "="*50)
        print("TOKENIZATION SUMMARY")
        print("="*50)
        print(f"Total Messages: {len(tokenized_data)}")
        print(f"Total Tokens: {total_tokens}")
        print(f"Vocabulary Size: {len(vocabulary)}")
        print(f"Average Tokens per Message: {total_tokens / len(tokenized_data):.1f}")
        print(f"Min Tokens per Message: {min(token_counts)}")
        print(f"Max Tokens per Message: {max(token_counts)}")
        
        # Show sample tokenized messages
        print(f"\nSample Tokenized Messages:")
        for i, data in enumerate(tokenized_data[:3]):
            print(f"{i+1}. [{data['label'].upper()}]")
            print(f"   Original: {data['original_text']}")
            print(f"   Tokens: {data['tokens']}")
            print(f"   Count: {data['token_count']}")


