import os
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import re


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
    
    def create_train_test_split(self, 
                              messages: List[SMSMessage], 
                              test_size: float = 0.2,
                              random_state: int = 42,
                              stratify: bool = True) -> Tuple[List[SMSMessage], List[SMSMessage]]:
        """
        Create train-test split with optional stratification.
        
        Args:
            messages: List of SMS messages
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            stratify: Whether to maintain class distribution
            
        Returns:
            Tuple of (train_messages, test_messages)
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Extract labels for stratification
        labels = [msg.label for msg in messages]
        
        if stratify:
            # Use sklearn's train_test_split for stratification
            train_indices, test_indices = train_test_split(
                range(len(messages)),
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
        else:
            # Simple random split
            train_indices, test_indices = train_test_split(
                range(len(messages)),
                test_size=test_size,
                random_state=random_state
            )
        
        train_messages = [messages[i] for i in train_indices]
        test_messages = [messages[i] for i in test_indices]
        
        print(f"Created train-test split: {len(train_messages)} train, {len(test_messages)} test")
        
        return train_messages, test_messages


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


class DataSplitter:
    """
    Enhanced train-test splitting system for SMS spam classification.
    
    Features:
    - Stratified splitting to maintain class distribution
    - Multiple splitting strategies
    - Validation set creation
    - Cross-validation support
    - Statistical analysis of splits
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    def create_stratified_split(self, 
                              messages: List[SMSMessage], 
                              test_size: float = 0.2,
                              validation_size: float = 0.0) -> Dict[str, List[SMSMessage]]:
        """
        Create stratified train-test split maintaining class distribution.
        
        Args:
            messages: List of SMS messages
            test_size: Proportion of data for testing (0.0 to 1.0)
            validation_size: Proportion of training data for validation (0.0 to 1.0)
            
        Returns:
            Dictionary with 'train', 'test', and optionally 'validation' splits
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        
        if validation_size < 0 or validation_size >= 1:
            raise ValueError("validation_size must be between 0 and 1")
        
        # Extract labels for stratification
        labels = [msg.label for msg in messages]
        
        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            range(len(messages)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        # Create train and test sets
        train_val_messages = [messages[i] for i in train_val_indices]
        test_messages = [messages[i] for i in test_indices]
        
        result = {
            'test': test_messages
        }
        
        # Second split: separate validation set if requested
        if validation_size > 0:
            train_val_labels = [msg.label for msg in train_val_messages]
            train_indices, val_indices = train_test_split(
                range(len(train_val_messages)),
                test_size=validation_size,
                random_state=self.random_state,
                stratify=train_val_labels
            )
            
            train_messages = [train_val_messages[i] for i in train_indices]
            val_messages = [train_val_messages[i] for i in val_indices]
            
            result['train'] = train_messages
            result['validation'] = val_messages
        else:
            result['train'] = train_val_messages
        
        return result
    
    def create_simple_split(self, 
                          messages: List[SMSMessage], 
                          test_size: float = 0.2) -> Dict[str, List[SMSMessage]]:
        """
        Create simple random train-test split without stratification.
        
        Args:
            messages: List of SMS messages
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        train_indices, test_indices = train_test_split(
            range(len(messages)),
            test_size=test_size,
            random_state=self.random_state
        )
        
        train_messages = [messages[i] for i in train_indices]
        test_messages = [messages[i] for i in test_indices]
        
        return {
            'train': train_messages,
            'test': test_messages
        }
    
    def analyze_split_distribution(self, splits: Dict[str, List[SMSMessage]]) -> Dict[str, Dict[str, int]]:
        """
        Analyze the class distribution in each split.
        
        Args:
            splits: Dictionary containing train/test/validation splits
            
        Returns:
            Dictionary with class distribution statistics for each split
        """
        analysis = {}
        
        for split_name, messages in splits.items():
            ham_count = sum(1 for msg in messages if msg.label == 'ham')
            spam_count = sum(1 for msg in messages if msg.label == 'spam')
            total_count = len(messages)
            
            analysis[split_name] = {
                'total': total_count,
                'ham': ham_count,
                'spam': spam_count,
                'ham_percentage': (ham_count / total_count * 100) if total_count > 0 else 0,
                'spam_percentage': (spam_count / total_count * 100) if total_count > 0 else 0
            }
        
        return analysis
    
    def print_split_analysis(self, splits: Dict[str, List[SMSMessage]]):
        """
        Print detailed analysis of the data splits.
        
        Args:
            splits: Dictionary containing train/test/validation splits
        """
        analysis = self.analyze_split_distribution(splits)
        
        print("\n" + "="*60)
        print("DATA SPLIT ANALYSIS")
        print("="*60)
        
        for split_name, stats in analysis.items():
            print(f"\n{split_name.upper()} SET:")
            print(f"  Total Messages: {stats['total']}")
            print(f"  Ham Messages: {stats['ham']} ({stats['ham_percentage']:.1f}%)")
            print(f"  Spam Messages: {stats['spam']} ({stats['spam_percentage']:.1f}%)")
        
        # Check if class distribution is maintained
        if 'train' in analysis and 'test' in analysis:
            train_ham_pct = analysis['train']['ham_percentage']
            test_ham_pct = analysis['test']['ham_percentage']
            ham_diff = abs(train_ham_pct - test_ham_pct)
            
            print(f"\nClass Distribution Consistency:")
            print(f"  Ham percentage difference (train vs test): {ham_diff:.1f}%")
            
            if ham_diff < 2.0:
                print("  ✓ Good: Class distribution is well maintained")
            elif ham_diff < 5.0:
                print("  ⚠ Warning: Moderate class distribution difference")
            else:
                print("  ✗ Poor: Significant class distribution difference")
    
    def create_cross_validation_splits(self, 
                                     messages: List[SMSMessage], 
                                     n_folds: int = 5) -> List[Dict[str, List[SMSMessage]]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            messages: List of SMS messages
            n_folds: Number of folds for cross-validation
            
        Returns:
            List of dictionaries, each containing train/test splits for one fold
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        
        from sklearn.model_selection import StratifiedKFold
        
        # Extract labels for stratification
        labels = [msg.label for msg in messages]
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        cv_splits = []
        
        for fold, (train_indices, test_indices) in enumerate(skf.split(messages, labels)):
            train_messages = [messages[i] for i in train_indices]
            test_messages = [messages[i] for i in test_indices]
            
            cv_splits.append({
                'fold': fold + 1,
                'train': train_messages,
                'test': test_messages
            })
        
        return cv_splits
    
    def print_cv_analysis(self, cv_splits: List[Dict[str, List[SMSMessage]]]):
        """
        Print analysis of cross-validation splits.
        
        Args:
            cv_splits: List of cross-validation splits
        """
        print(f"\n" + "="*60)
        print(f"CROSS-VALIDATION ANALYSIS ({len(cv_splits)} folds)")
        print("="*60)
        
        for split in cv_splits:
            fold_num = split['fold']
            train_messages = split['train']
            test_messages = split['test']
            
            train_ham = sum(1 for msg in train_messages if msg.label == 'ham')
            train_spam = sum(1 for msg in train_messages if msg.label == 'spam')
            test_ham = sum(1 for msg in test_messages if msg.label == 'ham')
            test_spam = sum(1 for msg in test_messages if msg.label == 'spam')
            
            print(f"\nFold {fold_num}:")
            print(f"  Train: {len(train_messages)} messages ({train_ham} ham, {train_spam} spam)")
            print(f"  Test:  {len(test_messages)} messages ({test_ham} ham, {test_spam} spam)")


def demonstrate_tokenization():
    """Demonstrate the tokenization functionality."""
    print("="*60)
    print("TOKENIZATION DEMONSTRATION")
    print("="*60)
    
    # Create sample messages
    sample_messages = [
        SMSMessage("ham", "Hey! How are you doing today?"),
        SMSMessage("spam", "WINNER!! You've won a £1000 prize! Call now!"),
        SMSMessage("ham", "Don't forget our meeting at 3pm"),
        SMSMessage("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts")
    ]
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(clean_text=True)
    
    # Tokenize messages
    tokenized_data = tokenizer.tokenize_messages(sample_messages)
    
    # Print summary
    tokenizer.print_tokenization_summary(tokenized_data)
    
    return tokenized_data


def demonstrate_data_splitting():
    """Demonstrate the data splitting functionality."""
    print("="*60)
    print("DATA SPLITTING DEMONSTRATION")
    print("="*60)
    
    # Create sample messages with known distribution
    sample_messages = []
    
    # Add 80 ham messages
    for i in range(80):
        sample_messages.append(SMSMessage("ham", f"Ham message number {i+1}"))
    
    # Add 20 spam messages
    for i in range(20):
        sample_messages.append(SMSMessage("spam", f"Spam message number {i+1}"))
    
    print(f"Created {len(sample_messages)} sample messages (80 ham, 20 spam)")
    
    # Initialize splitter
    splitter = DataSplitter(random_state=42)
    
    # Test stratified split
    print(f"\n1. STRATIFIED SPLIT (80/20 train/test):")
    stratified_splits = splitter.create_stratified_split(sample_messages, test_size=0.2)
    splitter.print_split_analysis(stratified_splits)
    
    # Test split with validation set
    print(f"\n2. STRATIFIED SPLIT WITH VALIDATION (60/20/20 train/val/test):")
    val_splits = splitter.create_stratified_split(sample_messages, test_size=0.2, validation_size=0.25)
    splitter.print_split_analysis(val_splits)
    
    # Test cross-validation
    print(f"\n3. CROSS-VALIDATION (5 folds):")
    cv_splits = splitter.create_cross_validation_splits(sample_messages, n_folds=5)
    splitter.print_cv_analysis(cv_splits)
    
    return stratified_splits, val_splits, cv_splits


def main_with_tokenization():
    """Main function that includes data loading and tokenization."""
    try:
        # Initialize data loader
        loader = DataLoader()
        
        # Load data
        messages, dataset_info = loader.load_data()
        
        # Print dataset information
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Total Messages: {dataset_info.total_messages}")
        print(f"Ham Messages: {dataset_info.ham_count} ({dataset_info.ham_percentage:.1f}%)")
        print(f"Spam Messages: {dataset_info.spam_count} ({dataset_info.spam_percentage:.1f}%)")
        
        # Initialize tokenizer
        tokenizer = TextTokenizer(clean_text=True)
        
        # Tokenize all messages
        print("\nTokenizing messages...")
        tokenized_data = tokenizer.tokenize_messages(messages)
        
        # Print tokenization summary
        tokenizer.print_tokenization_summary(tokenized_data)
        
        # Create train-test split on tokenized data
        print("\nCreating train-test split...")
        train_messages, test_messages = loader.create_train_test_split(messages)
        
        # Tokenize train and test sets separately
        train_tokenized = tokenizer.tokenize_messages(train_messages)
        test_tokenized = tokenizer.tokenize_messages(test_messages)
        
        print(f"\nTrain-Test Split (Tokenized):")
        print(f"Training Messages: {len(train_tokenized)}")
        print(f"Test Messages: {len(test_tokenized)}")
        
        # Get vocabulary from training set only
        train_vocabulary = tokenizer.get_vocabulary(train_tokenized)
        print(f"Training Vocabulary Size: {len(train_vocabulary)}")
        
        return {
            'messages': messages,
            'dataset_info': dataset_info,
            'tokenized_data': tokenized_data,
            'train_tokenized': train_tokenized,
            'test_tokenized': test_tokenized,
            'train_vocabulary': train_vocabulary
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def main_with_splitting():
    """Main function that includes data loading, tokenization, and splitting."""
    try:
        # Initialize components
        loader = DataLoader()
        tokenizer = TextTokenizer(clean_text=True)
        splitter = DataSplitter(random_state=42)
        
        # Load data
        messages, dataset_info = loader.load_data()
        
        # Print dataset information
        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)
        print(f"Total Messages: {dataset_info.total_messages}")
        print(f"Ham Messages: {dataset_info.ham_count} ({dataset_info.ham_percentage:.1f}%)")
        print(f"Spam Messages: {dataset_info.spam_count} ({dataset_info.spam_percentage:.1f}%)")
        
        # Tokenize messages
        print("\nTokenizing messages...")
        tokenized_data = tokenizer.tokenize_messages(messages)
        tokenizer.print_tokenization_summary(tokenized_data)
        
        # Create stratified train-test split
        print("\nCreating stratified train-test split...")
        splits = splitter.create_stratified_split(messages, test_size=0.2)
        splitter.print_split_analysis(splits)
        
        # Tokenize the splits
        train_tokenized = tokenizer.tokenize_messages(splits['train'])
        test_tokenized = tokenizer.tokenize_messages(splits['test'])
        
        # Get vocabulary from training set
        train_vocabulary = tokenizer.get_vocabulary(train_tokenized)
        
        print(f"\nFinal Results:")
        print(f"Training Set: {len(train_tokenized)} messages, {len(train_vocabulary)} unique words")
        print(f"Test Set: {len(test_tokenized)} messages")
        
        return {
            'messages': messages,
            'dataset_info': dataset_info,
            'splits': splits,
            'train_tokenized': train_tokenized,
            'test_tokenized': test_tokenized,
            'train_vocabulary': train_vocabulary
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None


class TestSuite:
    """
    Comprehensive test suite for SMS spam classification components.
    
    Tests:
    - Data loading functionality
    - Text tokenization
    - Train-test splitting
    - Integration between components
    """
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_test(self, test_name: str, test_function):
        """
        Run a single test and record results.
        
        Args:
            test_name: Name of the test
            test_function: Function that returns True if test passes
        """
        try:
            result = test_function()
            if result:
                print(f"✓ PASS: {test_name}")
                self.passed_tests += 1
            else:
                print(f"✗ FAIL: {test_name}")
                self.failed_tests += 1
        except Exception as e:
            print(f"✗ ERROR: {test_name} - {str(e)}")
            self.failed_tests += 1
        
        self.test_results.append({
            'name': test_name,
            'passed': result if 'result' in locals() else False,
            'error': str(e) if 'e' in locals() else None
        })
    
    def test_data_loading(self):
        """Test data loading functionality."""
        print("\n" + "="*60)
        print("TESTING DATA LOADING")
        print("="*60)
        
        def test_file_exists():
            """Test that data file exists."""
            loader = DataLoader()
            return os.path.exists(loader.data_file)
        
        def test_load_data():
            """Test that data can be loaded successfully."""
            loader = DataLoader()
            messages, dataset_info = loader.load_data()
            return (len(messages) > 0 and 
                   dataset_info.total_messages > 0 and
                   dataset_info.ham_count > 0 and
                   dataset_info.spam_count > 0)
        
        def test_message_validation():
            """Test that loaded messages are properly validated."""
            loader = DataLoader()
            messages, _ = loader.load_data()
            
            # Check that all messages have valid labels
            valid_labels = all(msg.label in ['ham', 'spam'] for msg in messages)
            
            # Check that all messages have non-empty text
            non_empty_text = all(len(msg.text.strip()) > 0 for msg in messages)
            
            return valid_labels and non_empty_text
        
        def test_dataset_statistics():
            """Test that dataset statistics are reasonable."""
            loader = DataLoader()
            messages, dataset_info = loader.load_data()
            
            # Check that percentages add up to 100
            total_percentage = dataset_info.ham_percentage + dataset_info.spam_percentage
            percentage_correct = abs(total_percentage - 100.0) < 0.1
            
            # Check that counts match total
            count_correct = (dataset_info.ham_count + dataset_info.spam_count == 
                           dataset_info.total_messages)
            
            return percentage_correct and count_correct
        
        # Run data loading tests
        self.run_test("Data file exists", test_file_exists)
        self.run_test("Data loads successfully", test_load_data)
        self.run_test("Message validation works", test_message_validation)
        self.run_test("Dataset statistics are correct", test_dataset_statistics)
    
    def test_tokenization(self):
        """Test text tokenization functionality."""
        print("\n" + "="*60)
        print("TESTING TOKENIZATION")
        print("="*60)
        
        def test_basic_tokenization():
            """Test basic tokenization functionality."""
            tokenizer = TextTokenizer(clean_text=True)
            test_text = "Hello world! This is a test message."
            tokens = tokenizer.tokenize_message(test_text)
            
            # The current tokenizer keeps punctuation attached to words
            expected_tokens = ["hello", "world!", "this", "is", "a", "test", "message."]
            return tokens == expected_tokens
        
        def test_empty_text_handling():
            """Test handling of empty or invalid text."""
            tokenizer = TextTokenizer(clean_text=True)
            
            # Test empty string
            empty_tokens = tokenizer.tokenize_message("")
            empty_handled = len(empty_tokens) == 0
            
            # Test whitespace only
            whitespace_tokens = tokenizer.tokenize_message("   \n\t  ")
            whitespace_handled = len(whitespace_tokens) == 0
            
            return empty_handled and whitespace_handled
        
        def test_text_cleaning():
            """Test text cleaning functionality."""
            tokenizer = TextTokenizer(clean_text=True)
            
            # Test case conversion
            cleaned = tokenizer.clean_message("HELLO World!")
            case_converted = cleaned == "hello world!"
            
            # Test whitespace normalization
            whitespace_cleaned = tokenizer.clean_message("hello    world\n\n")
            whitespace_normalized = whitespace_cleaned == "hello world"
            
            return case_converted and whitespace_normalized
        
        def test_message_batch_processing():
            """Test processing multiple messages."""
            tokenizer = TextTokenizer(clean_text=True)
            
            test_messages = [
                SMSMessage("ham", "Hello world"),
                SMSMessage("spam", "Free money now!"),
                SMSMessage("ham", "How are you?")
            ]
            
            tokenized_data = tokenizer.tokenize_messages(test_messages)
            
            # Check that all messages were processed
            all_processed = len(tokenized_data) == len(test_messages)
            
            # Check that each message has required fields
            required_fields = all(
                all(key in data for key in ['label', 'original_text', 'tokens', 'token_count'])
                for data in tokenized_data
            )
            
            return all_processed and required_fields
        
        def test_vocabulary_extraction():
            """Test vocabulary extraction."""
            tokenizer = TextTokenizer(clean_text=True)
            
            test_messages = [
                SMSMessage("ham", "hello world"),
                SMSMessage("spam", "hello money"),
                SMSMessage("ham", "world test")
            ]
            
            tokenized_data = tokenizer.tokenize_messages(test_messages)
            vocabulary = tokenizer.get_vocabulary(tokenized_data)
            
            # Should have unique words: hello, world, money, test
            expected_vocab = ["hello", "money", "test", "world"]
            return sorted(vocabulary) == expected_vocab
        
        # Run tokenization tests
        self.run_test("Basic tokenization works", test_basic_tokenization)
        self.run_test("Empty text handling", test_empty_text_handling)
        self.run_test("Text cleaning works", test_text_cleaning)
        self.run_test("Batch message processing", test_message_batch_processing)
        self.run_test("Vocabulary extraction", test_vocabulary_extraction)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        print("\n" + "="*60)
        print("TESTING DATA SPLITTING")
        print("="*60)
        
        def test_stratified_split():
            """Test stratified train-test splitting."""
            # Create test data with known distribution
            test_messages = []
            for i in range(80):
                test_messages.append(SMSMessage("ham", f"ham message {i}"))
            for i in range(20):
                test_messages.append(SMSMessage("spam", f"spam message {i}"))
            
            splitter = DataSplitter(random_state=42)
            splits = splitter.create_stratified_split(test_messages, test_size=0.2)
            
            # Check that we have train and test sets
            has_train_test = 'train' in splits and 'test' in splits
            
            # Check that split sizes are correct
            total_messages = len(test_messages)
            expected_test_size = int(total_messages * 0.2)
            expected_train_size = total_messages - expected_test_size
            
            correct_sizes = (len(splits['train']) == expected_train_size and 
                           len(splits['test']) == expected_test_size)
            
            return has_train_test and correct_sizes
        
        def test_class_distribution_preservation():
            """Test that class distribution is preserved in splits."""
            # Create test data with 80% ham, 20% spam
            test_messages = []
            for i in range(80):
                test_messages.append(SMSMessage("ham", f"ham message {i}"))
            for i in range(20):
                test_messages.append(SMSMessage("spam", f"spam message {i}"))
            
            splitter = DataSplitter(random_state=42)
            splits = splitter.create_stratified_split(test_messages, test_size=0.2)
            
            # Calculate class distributions
            train_ham = sum(1 for msg in splits['train'] if msg.label == 'ham')
            train_spam = sum(1 for msg in splits['train'] if msg.label == 'spam')
            test_ham = sum(1 for msg in splits['test'] if msg.label == 'ham')
            test_spam = sum(1 for msg in splits['test'] if msg.label == 'spam')
            
            train_ham_pct = train_ham / len(splits['train']) * 100
            test_ham_pct = test_ham / len(splits['test']) * 100
            
            # Class distribution should be similar (within 5%)
            distribution_preserved = abs(train_ham_pct - test_ham_pct) < 5.0
            
            return distribution_preserved
        
        def test_validation_split():
            """Test train/validation/test splitting."""
            # Create test data
            test_messages = []
            for i in range(100):
                test_messages.append(SMSMessage("ham", f"ham message {i}"))
            for i in range(25):
                test_messages.append(SMSMessage("spam", f"spam message {i}"))
            
            splitter = DataSplitter(random_state=42)
            splits = splitter.create_stratified_split(test_messages, test_size=0.2, validation_size=0.25)
            
            # Check that we have all three sets
            has_all_sets = all(key in splits for key in ['train', 'validation', 'test'])
            
            # Check that all messages are accounted for
            total_split = (len(splits['train']) + len(splits['validation']) + len(splits['test']))
            all_accounted = total_split == len(test_messages)
            
            return has_all_sets and all_accounted
        
        def test_cross_validation():
            """Test cross-validation splitting."""
            # Create test data
            test_messages = []
            for i in range(50):
                test_messages.append(SMSMessage("ham", f"ham message {i}"))
            for i in range(10):
                test_messages.append(SMSMessage("spam", f"spam message {i}"))
            
            splitter = DataSplitter(random_state=42)
            cv_splits = splitter.create_cross_validation_splits(test_messages, n_folds=5)
            
            # Check that we have 5 folds
            correct_fold_count = len(cv_splits) == 5
            
            # Check that each fold has train and test sets
            all_folds_valid = all(
                'train' in fold and 'test' in fold and 'fold' in fold
                for fold in cv_splits
            )
            
            return correct_fold_count and all_folds_valid
        
        # Run splitting tests
        self.run_test("Stratified split works", test_stratified_split)
        self.run_test("Class distribution preserved", test_class_distribution_preservation)
        self.run_test("Validation split works", test_validation_split)
        self.run_test("Cross-validation works", test_cross_validation)
    
    def test_integration(self):
        """Test integration between all components."""
        print("\n" + "="*60)
        print("TESTING INTEGRATION")
        print("="*60)
        
        def test_full_pipeline():
            """Test the complete pipeline from loading to splitting."""
            try:
                # Load data
                loader = DataLoader()
                messages, dataset_info = loader.load_data()
                
                # Tokenize
                tokenizer = TextTokenizer(clean_text=True)
                tokenized_data = tokenizer.tokenize_messages(messages)
                
                # Split data
                splitter = DataSplitter(random_state=42)
                splits = splitter.create_stratified_split(messages, test_size=0.2)
                
                # Tokenize splits
                train_tokenized = tokenizer.tokenize_messages(splits['train'])
                test_tokenized = tokenizer.tokenize_messages(splits['test'])
                
                # Get vocabulary
                train_vocabulary = tokenizer.get_vocabulary(train_tokenized)
                
                # Check that everything worked
                pipeline_success = (
                    len(messages) > 0 and
                    len(tokenized_data) > 0 and
                    len(splits['train']) > 0 and
                    len(splits['test']) > 0 and
                    len(train_tokenized) > 0 and
                    len(test_tokenized) > 0 and
                    len(train_vocabulary) > 0
                )
                
                return pipeline_success
                
            except Exception as e:
                print(f"Integration test error: {e}")
                return False
        
        def test_data_consistency():
            """Test that data remains consistent through the pipeline."""
            try:
                # Load and process data
                loader = DataLoader()
                messages, _ = loader.load_data()
                
                # Take a small sample for testing
                sample_messages = messages[:100]
                
                # Process through pipeline
                tokenizer = TextTokenizer(clean_text=True)
                splitter = DataSplitter(random_state=42)
                
                splits = splitter.create_stratified_split(sample_messages, test_size=0.2)
                train_tokenized = tokenizer.tokenize_messages(splits['train'])
                
                # Check that labels are preserved
                original_labels = [msg.label for msg in splits['train']]
                tokenized_labels = [data['label'] for data in train_tokenized]
                
                labels_preserved = original_labels == tokenized_labels
                
                return labels_preserved
                
            except Exception as e:
                print(f"Consistency test error: {e}")
                return False
        
        # Run integration tests
        self.run_test("Full pipeline works", test_full_pipeline)
        self.run_test("Data consistency maintained", test_data_consistency)
    
    def run_all_tests(self):
        """Run all tests and print summary."""
        print("="*80)
        print("RUNNING COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        # Run all test categories
        self.test_data_loading()
        self.test_tokenization()
        self.test_data_splitting()
        self.test_integration()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.passed_tests + self.failed_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests) * 100):.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Your implementation is working correctly.")
        else:
            print(f"\n⚠️  {self.failed_tests} tests failed. Please review the implementation.")
        
        return self.failed_tests == 0


def run_quick_demo():
    """Run a quick demonstration of all functionality."""
    print("="*80)
    print("QUICK DEMONSTRATION")
    print("="*80)
    
    try:
        # Initialize components
        loader = DataLoader()
        tokenizer = TextTokenizer(clean_text=True)
        splitter = DataSplitter(random_state=42)
        
        # Load data
        print("1. Loading data...")
        messages, dataset_info = loader.load_data()
        print(f"   ✓ Loaded {dataset_info.total_messages} messages")
        print(f"   ✓ Ham: {dataset_info.ham_count} ({dataset_info.ham_percentage:.1f}%)")
        print(f"   ✓ Spam: {dataset_info.spam_count} ({dataset_info.spam_percentage:.1f}%)")
        
        # Tokenize
        print("\n2. Tokenizing messages...")
        tokenized_data = tokenizer.tokenize_messages(messages)
        vocabulary = tokenizer.get_vocabulary(tokenized_data)
        print(f"   ✓ Tokenized {len(tokenized_data)} messages")
        print(f"   ✓ Vocabulary size: {len(vocabulary)}")
        
        # Split data
        print("\n3. Creating train-test split...")
        splits = splitter.create_stratified_split(messages, test_size=0.2)
        splitter.print_split_analysis(splits)
        
        # Final summary
        print(f"\n4. Final Results:")
        print(f"   ✓ Training set: {len(splits['train'])} messages")
        print(f"   ✓ Test set: {len(splits['test'])} messages")
        print(f"   ✓ Ready for classifier implementation!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    test_suite = TestSuite()
    all_tests_passed = test_suite.run_all_tests()
    
    if all_tests_passed:
        print("\n" + "="*80)
        print("RUNNING QUICK DEMO")
        print("="*80)
        run_quick_demo()