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


