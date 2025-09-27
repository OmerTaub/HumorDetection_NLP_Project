#!/usr/bin/env python3
"""
Raw Data Preprocessing Script
============================

This script combines three CSV data sources:
1. shortjokes.csv - Contains jokes (all humorous, labeled as 1)
2. puns_pos_neg_data.csv - Contains labeled puns/sayings (1 for funny, -1 for not funny)
3. kaggle_dataset.csv - Contains mixed humor data (take only non-humorous entries)

The script:
- Loads and deduplicates data from all sources
- Combines all sources and performs global deduplication
- Creates contamination-free train/validation/test splits (0.7/0.15/0.15)
- Balances the test set between positive and negative samples
- Verifies no data contamination between splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import random
import hashlib
from collections import defaultdict

def normalize_text(text):
    """Normalize text for deduplication by removing extra whitespace and converting to lowercase."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip().lower()

def create_text_hash(text):
    """Create a hash of the normalized text for efficient deduplication."""
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def deduplicate_data(data):
    """Remove duplicate texts from data, keeping the first occurrence."""
    print(f"Deduplicating data... Starting with {len(data)} examples")
    
    seen_hashes = set()
    deduplicated_data = []
    
    for text, label in data:
        text_hash = create_text_hash(text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            deduplicated_data.append((text, label))
    
    removed_count = len(data) - len(deduplicated_data)
    print(f"Removed {removed_count} duplicate texts. {len(deduplicated_data)} unique texts remain.")
    
    return deduplicated_data

def load_shortjokes_data(filepath):
    """Load shortjokes CSV and return as list of (text, label) tuples."""
    print(f"Loading shortjokes data from {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                # Extract joke text and assign label 1 (humorous)
                joke_text = row[1].strip()
                if joke_text:  # Only add non-empty jokes
                    data.append((joke_text, 1))
    
    print(f"Loaded {len(data)} jokes from shortjokes dataset")
    return data

def load_puns_data(filepath):
    """Load puns CSV and return as list of (text, label) tuples."""
    print(f"Loading puns data from {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                try:
                    label = int(row[0].strip())
                    text = row[1].strip()
                    
                    if text:  # Only add non-empty texts
                        # Convert -1 labels to 0 for binary classification
                        binary_label = 1 if label == 1 else 0
                        data.append((text, binary_label))
                except ValueError:
                    continue  # Skip rows with invalid labels
    
    print(f"Loaded {len(data)} examples from puns dataset")
    return data

def load_kaggle_data(filepath):
    """Load Kaggle dataset CSV and return only non-humorous entries as list of (text, label) tuples."""
    print(f"Loading Kaggle data from {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 2:
                try:
                    text = row[0].strip()
                    humor_value = row[1].strip().lower()
                    
                    # Only take non-humorous entries (False)
                    if text and humor_value == 'false':
                        data.append((text, 0))  # Label as 0 (not funny)
                except (ValueError, IndexError):
                    continue  # Skip rows with invalid data
    
    print(f"Loaded {len(data)} non-humorous examples from Kaggle dataset")
    return data

def split_data_by_source(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets using the specified ratios.
    No balancing - just proportional splits from the source.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Shuffle data for randomness
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def combine_and_deduplicate_splits(all_data_sources):
    """
    Combine all data sources, deduplicate globally, and create contamination-free splits.
    
    Args:
        all_data_sources: List of data lists, each containing (text, label) tuples
    
    Returns:
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    """
    print("Combining all data sources...")
    
    # Combine all data from all sources
    all_data = []
    for data_source in all_data_sources:
        all_data.extend(data_source)
    
    print(f"Total combined data: {len(all_data)} examples")
    
    # Global deduplication
    deduplicated_data = deduplicate_data(all_data)
    
    # Shuffle for random splitting
    random.shuffle(deduplicated_data)
    
    # Split into train/val/test with 0.7/0.15/0.15 ratios
    total_size = len(deduplicated_data)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)
    
    train_data = deduplicated_data[:train_size]
    val_data = deduplicated_data[train_size:train_size + val_size]
    test_data = deduplicated_data[train_size + val_size:]
    
    print(f"Split sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Balance the test set between positive and negative samples
    test_positive = [item for item in test_data if item[1] == 1]
    test_negative = [item for item in test_data if item[1] == 0]
    
    print(f"Before balancing test set: {len(test_positive)} positive, {len(test_negative)} negative")
    
    # Use the smaller class size to balance test set
    min_test_size = min(len(test_positive), len(test_negative))
    
    # Randomly sample to balance test set
    random.shuffle(test_positive)
    random.shuffle(test_negative)
    
    balanced_test_positive = test_positive[:min_test_size]
    balanced_test_negative = test_negative[:min_test_size]
    
    # Move excess test samples back to training set to avoid waste
    excess_positive = test_positive[min_test_size:]
    excess_negative = test_negative[min_test_size:]
    train_data.extend(excess_positive)
    train_data.extend(excess_negative)
    
    # Create final balanced test set
    balanced_test = balanced_test_positive + balanced_test_negative
    random.shuffle(balanced_test)
    
    # Shuffle training data after adding excess test samples
    random.shuffle(train_data)
    
    print(f"After balancing test set: {len(balanced_test_positive)} positive, {len(balanced_test_negative)} negative")
    print(f"Total balanced test set size: {len(balanced_test)}")
    print(f"Final training set size: {len(train_data)} (includes excess test samples)")
    
    # Verify no contamination between splits
    verify_no_contamination(train_data, val_data, balanced_test)
    
    # Separate texts and labels
    train_texts = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]
    
    val_texts = [item[0] for item in val_data]
    val_labels = [item[1] for item in val_data]
    
    test_texts = [item[0] for item in balanced_test]
    test_labels = [item[1] for item in balanced_test]
    
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def verify_no_contamination(train_data, val_data, test_data):
    """Verify that there is no overlap between train, validation, and test sets."""
    print("Verifying no contamination between splits...")
    
    train_hashes = set(create_text_hash(text) for text, _ in train_data)
    val_hashes = set(create_text_hash(text) for text, _ in val_data)
    test_hashes = set(create_text_hash(text) for text, _ in test_data)
    
    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)
    
    contamination_found = len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    
    if contamination_found:
        print(f"❌ CONTAMINATION DETECTED!")
        print(f"   Train-Val overlap: {len(train_val_overlap)}")
        print(f"   Train-Test overlap: {len(train_test_overlap)}")
        print(f"   Val-Test overlap: {len(val_test_overlap)}")
        raise ValueError("Data contamination detected! This should not happen with the new splitting method.")
    else:
        print("✅ No contamination detected between splits!")
    
    return True

def save_split_to_csv(texts, labels, filepath):
    """Save a data split to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])  # Header
        
        for text, label in zip(texts, labels):
            writer.writerow([text, label])
    
    print(f"Saved {len(texts)} examples to {filepath}")

def print_split_statistics(train_data, val_data, test_data):
    """Print statistics about the data splits."""
    def count_labels(labels):
        pos_count = sum(1 for label in labels if label == 1)
        neg_count = sum(1 for label in labels if label == 0)
        return pos_count, neg_count
    
    train_pos, train_neg = count_labels(train_data[1])
    val_pos, val_neg = count_labels(val_data[1])
    test_pos, test_neg = count_labels(test_data[1])
    
    print("\nData Split Statistics:")
    print("=" * 50)
    print(f"Train: {len(train_data[0])} examples ({train_pos} positive, {train_neg} negative)")
    print(f"Val:   {len(val_data[0])} examples ({val_pos} positive, {val_neg} negative)")
    print(f"Test:  {len(test_data[0])} examples ({test_pos} positive, {test_neg} negative)")
    
    total_examples = len(train_data[0]) + len(val_data[0]) + len(test_data[0])
    total_pos = train_pos + val_pos + test_pos
    total_neg = train_neg + val_neg + test_neg
    
    print(f"\nTotal: {total_examples} examples ({total_pos} positive, {total_neg} negative)")
    print(f"Split ratios: Train={len(train_data[0])/total_examples:.3f}, "
          f"Val={len(val_data[0])/total_examples:.3f}, "
          f"Test={len(test_data[0])/total_examples:.3f}")

def main():
    """Main preprocessing function."""
    print("Starting raw data preprocessing...")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define paths relative to repository root
    repo_root = Path(__file__).resolve().parents[2]
    raw_data_dir = repo_root / "data" / "raw"
    processed_data_dir = repo_root / "data" / "processed"
    
    shortjokes_path = raw_data_dir / "shortjokes.csv"
    puns_path = raw_data_dir / "puns_pos_neg_data.csv"
    kaggle_path = raw_data_dir / "kaggle_dataset.csv"
    
    # Load data from all three sources
    print("Loading data from all sources...")
    shortjokes_data = load_shortjokes_data(shortjokes_path)
    puns_data = load_puns_data(puns_path)
    kaggle_data = load_kaggle_data(kaggle_path)
    
    # Deduplicate each source individually first
    print(f"\nDeduplicating individual data sources...")
    shortjokes_data = deduplicate_data(shortjokes_data)
    puns_data = deduplicate_data(puns_data)
    kaggle_data = deduplicate_data(kaggle_data)
    
    # Combine all sources and create contamination-free splits
    print(f"\nCombining all sources and creating contamination-free splits...")
    train_data, val_data, test_data = combine_and_deduplicate_splits(
        [shortjokes_data, puns_data, kaggle_data]
    )
    
    # Print statistics
    print_split_statistics(train_data, val_data, test_data)
    
    # Save splits to CSV files
    print(f"\nSaving processed data to {processed_data_dir}...")
    
    save_split_to_csv(train_data[0], train_data[1], processed_data_dir / "humor_train.csv")
    save_split_to_csv(val_data[0], val_data[1], processed_data_dir / "humor_val.csv")
    save_split_to_csv(test_data[0], test_data[1], processed_data_dir / "humor_test.csv")
    
    print("\nPreprocessing completed successfully!")
    print(f"Processed files saved in: {processed_data_dir}")

if __name__ == "__main__":
    main()
