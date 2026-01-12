#!/usr/bin/env python3
"""
Split dataset into train and validation sets.
Usage: python split_dataset.py
"""

import random

# Configuration
INPUT_FILE = "processed_data/train_subset_clean.txt"
TRAIN_OUTPUT = "processed_data/train.txt"
VALID_OUTPUT = "processed_data/validation.txt"
VALIDATION_SPLIT = 0.05  # 5% for validation, 95% for training
SEED = 42

def split_dataset():
    print(f"Reading dataset from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total examples: {total_lines:,}")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(SEED)
    random.shuffle(lines)
    
    # Calculate split point
    valid_size = int(total_lines * VALIDATION_SPLIT)
    train_size = total_lines - valid_size
    
    print(f"\nSplit:")
    print(f"  Training:   {train_size:,} examples ({(1-VALIDATION_SPLIT)*100:.1f}%)")
    print(f"  Validation: {valid_size:,} examples ({VALIDATION_SPLIT*100:.1f}%)")
    
    # Split data
    train_lines = lines[:train_size]
    valid_lines = lines[train_size:]
    
    # Write training set
    print(f"\nWriting training set to: {TRAIN_OUTPUT}")
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation set
    print(f"Writing validation set to: {VALID_OUTPUT}")
    with open(VALID_OUTPUT, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    print("\n✅ Dataset split complete!")
    print(f"   Train file size: {len(''.join(train_lines)) / (1024**2):.1f} MB")
    print(f"   Valid file size: {len(''.join(valid_lines)) / (1024**2):.1f} MB")

if __name__ == "__main__":
    split_dataset()
