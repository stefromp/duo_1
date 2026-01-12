#!/usr/bin/env python3
"""
Fix train/val data split to ensure both come from the same distribution.
This prevents misleading validation metrics.
"""

import random
import os

print("=" * 60)
print("FIXING TRAIN/VAL DATA SPLIT")
print("=" * 60)

# Set paths
train_file = 'processed_data/train_subset_clean.txt'
val_file = 'processed_data/val_subset_clean.txt'

# Check files exist
if not os.path.exists(train_file):
    raise FileNotFoundError(f"Training file not found: {train_file}")
if not os.path.exists(val_file):
    raise FileNotFoundError(f"Validation file not found: {val_file}")

# Read all data
print("\n📖 Reading files...")
with open(train_file, 'r', encoding='utf-8') as f:
    train_lines = f.readlines()
print(f"   Train: {len(train_lines):,} lines")

with open(val_file, 'r', encoding='utf-8') as f:
    val_lines = f.readlines()
print(f"   Val:   {len(val_lines):,} lines")

original_total = len(train_lines) + len(val_lines)
print(f"   Total: {original_total:,} lines")

# Combine and shuffle
print("\n🔀 Combining and shuffling all data...")
all_lines = train_lines + val_lines
random.seed(42)  # Fixed seed for reproducibility
random.shuffle(all_lines)
print(f"   ✓ Shuffled {len(all_lines):,} lines")

# Split 90/10 (maintaining approximately the same ratio as before)
split_ratio = 0.9
split_idx = int(len(all_lines) * split_ratio)
new_train = all_lines[:split_idx]
new_val = all_lines[split_idx:]

print(f"\n✂️  Splitting data ({split_ratio*100:.0f}/{(1-split_ratio)*100:.0f})...")
print(f"   New Train: {len(new_train):,} lines ({len(new_train)/len(all_lines)*100:.1f}%)")
print(f"   New Val:   {len(new_val):,} lines ({len(new_val)/len(all_lines)*100:.1f}%)")

# Verify no data loss
assert len(new_train) + len(new_val) == original_total, "Data loss detected!"

# Create backup of original files
import shutil
backup_dir = 'processed_data/backup_original'
os.makedirs(backup_dir, exist_ok=True)

print(f"\n💾 Creating backups in {backup_dir}/...")
shutil.copy2(train_file, os.path.join(backup_dir, 'train_subset_clean.txt'))
shutil.copy2(val_file, os.path.join(backup_dir, 'val_subset_clean.txt'))
print("   ✓ Backups created")

# Save new splits
print("\n💾 Writing new splits...")
with open(train_file, 'w', encoding='utf-8') as f:
    f.writelines(new_train)
print(f"   ✓ Wrote {len(new_train):,} lines to {train_file}")

with open(val_file, 'w', encoding='utf-8') as f:
    f.writelines(new_val)
print(f"   ✓ Wrote {len(new_val):,} lines to {val_file}")

# Show sample from each split
print("\n📄 Sample from NEW TRAIN split (first 200 chars):")
print("   " + new_train[0][:200].replace('\n', ' ') + "...")

print("\n📄 Sample from NEW VAL split (first 200 chars):")
print("   " + new_val[0][:200].replace('\n', ' ') + "...")

print("\n" + "=" * 60)
print("✅ DATA SPLIT FIXED SUCCESSFULLY!")
print("=" * 60)
print("\n📝 Next steps:")
print("   1. Review the changes")
print("   2. Commit and push to GitHub:")
print("      git add processed_data/")
print("      git commit -m 'Fix train/val split - ensure same domain'")
print("      git push")
print("\n" + "=" * 60)
