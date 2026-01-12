#!/usr/bin/env python3
"""
Prepare data files for Kaggle Dataset upload.
Creates a clean directory with just the essential files.
"""

import os
import shutil
from pathlib import Path

print("=" * 60)
print("PREPARING DATA FILES FOR KAGGLE UPLOAD")
print("=" * 60)

# Create export directory
export_dir = Path("kaggle_dataset_export")
export_dir.mkdir(exist_ok=True)

# Files to copy
files_to_copy = [
    "processed_data/train_subset_clean.txt",
    "processed_data/val_subset_clean.txt"
]

print(f"\n📁 Creating export directory: {export_dir}/")

# Copy files
for file_path in files_to_copy:
    src = Path(file_path)
    if src.exists():
        dst = export_dir / src.name
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"   ✓ Copied: {src.name} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ Missing: {file_path}")

# Create dataset metadata for Kaggle
metadata = {
    "title": "DUO Custom Training Data",
    "id": "stefromp/duo-custom-training-data",
    "licenses": [{"name": "CC0-1.0"}]
}

import json
with open(export_dir / "dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Created: dataset-metadata.json")

# Create README
readme_content = """# DUO Custom Training Data

This dataset contains training and validation data for the DUO (Diffusion for Unsupervised Ordering) language model.

## Files

- `train_subset_clean.txt`: Training data (204,166 lines, ~48MB)
- `val_subset_clean.txt`: Validation data (22,686 lines, ~4.8MB)

## Data Split

- Split ratio: 90% train / 10% validation
- Random shuffle with seed=42
- Same domain distribution across both splits

## Usage

In your Kaggle notebook:

```python
# Add this dataset to your notebook using "Add Data" button
# Then copy files to your working directory:

!mkdir -p processed_data
!cp /kaggle/input/duo-custom-training-data/train_subset_clean.txt processed_data/
!cp /kaggle/input/duo-custom-training-data/val_subset_clean.txt processed_data/
```

## Source

Data prepared from general web text corpus, properly split to avoid domain mismatch.
"""

with open(export_dir / "README.md", "w") as f:
    f.write(readme_content)
print(f"   ✓ Created: README.md")

print("\n" + "=" * 60)
print("✅ EXPORT COMPLETE!")
print("=" * 60)
print(f"\n📂 Files ready in: {export_dir.absolute()}/")
print("\n📤 NEXT STEPS:")
print("   1. Go to: https://www.kaggle.com/datasets")
print("   2. Click 'New Dataset'")
print("   3. Upload ALL files from kaggle_dataset_export/")
print("   4. Set title: 'DUO Custom Training Data'")
print("   5. Click 'Create'")
print("   6. Add dataset to your notebook using 'Add Data' button")
print("\n" + "=" * 60)
