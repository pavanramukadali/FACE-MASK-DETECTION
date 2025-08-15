"""
split_dataset.py
Usage: python split_dataset.py
Expects raw/with_mask and raw/without_mask. Produces dataset/{train,val,test}/{class}/
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_ROOT = Path('data')
OUT_ROOT = Path('dataset')
CLASSES = ['with_mask', 'without_mask']
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
VALID_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def gather_files(folder):
    return [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS and p.is_file()]

def ensure_dirs():
    for split in ['train','val','test']:
        for cls in CLASSES:
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

def copy_list(paths, dest):
    for p in paths:
        shutil.copy2(p, dest / p.name)

def main():
    ensure_dirs()
    for cls in CLASSES:
        src = RAW_ROOT / cls
        files = gather_files(src)
        if not files:
            print(f"Warning: no files found in {src}")
            continue

        train_files, temp_files = train_test_split(files, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
        # split temp into val and test equally
        val_size_rel = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
        val_files, test_files = train_test_split(temp_files, train_size=val_size_rel, random_state=RANDOM_STATE)

        copy_list(train_files, OUT_ROOT / 'train' / cls)
        copy_list(val_files, OUT_ROOT / 'val' / cls)
        copy_list(test_files, OUT_ROOT / 'test' / cls)

        print(f"{cls}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("Done. Check the 'dataset' folder.")

if __name__ == '__main__':
    main()
