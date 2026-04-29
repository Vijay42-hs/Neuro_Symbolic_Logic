import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as T

# --- SETTINGS (Matched to Jammal et al., 2025) ---
CONFIG = {
    'img_size': (128, 128),
    'target_counts': {
        'C-NMC': 7272,  # Majority class count in C-NMC
        'ALL_IDB': 985  # Majority class count in ALL
    },
    'aug_params': {
        'rotation': 30,
        'zoom': (0.8, 1.2),
        'brightness': (0.8, 1.2),
        'h_flip': 0.5
    }
}

# --- TRANSFORMS ---
def get_augmentor():
    return T.Compose([
        T.RandomRotation(degrees=CONFIG['aug_params']['rotation']),
        T.RandomAffine(degrees=0, scale=CONFIG['aug_params']['zoom']),
        T.ColorJitter(brightness=CONFIG['aug_params']['brightness']),
        T.RandomHorizontalFlip(p=CONFIG['aug_params']['h_flip']),
        T.Resize(CONFIG['img_size'])
    ])

def get_standardizer():
    return T.Compose([
        T.Resize(CONFIG['img_size'])
    ])

# --- PROCESSING ENGINE ---
def process_dataset(dataset_name, input_root, output_root, search_pattern):
    """
    dataset_name: Name for the metadata file
    input_root: The folder where we start looking for images
    output_root: Where to save processed images
    search_pattern: How to find the class folders (e.g., recursive or direct)
    """
    print(f"\n🚀 Starting Preprocessing for: {dataset_name}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    target_n = CONFIG['target_counts'][dataset_name]
    augmentor = get_augmentor()
    standardizer = get_standardizer()
    
    all_records = [] # For the CSV metadata

    # 1. Define classes based on dataset type
    if dataset_name == 'C-NMC':
        classes = ['all', 'hem'] # Specific to C-NMC
    else:
        # For ALL dataset (Original folder), read subfolders (Benign, Early, etc.)
        classes = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

    print(f"   Classes found: {classes}")

    for cls in classes:
        # Create class folder in output
        cls_output_dir = os.path.join(output_root, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        # 2. Find Images (Handles the complex folder structure of C-NMC automatically)
        # We use recursive search to find images even inside fold_0, fold_1, etc.
        if dataset_name == 'C-NMC':
            # Search inside C-NMC 2019 (PKG)/C-NMC_training_data/**/class/*.bmp
            search_path = os.path.join(input_root, '**', cls, '*.bmp')
        else:
            # Search inside Original/class/*.jpg (or .tif/.bmp)
            search_path = os.path.join(input_root, cls, '*')
        
        # Gather all files
        images = glob.glob(search_path, recursive=True)
        # Filter for valid image extensions just in case
        images = [f for f in images if f.lower().endswith(('.bmp', '.jpg', '.png', '.tif', '.tiff'))]
        
        current_count = len(images)
        print(f"   ➜ Class '{cls}': Found {current_count} images.")

        if current_count == 0:
            print(f"      ⚠️ WARNING: No images found for {cls}. Check folder paths!")
            continue

        # 3. Process Original Images
        print(f"      Standardizing originals...")
        for img_path in tqdm(images, leave=False):
            try:
                img = Image.open(img_path).convert('RGB')
                img_std = standardizer(img)
                
                # Save
                fname = f"orig_{os.path.basename(img_path)}"
                if not fname.lower().endswith(".jpg"): fname = os.path.splitext(fname)[0] + ".jpg"
                
                save_path = os.path.join(cls_output_dir, fname)
                img_std.save(save_path, quality=95)
                
                all_records.append({'path': save_path, 'label': cls, 'is_augmented': False})
            except Exception as e:
                print(f"      [Error] Could not process {img_path}: {e}")

        # 4. Generate Synthetic Images (Balancing)
        needed = target_n - current_count
        if needed > 0:
            print(f"      ⚠️ Imbalance detected! Generating {needed} synthetic images...")
            
            # Randomly sample originals to augment
            source_imgs = np.random.choice(images, needed, replace=True)
            
            for i, src_path in enumerate(tqdm(source_imgs, leave=False)):
                try:
                    img = Image.open(src_path).convert('RGB')
                    img_aug = augmentor(img)
                    
                    fname = f"aug_{i}_{os.path.basename(src_path)}"
                    if not fname.lower().endswith(".jpg"): fname = os.path.splitext(fname)[0] + ".jpg"
                    
                    save_path = os.path.join(cls_output_dir, fname)
                    img_aug.save(save_path, quality=95)
                    
                    all_records.append({'path': save_path, 'label': cls, 'is_augmented': True})
                except Exception as e:
                    pass
        else:
            print(f"      ✅ Class is balanced (or already exceeds target).")

    # 5. Create 5-Fold Splits
    if len(all_records) > 0:
        print("   Generating 5-Fold Metadata...")
        df = pd.DataFrame(all_records)
        
        # We use Stratified K-Fold to ensure every fold has the same % of cancer/normal
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        df['fold'] = -1
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            df.loc[val_idx, 'fold'] = fold
            
        csv_path = os.path.join(output_root, f"{dataset_name}_metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"   ✅ Done! Metadata saved to {csv_path}")
    else:
        print("   ❌ No records to save. Check your input paths.")

# --- EXECUTION ---
if __name__ == "__main__":
    BASE_DIR = os.getcwd()
    
    # 1. Process ALL Dataset (From 'Original' folder)
    # Based on your screenshot, 'Original' is in the root
    process_dataset(
        dataset_name="ALL_IDB",
        input_root=os.path.join(BASE_DIR, "Original"), 
        output_root=os.path.join(BASE_DIR, "data_processed", "ALL_IDB"),
        search_pattern="direct"
    )

    # 2. Process C-NMC (From 'C-NMC 2019 (PKG)/C-NMC_training_data')
    # Based on your screenshot
    process_dataset(
        dataset_name="C-NMC",
        input_root=os.path.join(BASE_DIR, "C-NMC 2019 (PKG)", "C-NMC_training_data"),
        output_root=os.path.join(BASE_DIR, "data_processed", "C-NMC"),
        search_pattern="recursive" 
    )