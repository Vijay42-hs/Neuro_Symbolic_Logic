import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time

# --- CONFIGURATION (Exact settings from Jammal et al. 2025) ---
CONFIG = {
    'img_size': (128, 128),
    'batch_size': 32,          # 
    'epochs': 10,              # 
    'lr': 0.01,                # 
    'momentum': 0.9,           # 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,          # Set to 0 for Windows compatibility
    'fold_to_train': 0         # We start with Fold 0 for validation
}

# --- CUSTOM DATASET CLASS ---
class LeukemiaDataset(Dataset):
    def __init__(self, metadata_csv, root_dir, transform=None):
        self.df = pd.read_csv(metadata_csv)
        self.root_dir = root_dir
        self.transform = transform
        
        # Auto-detect classes from the dataframe
        self.classes = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label_str = row['label']
        label = self.class_to_idx[label_str]
        
        # Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Error loading: {img_path}")
            # Return a blank image to prevent crash (rare fallback)
            image = Image.new('RGB', CONFIG['img_size'])

        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- TRAINING ENGINE ---
def train_model(dataset_name, metadata_path):
    print(f"\n🚀 Training EfficientNet-B0 on {dataset_name} (Fold {CONFIG['fold_to_train']})")
    
    # 1. Prepare Data
    df = pd.read_csv(metadata_path)
    
    # Split by Fold (Paper uses 5-Fold Cross Validation) [cite: 2803]
    train_df = df[df['fold'] != CONFIG['fold_to_train']]
    val_df = df[df['fold'] == CONFIG['fold_to_train']]
    
    # Save temp CSVs for the Dataset class to read
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)
    
    # Transforms (Standardization only for training loop inputs)
    # Note: Augmentation was already done physically in preprocessing!
    data_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = LeukemiaDataset('temp_train.csv', '', transform=data_transform)
    val_dataset = LeukemiaDataset('temp_val.csv', '', transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    num_classes = len(train_dataset.classes)
    print(f"   Classes: {train_dataset.classes}")
    
    # 2. Build Model (EfficientNet-B0) [cite: 2688]
    model = models.efficientnet_b0(weights='DEFAULT')
    
    # Replace the 'classifier' head for our number of classes
    # EfficientNet-B0 classifier layer is usually (1280 -> 1000). We change to (1280 -> num_classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(CONFIG['device'])
    
    # 3. Setup Loss & Optimizer (SGD) [cite: 2787, 2910]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG['lr'], momentum=CONFIG['momentum'])
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        print("-" * 10)
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"   Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(CONFIG['device'])
                labels = labels.to(CONFIG['device'])
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"   Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # --- SAVE BEST MODEL ---
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f"{dataset_name}_best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"   🔥 New Best Model Saved! ({best_acc:.4f})")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")
    
    # Cleanup temp files
    if os.path.exists('temp_train.csv'): os.remove('temp_train.csv')
    if os.path.exists('temp_val.csv'): os.remove('temp_val.csv')

# --- EXECUTION ---
if __name__ == "__main__":
    BASE_DIR = r"C:\procoder\Neuro Symbolic Ai\openlab"
    
    # Train on ALL Dataset
    all_meta = os.path.join(BASE_DIR, "data_processed", "ALL_IDB", "ALL_IDB_metadata.csv")
    if os.path.exists(all_meta):
        train_model("ALL_IDB", all_meta)
    
    # Train on C-NMC Dataset
    cnmc_meta = os.path.join(BASE_DIR, "data_processed", "C-NMC", "C-NMC_metadata.csv")
    if os.path.exists(cnmc_meta):
        train_model("C-NMC", cnmc_meta)