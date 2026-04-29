import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import time
import os
import cv2

# --- CONFIGURATION ---
BASE_DIR = r"C:\procoder\Neuro Symbolic Ai\openlab"
DATASET_NAME = "ALL_IDB"  # Change to "C-NMC" to evaluate that one
MODEL_PATH = f"{DATASET_NAME}_best_model.pth"
METADATA_PATH = os.path.join(BASE_DIR, "data_processed", DATASET_NAME, f"{DATASET_NAME}_metadata.csv")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_CLASSES = 4 if DATASET_NAME == "ALL_IDB" else 2

from symbolic_logic import MedicalLogicBridge
from main import extract_visual_features

# --- LOGIC BRIDGE SETUP ---
# Initialize the real RAG system
print("🧠 Initializing Logic Bridge...")
bridge = MedicalLogicBridge("medical_rules.txt")

# --- DATASET CLASS ---
class EvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.classes = sorted(df['label'].unique().tolist())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = self.class_to_idx[row['label']]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

# --- EVALUATION ENGINE ---
def evaluate():
    print(f"🚀 STARTING EVALUATION FOR: {DATASET_NAME}")
    print(f"   Model: {MODEL_PATH}")
    
    # 1. Load Data (Use Fold 0 as Test Set)
    df = pd.read_csv(METADATA_PATH)
    test_df = df[df['fold'] == 0] # Using Fold 0 for validation metrics
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = EvalDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = test_dataset.classes
    print(f"   Classes: {class_names}")

    # 2. Load Model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Run Inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_time = 0
    logic_flags = 0
    
    print("\n   🧠 Running Inference & Logic Check...")
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader):
            images = images.to(DEVICE)
            
            # Start Timer (Efficiency Metric)
            start = time.time()
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            # End Timer
            end = time.time()
            total_time += (end - start)
            
            # --- REAL NEURO-SYMBOLIC CHECK ---
            # 1. Convert tensor back to image path to get features
            # (In a live system, we'd use the tensor directly if we had a tensor-based feature extractor,
            # but here we use OpenCV on the file path as per main.py design)
            
            for i in range(len(preds)):
                pred_idx = preds[i].item()
                pred_label = class_names[pred_idx]
                img_path = paths[i]
                
                # A. Extract Visual Features (The "Symbolic" View)
                visual_features = extract_visual_features(img_path)
                
                # B. Verify with Logic Bridge
                is_valid, explanation = bridge.verify_prediction(pred_label, visual_features)
                
                if not is_valid:
                    logic_flags += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- 4. CALCULATE METRICS ---
    
    # A. Speed
    avg_latency = (total_time / len(test_dataset)) * 1000 # ms
    print(f"\n⚡ EFFICIENCY REPORT:")
    print(f"   Avg Inference Time: {avg_latency:.2f} ms/image")
    print(f"   Throughput: {1000/avg_latency:.1f} images/sec")
    
    # B. Classification Report
    print("\n📊 CLASSIFICATION METRICS:")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    
    # C. Logic Guardrail Stats
    print(f"\n🛡️ NEURO-SYMBOLIC SAFETY:")
    print(f"   Total Samples: {len(test_dataset)}")
    print(f"   Logic Interventions: {logic_flags}")
    print(f"   Rejection Rate: {(logic_flags/len(test_dataset))*100:.2f}% (Real Logic Verification)")

    # D. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {DATASET_NAME}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('Figure_6_Confusion_Matrix.png')
    print("   ✅ Saved: Figure_6_Confusion_Matrix.png")
    
    # E. ROC-AUC Plot (Multi-class)
    y_test_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
    y_score = np.array(all_probs)
    
    plt.figure(figsize=(10, 8))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {DATASET_NAME}')
    plt.legend(loc="lower right")
    plt.savefig('Figure_7_ROC_Curve.png')
    print("   ✅ Saved: Figure_7_ROC_Curve.png")

if __name__ == "__main__":
    evaluate()