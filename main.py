import torch
import cv2
import numpy as np
import os
import glob
from torchvision import models, transforms
from PIL import Image
from symbolic_logic import MedicalLogicBridge

# --- CONFIGURATION ---
MODEL_PATH = "ALL_IDB_best_model.pth"  # Your trained model
IMAGE_DIR = r"C:\procoder\Neuro Symbolic Ai\openlab\data_processed\ALL_IDB\Early" # Folder to test images from
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. FEATURE EXTRACTOR (The "Neuro" Vision Part) ---
def extract_visual_features(image_path):
    """
    Uses OpenCV to measure physical properties of the cell.
    This replaces the 'black box' with measurable features.
    Updated: Now includes Texture (Chromatin) and Nucleoli detection.
    """
    # Load image in Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Thresholding to find the dark nucleus
    # Cells are usually darker than background in blood smears
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Find Contours (Shapes) - Use RETR_CCOMP to find holes (nucleoli) inside
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    features = []
    
    if contours:
        # Get largest contour (assuming it's the nucleus)
        # We need the index calculate hierarchy
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(areas)
        nucleus = contours[max_idx]
        area = areas[max_idx]
        perimeter = cv2.arcLength(nucleus, True)
        
        # --- SHAPE FEATURES ---
        # Calculate 'Circularity' (1.0 = perfect circle, < 0.6 = irregular)
        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
        else:
            circularity = 0
            
        # Logic Mapping (Mapping pixels to words)
        # Note: These thresholds are calibrated for 128x128 images
        if area > 2000:
            features.append("large size")
        elif area < 1000:
            features.append("small size")
        else:
            features.append("medium size")
            
        if circularity < 0.7:
            features.append("irregular nuclear contours")
        else:
            features.append("regular nuclear contours")

        # --- TEXTURE & NUCLEOLI FEATURES ---
        
        # 1. Nucleoli Detection (Holes inside the nucleus)
        # hierarchy[0][i] = [Next, Previous, First_Child, Parent]
        # If the nucleus contour has a child, it likely has a nucleolus (hole in binary mask)
        # We check if hierarchy[0][max_idx][2] != -1 (it has a child)
        if hierarchy is not None and hierarchy[0][max_idx][2] != -1:
            features.append("prominent nucleoli")
        
        # 2. Chromatin Texture (Pixel Intensity Variance)
        # We mask the original image with the nucleus to only analyze nuclear pixels
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [nucleus], -1, 255, -1)
        mean, std_dev = cv2.meanStdDev(img, mask=mask)
        
        # Heuristic: High variance = Heterogeneous/Condensed Chromatin
        # Low variance = Smooth/Homogeneous
        # Threshold (20) is empirical for 8-bit images
        if std_dev[0][0] > 20: 
            features.append("condensed chromatin")
        else:
            features.append("fine chromatin")
            
    return features

# --- 2. LOAD NEURAL MODEL ---
def load_model(path, num_classes=4):
    print(f"   🤖 Loading Neural Brain from {path}...")
    model = models.efficientnet_b0(weights=None) # No internet needed if weights loaded
    
    # Rebuild the classifier head exactly as in training
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("   ✅ Model Loaded Successfully.")
        return model
    except FileNotFoundError:
        print("   ❌ Model file not found! Did you run train_baseline.py?")
        return None

# --- 3. MAIN PIPELINE ---
def run_diagnosis():
    print("\n🚀 STARTING NEURO-SYMBOLIC DIAGNOSIS SYSTEM")
    print("=============================================")
    
    # A. Initialize Logic Bridge
    bridge = MedicalLogicBridge("medical_rules.txt")
    
    # B. Load Model
    model = load_model(MODEL_PATH)
    if not model: return

    # C. Get a Random Test Image
    # C. Get a Random Test Image (Improved to find .jpg, .png, .bmp)
    test_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        test_images.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
    if not test_images:
        print(f"   ❌ No images found in: {IMAGE_DIR}")
        return
    if not test_images:
        print("   ❌ No images found. Check IMAGE_DIR path.")
        return
        
    # Pick the first one for demo
    target_image_path = test_images[0]
    print(f"\n📸 Analyzing Patient Sample: {os.path.basename(target_image_path)}")
    
    # --- STEP 1: NEURAL PREDICTION ---
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.open(target_image_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    # Mapping indices to class names (Must match training order)
    # Usually: ['Benign', 'Early', 'Pre', 'Pro']
    classes = ['Benign', 'Early', 'Pre', 'Pro'] 
    predicted_label = classes[predicted_idx.item()]
    
    print(f"   🧠 Neural Network Diagnosis: {predicted_label} ({confidence.item()*100:.1f}%)")

    # --- STEP 2: VISUAL FEATURE EXTRACTION (OpenCV) ---
    extracted_features = extract_visual_features(target_image_path)
    print(f"   👁️  Computer Vision Detected: {extracted_features}")

    # --- STEP 3: SYMBOLIC VERIFICATION (The "Paper" Contribution) ---
    # We ask the bridge: "Does a 'Pro-B' diagnosis make sense given these features?"
    
    is_valid, explanation = bridge.verify_prediction(predicted_label, extracted_features)
    
    print("\n📝 FINAL REPORT:")
    print("-" * 20)
    print(f"Diagnosis: {predicted_label}")
    print(f"Validation: {explanation}")
    
    if is_valid:
        print("✅ SYSTEM DECISION: APPROVED (High Confidence)")
    else:
        print("🛑 SYSTEM DECISION: FLAGGED FOR REVIEW (Logic Mismatch)")

if __name__ == "__main__":
    run_diagnosis()