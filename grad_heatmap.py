import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = "ALL_IDB_best_model.pth"
IMAGE_PATH = r"C:\procoder\Neuro Symbolic Ai\openlab\data_processed\ALL_IDB\Pro\aug_0_WBC-Malignant-Pro-268.jpg" # Use the exact image you just tested
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_visual_explanation():
    print(f"📸 Generating XAI Heatmap for: {IMAGE_PATH}")
    
    # 1. Load Model
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 4) # 4 classes for ALL
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    # 2. Target Layer for Heatmap
    # For EfficientNet, the last convolutional layer is usually in 'features[-1]'
    target_layers = [model.features[-1]]

    # 3. Prepare Image
    rgb_img = cv2.imread(IMAGE_PATH, 1)[:, :, ::-1] # BGR to RGB
    rgb_img = cv2.resize(rgb_img, (128, 128))
    rgb_img = np.float32(rgb_img) / 255
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(Image.open(IMAGE_PATH)).unsqueeze(0).to(DEVICE)

    # 4. Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # We want to explain the "Pro" class (Index 3, assuming sorted: Benign, Early, Pre, Pro)
    # If you aren't sure of the index, the code below grabs the highest predicted one
    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # 5. Overlay Heatmap
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # 6. Save/Show
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(rgb_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"AI Attention (Grad-CAM)")
    plt.imshow(visualization)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("explanation_result.png")
    print("✅ Heatmap saved as 'explanation_result.png'. Open it to see where the AI looked!")
    plt.show()

if __name__ == "__main__":
    generate_visual_explanation()