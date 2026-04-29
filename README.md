# Neuro-Symbolic Logic for Leukemia Diagnosis

A research project focused on building a hybrid **Neuro-Symbolic Artificial Intelligence** system that diagnoses Leukemia from blood smear images by combining the pattern recognition of deep learning with the interpretability of symbolic medical logic.

---

## 📌 Project Overview

Traditional neural networks act as "black boxes." This project bridges the gap between raw prediction and medical reasoning by using a hybrid approach:

1. **Neural Component:** An `EfficientNet-B0` model classifies the cell image.
2. **Visual Feature Extractor:** OpenCV extracts explicit physical properties (size, circularity, nucleoli, chromatin texture).
3. **Symbolic Verification:** A Logic Bridge (powered by ChromaDB and SentenceTransformers) retrieves established medical rules to verify if the extracted visual features match the neural diagnosis.

If the visual evidence contradicts the established medical rule for the predicted class, the system flags the decision for human review.

---

## 🧠 Models & Logic System

| Component | Technology | Role |
|---|---|---|
| Neural Vision | `EfficientNet-B0` (PyTorch) | Image classification (Benign, Early, Pre, Pro) |
| Feature Extraction | OpenCV | Quantifying nucleus size, shape, and texture |
| Vector Database | `ChromaDB` | Storing medical rules as vector embeddings |
| Embedder | `all-MiniLM-L6-v2` | Converting text rules into semantic vectors |
| Explainability | `Grad-CAM` | Visualizing model attention/heatmaps |

---

## 📊 Datasets

The system is configured to train and evaluate on two standard benchmark datasets for Leukemia detection:

| Dataset | Classes | Description |
|---|---|---|
| ALL_IDB | Benign, Early, Pre, Pro | Acute Lymphoblastic Leukemia image database |
| C-NMC 2019 | all, hem | B-lineage ALL cell classification challenge |

*Note: Raw dataset files and large `.pth` model weights are excluded from this repository.*

---

## 🗂️ Repository Structure

```text
Neuro Symbolic Ai/
├── main.py                  # Core Neuro-Symbolic diagnosis pipeline
├── symbolic_logic.py        # Logic Bridge (ChromaDB + SentenceTransformers)
├── data_preprocessing.py    # Data standardization, augmentation & K-Fold splitting
├── train_baseline.py        # EfficientNet-B0 PyTorch training script
├── evaluate_performance.py  # Calculates Metrics (ROC, Confusion Matrix, Latency)
├── grad_heatmap.py          # Generates Grad-CAM XAI visual explanations
│
├── medical_rules.txt        # Symbolic knowledge base of medical rules
├── Figure_6_Confusion_Matrix.png # Evaluation output sample
├── Figure_7_ROC_Curve.png        # Evaluation output sample
├── explanation_result.png        # Grad-CAM heatmap sample
│
└── README.md
```

---

## ⚙️ Pipeline

```text
1. Preprocessing    →  data_preprocessing.py (Augmentation & 5-Fold Splitting)
2. Training         →  train_baseline.py     (EfficientNet-B0 Fine-Tuning)
3. Rule Ingestion   →  symbolic_logic.py     (Embed medical_rules.txt into ChromaDB)
4. Evaluation       →  evaluate_performance.py (Accuracy, ROC, Logic Guardrail Stats)
5. XAI Heatmaps     →  grad_heatmap.py       (Grad-CAM Visualizations)
6. Live Diagnosis   →  main.py               (End-to-End Neuro-Symbolic Inference)
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python pandas numpy seaborn matplotlib scikit-learn chromadb sentence-transformers grad-cam tqdm Pillow
```

### 1. Data Preprocessing

Place your raw datasets in the project directory, then run:

```bash
python data_preprocessing.py
```

### 2. Train the Neural Model

Train the baseline EfficientNet-B0 model:

```bash
python train_baseline.py
```

### 3. Run Evaluation Metrics

Test the model's accuracy, speed, and the Logic Guardrail rejection rate:

```bash
python evaluate_performance.py
```

### 4. Run the Neuro-Symbolic Pipeline

Run an end-to-end diagnosis on a sample image:

```bash
python main.py
```

### 5. Generate Grad-CAM Heatmaps

Visualize where the neural network is focusing its attention:

```bash
python grad_heatmap.py
```

---

## 📬 Author

**Vijay** — [GitHub](https://github.com/Vijay42-hs)
