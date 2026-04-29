# Neuro-Symbolic Logic for Leukemia Diagnosis

This repository contains a Neuro-Symbolic Artificial Intelligence system designed to diagnose Leukemia from blood smear images. The system combines the pattern recognition strengths of deep learning (Neural) with the interpretability and reasoning of symbolic logic (Symbolic) based on medical knowledge rules.

## Overview

The system uses a hybrid approach:
1. **Neural Component (Deep Learning):** An `EfficientNet-B0` model acts as the primary classifier, quickly predicting the class of the image (e.g., Benign, Early, Pre, Pro).
2. **Visual Feature Extractor (Computer Vision):** OpenCV is used to extract explicit, measurable physical properties from the cell nuclei, such as size, circularity, nucleoli presence, and chromatin texture.
3. **Symbolic Verification (RAG / Logic Bridge):** Using a Vector Database (`ChromaDB`) and `SentenceTransformers`, the system retrieves established medical rules (e.g., "Pro-B cells have large size and prominent nucleoli") and verifies if the neural prediction matches the explicit visual features extracted from the image. 

If the logic matches, the system approves the diagnosis. If it contradicts medical rules, it flags the decision for human review.

## Features

- **Data Preprocessing:** Standardized augmentation and 5-Fold cross-validation splits for both ALL_IDB and C-NMC datasets.
- **Baseline Training:** Training scripts for `EfficientNet-B0` with performance logging.
- **Neuro-Symbolic Pipeline:** An end-to-end diagnosis pipeline combining the Neural model and the Logic Bridge.
- **Explainability (XAI):** `Grad-CAM` visualizations to show where the AI model is focusing its attention, alongside logic-based explanations.
- **Evaluation Engine:** Extensive evaluation tools providing classification reports, confusion matrices, ROC-AUC curves, and "Logic Guardrail" intervention statistics.

## Project Structure

- `main.py`: The core pipeline executing the Neuro-Symbolic diagnosis on test images.
- `symbolic_logic.py`: Implements the `MedicalLogicBridge`, converting text rules into vector embeddings and verifying visual features against retrieved rules.
- `data_preprocessing.py`: Handles data standardization, synthetic image generation (balancing), and 5-Fold split metadata creation.
- `train_baseline.py`: PyTorch training script for the EfficientNet-B0 baseline model.
- `evaluate_performance.py`: Generates comprehensive metrics (ROC, Confusion Matrix, Inference Speed, Logic Rejection Rate).
- `grad_heatmap.py`: Generates visual Grad-CAM explanations for model predictions.
- `medical_rules.txt`: The text file containing human-readable medical rules ingested by the Symbolic Logic engine.

## Note on Datasets

This system is configured to work with standard Leukemia datasets (e.g., `ALL_IDB` and `C-NMC 2019`). Due to size constraints and privacy, the raw datasets and trained `.pth` model weights are not included in this repository. You must provide your own data and run the preprocessing/training scripts to generate the models.
