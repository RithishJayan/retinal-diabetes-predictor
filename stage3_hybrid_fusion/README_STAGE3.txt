# Stage 3 – Hybrid Fusion Model

This folder contains the final fusion stage for the diabetic retinopathy detection system. This stage combines predictions from:

- Stage 1 (clinical ML model)
- Stage 2 (CNN retinal image model)

## Expected Files:

- `fusion_model.ipynb`  
  → Fusion model code that loads predictions from previous stages and applies Random Forest.

- `fusion_output.png`  
  → Confusion matrix image of the final predictions.

- `val_indices.pkl`  
  → Used to align clinical and image predictions by patient index.

## Description:

This stage uses the aligned predictions from both previous stages to train a Random Forest classifier that outputs the final diabetic diagnosis.

It mimics the real-world decision process where doctors rely on both test results and image reports for final decisions.
