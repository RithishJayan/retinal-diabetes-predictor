# Stage 2 – Retinal Image-Based Prediction (CNN)

This folder contains all files related to the second stage of the diabetic retinopathy prediction pipeline. Stage 2 focuses on analyzing retinal fundus images using a Convolutional Neural Network (CNN).

## Expected Files to Upload Here:

- `image_model.ipynb`  
  → Jupyter notebook that trains the CNN on preprocessed fundus images.

- `image_preds.npy`  
  → Saved prediction probabilities from the CNN model on the validation/test set.

- `retina_images/` (optional)  
  → Folder containing preprocessed retinal images used for training and testing.

## Description

The CNN in this stage was designed to identify signs of diabetic retinopathy directly from image data, such as microaneurysms, hemorrhages, and exudates. Focal loss was used to counter class imbalance during training.

This stage outputs a probability score per image which is then used as input for the fusion model in Stage 3.

