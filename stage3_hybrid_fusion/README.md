 ---Retinal Diabetes Predictor — Hybrid ML + Deep Learning System---

This project is an advanced medical machine learning system designed to predict Diabetic Retinopathy (DR) using both clinical health records and retinal fundus images. By combining structured clinical features and image-based deep learning, this pipeline replicates how doctors analyze both test results and eye scans for early DR diagnosis.


PROJECT SUMMARY:
Diabetic Retinopathy (DR) is a leading cause of preventable blindness. Accurate early detection is critical. However, relying on a single type of data (either clinical or image) can result in misdiagnosis due to data noise or variability.

This project proposes a three-stage hybrid architecture that mimics real-world medical diagnosis. The stages include:

Clinical Model – Uses machine learning (Logistic Regression or XGBoost) to analyze patient health metrics.

Image Model – A Convolutional Neural Network (CNN) processes retinal images to detect signs of retinopathy.

Fusion Model – Combines predictions from both sources using a Random Forest classifier to produce a final, more accurate diagnosis.

This hybrid design enhances predictive power by capturing both quantitative and visual patterns of disease progression.


PROBLEM STATEMENT:
Build a system to predict diabetic retinopathy by:

Using clinical patient data (age, BMI, glucose levels, etc.).

Using retinal image analysis through CNNs.

Combining both for robust hybrid predictions.



STAGES IN DETAIL:
🔹 Stage 1 – Clinical ML Model
Input: CSV file with features like Glucose, BMI, Insulin, Age, Pregnancies

Model: Logistic Regression (with optional XGBoost)

Balancing: Class weights used to address imbalance

Output: Binary prediction (0 = No DR, 1 = DR)

Goal: Establish a baseline risk score using non-visual data

🔹 Stage 2 – CNN Image Model
Input: Retinal fundus images (preprocessed and resized)

Model: Custom CNN built with Keras

Loss Function: Focal Loss (to tackle imbalance)

Output: Image prediction score (0 or 1)

Challenge Solved: Visual detection of DR markers (e.g., hemorrhages, exudates)

🔹 Stage 3 – Fusion Model
Input: Concatenated outputs from Stage 1 & 2

Validation Set: Carefully matched via val_indices.pkl

Model: Random Forest classifier

Output: Final prediction with enhanced accuracy

Why It Works: Fuses numerical and visual understanding for a holistic result


DATASET USED:
CLINICAL DATA:
Pima Indian Diabetes Dataset (Kaggle)


RETINAL IMAGES:
APTOS 2019 Blindness Detection (Kaggle)


PERFORMANCE AND EVALUATION:
Stage 1: Clinical model performs decently but lacks visual evidence.

Stage 2: CNN shows good pattern recognition but overfits with limited data.

Stage 3: Fusion significantly boosts performance by balancing both modalities.

Visual outputs like confusion matrices and ROC curves are saved for each stage.



How to Run the Project
Install requirements
Run:

nginx
Copy
Edit
pip install -r requirements.txt
Run in order:

Stage 1: clinical_model.ipynb

Stage 2: image_model.ipynb

Stage 3: fusion_model.ipynb

Ensure the val_indices.pkl file aligns the validation patients across stages.


EXAMPLE VISUALS:
Confusion matrix of final fusion output

Accuracy/loss plots of CNN model

ROC-AUC values of each model
(Saved in respective stage folders)


FUTURE ENHANCEMENT:
Add Grad-CAM to explain CNN predictions

Use weighted soft-voting or neural fusion layers

Deploy using Streamlit or Flask as a web app

Add a clinical interface for doctors to upload data and images

Integrate with EHR systems for live clinical use


KNOWN LIMITATIONS:
Limited dataset size in image model

Overfitting observed in CNN without regularization

Fusion requires precise alignment of matching patients across datasets


