# 🧠 Diabetes Prediction with Explainable Machine Learning

This project builds a diabetes risk prediction model using structured health data and explains the model’s predictions using SHAP (SHapley Additive exPlanations). It uses XGBoost for classification and SHAP for model interpretability.

---

## 🎯 Objective

- Predict the likelihood of diabetes using health metrics
- Use XGBoost for high-accuracy classification
- Explain model behavior using SHAP visualizations

---

## 📊 Dataset

- **Source**: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768
- **Features**: Age, Glucose, BMI, Insulin, etc.
- **Target**: 1 = Diabetic, 0 = Non-Diabetic

---

## 🛠️ Tools & Libraries

- Python
- Jupyter Notebook
- XGBoost
- SHAP
- scikit-learn
- seaborn & matplotlib

---

## 📈 Project Workflow

1. **Data Cleaning**: Replaced 0s with NaN and filled using median.
2. **EDA**: Plotted distributions and feature correlation.
3. **Modeling**: Trained XGBoost classifier with 76% accuracy.
4. **Evaluation**: Used confusion matrix, precision, recall, and F1-score.
5. **Explainability**: Visualized SHAP values to explain predictions.

---

## 🔍 SHAP Output

- **Glucose**, **BMI**, and **Age** were top predictors.
- SHAP beeswarm plot shows how each feature influenced predictions.

---

## 📁 Folder Contents

| File/Folder                 | Purpose                                      |
|-----------------------------|----------------------------------------------|
| `clinical_model.ipynb`      | Main notebook with full pipeline             |
| `diabetes.csv`              | Dataset used for training                    |
| `model.pkl`                 | Trained XGBoost model                        |
| `shap_plot.png` *(optional)*| SHAP feature importance plot (saved image)  |
| `requirements.txt`          | All required Python libraries                |

---

## ▶️ How to Run

1. Clone the repo or download the folder  
2. Install dependencies:




pip install -r requirements.txt


3. Open `clinical_model.ipynb` in Jupyter Notebook   
4. Run all cells to see:
- EDA
- XGBoost predictions
- Confusion matrix
- SHAP explanation




---

## 📌 Future Scope


- Integrate into a Flask/Streamlit web app
- Add image-based prediction (Stage 2: Retinal scan using CNN)
- Enable real-time inference and API deployment
- Combine with Explainable Deep Learning for vision + tabular fusion


