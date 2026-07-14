# 🫀 Heart Disease Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help doctors make timely clinical decisions and improve patient outcomes.

This project uses Machine Learning to predict the likelihood of heart disease based on clinical attributes from the Cleveland Heart Disease Dataset. Multiple preprocessing techniques and hyperparameter tuning were applied to improve prediction performance.

---

## 🎯 Objectives

- Predict the presence of heart disease using patient health records.
- Perform data preprocessing and feature scaling.
- Train and evaluate a Random Forest classifier.
- Compare prediction performance using multiple evaluation metrics.
- Save the trained model for future predictions.

---

## 📂 Dataset

- **Dataset:** Cleveland Heart Disease Dataset
- **Target Variable:** `target`
- **Number of Features:** 13
- **Prediction Type:** Binary Classification

Features include:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Jupyter Notebook

---

## ⚙️ Machine Learning Pipeline

1. Data Loading
2. Data Preprocessing
3. Missing Value Handling
4. Feature Scaling
5. Train-Test Split
6. Hyperparameter Tuning using GridSearchCV
7. Random Forest Classification
8. Model Evaluation
9. Model Saving

---
## 🤖 Machine Learning Models Compared

The following classification algorithms were implemented and evaluated:

| Algorithm | Accuracy |
|-----------|---------:|
| Random Forest | **90.16%** |
| K-Nearest Neighbors (KNN) | 85.25% |
| Support Vector Machine (SVM) | 86.89% |
| Decision Tree | 83.61% |
| Logistic Regression | 88.52% |
| Naive Bayes | 81.97% |

🏆 **Random Forest achieved the highest accuracy and ROC-AUC score, making it the final selected model.**

## 📊 Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **90.16%** |
| ROC-AUC | **0.9481** |
| Precision | **0.90** |
| Recall | **0.90** |
| F1-Score | **0.90** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| No Heart Disease (0) | 0.94 | 0.88 | 0.91 |
| Heart Disease (1) | 0.87 | 0.93 | 0.90 |

---

## 📈 Visualizations

The notebook contains:

- Target Distribution
- Correlation Heatmap
- Confusion Matrix
- ROC Curve
- Feature Importance

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── dataset/
│   └── Heart_disease_cleveland_new.csv
│
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb
│
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── images/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
└── HeartDiseasePrediction.pdf
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Salma1612/Heart-Disease-Prediction.git
```

Move into the project directory:

```bash
cd Heart-Disease-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Train the model:

```bash
python train.py
```

Predict using the trained model:

```bash
python predict.py
```

---

## 📦 Saved Models

The trained models are stored inside the **models/** directory.

- random_forest.pkl
- scaler.pkl

---

## 📌 Future Improvements

- Deploy as a Flask or Streamlit web application.
- Compare additional machine learning algorithms.
- Perform feature engineering.
- Integrate deep learning models.
- Deploy on cloud platforms.

---

## 👩‍💻 Authors

**Shaik Salma**



---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share it with others

---
