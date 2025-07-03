# 📊 Student Performance Prediction

This project uses machine learning models to predict student academic outcomes based on various social, behavioral, and school-related features. It applies multiple classification algorithms and resampling techniques (SMOTE) to handle class imbalance and evaluate model performance.

---

## 🧠 Objective

To predict whether a student will **pass (1)** or **fail (0)** using historical student data, leveraging multiple machine learning classifiers and evaluating them based on precision, recall, F1-score, and accuracy.

---

## 📂 Project Structure

```
student-performance-predictor/
│
├── dataset/
│   └── student-mat.csv
│
├── logistic_model/
│   ├── preprocessing_and_training.ipynb
│   ├── preprocessor.pkl
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---


## 📁 Dataset

- **Source**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **File Used**: `student-mat.csv` (Math course performance)
- **Target Variable**: Pass/Fail (based on final grade `G3`)

### 🔍 Features Used
Categorical and numerical features such as:
- Study time, travel time, family support, paid classes  
- Internet access, romantic relationship, absences  
- Daily and weekly alcohol consumption, health, etc.

---

## ⚙️ Workflow

- Data Preprocessing  
- Handling categorical features with One-Hot Encoding
- Train-test split with stratification  
- Handling Imbalanced Data using **SMOTE**
---

## 🤖 Models Used

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)  

---

## 📐 Evaluation Metrics

- Accuracy  
- F1-Score (per class)  
- Confusion Matrix  
- Classification Report  

---

## 📈 Results Snapshot

| Model           | Accuracy | F1-Score (Class 0) | F1-Score (Class 1) |
|-----------------|----------|--------------------|--------------------|
| Logistic        | 0.66     | 0.37               | 0.77               |
| KNN             | 0.66     | 0.18               | 0.78               |
| Decision Tree   | 0.68     | 0.44               | 0.78               |
| Random Forest   | 0.68     | 0.39               | 0.79               |
| SVM             | 0.54     | 0.25               | 0.67               |


## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- Seaborn & Matplotlib (for visualizations)  
- Jupyter Notebook  

---

## 🚀 Streamlit Web App

The project includes a user-friendly **Streamlit app** to predict a student's performance.

### Features:
- 📋 Input student features like study time, failures, family support, alcohol consumption, etc.
- 🔄 Choose from multiple models: Logistic Regression, Decision Tree, Random Forest, SVM, or KNN.
- 🎯 Predict if the student is likely to **pass** or **fail**.
- 📈 Get probability of passing if supported by the model.

---

## 🌐 Live Demo

Try the app here: [Student Performance Predictor](https://smart-grade-predictor.onrender.com/)

---


## 📌 Future Work
- Add performance metrics and confusion matrix to the app
- Deploy app on Streamlit Cloud
- Expand dataset with `student-por.csv` for more robust training

