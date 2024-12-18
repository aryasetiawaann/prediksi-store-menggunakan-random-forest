# Stroke Prediction Using Random Forest Algorithm

This repository contains the implementation of a machine learning project aimed at predicting the likelihood of stroke occurrences based on patient data. The model employs the **Random Forest Algorithm** to provide robust and interpretable predictions.

## Project Overview

- **Title:** Stroke Prediction Using Random Forest
- **Course:** Machine Learning (IF540-L)
- **Semester:** Gasal 2024/2025
- **Dataset Source:** [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

## Dataset Description

The dataset contains detailed patient information to predict stroke occurrences. Each row provides attributes such as age, gender, BMI, glucose levels, and medical history. Below is a brief overview of the dataset:

| **Attribute**       | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| `id`                | Unique identifier                                                              |
| `gender`            | Male, Female, or Other                                                         |
| `age`               | Age of the patient                                                             |
| `hypertension`      | 0 = No hypertension, 1 = Has hypertension                                       |
| `heart_disease`     | 0 = No heart disease, 1 = Has heart disease                                     |
| `ever_married`      | Yes or No                                                                      |
| `work_type`         | Type of employment (e.g., Private, Govt_job, Self-employed, etc.)               |
| `Residence_type`    | Rural or Urban                                                                 |
| `avg_glucose_level` | Average blood glucose level                                                    |
| `bmi`               | Body Mass Index                                                               |
| `smoking_status`    | Smoking status (formerly smoked, never smoked, smokes, or unknown)             |
| `stroke`            | Target variable (1 = Stroke occurred, 0 = No stroke)                          |

## Methodology

1. **Data Preparation:**
   - Removed irrelevant features (`id`).
   - Handled missing values using mean imputation for the `bmi` feature.
   - Encoded categorical variables into numerical representations using `LabelEncoder`.
   - Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.

2. **Exploratory Data Analysis (EDA):**
   - Visualized the distribution of features and their correlation with stroke occurrence.
   - Identified key risk factors: age, hypertension, heart disease, and glucose levels.

3. **Model Training:**
   - Used `RandomForestClassifier` from scikit-learn.
   - Fine-tuned hyperparameters with `GridSearchCV` for optimal performance.
   - Partitioned data into training and testing sets with and without oversampling.

4. **Evaluation:**
   - Compared model performance with and without oversampling.
   - Metrics: Accuracy, Precision, Recall, F1-Score.
   - Displayed results using confusion matrices and detailed classification reports.

## Results

- **Model Without Oversampling:**
  - Training Accuracy: **85.4%**
  - Test Accuracy: **83.8%**
  - Struggled with minority class (stroke cases).

- **Model With Oversampling:**
  - Training Accuracy: **100%**
  - Test Accuracy: **94.2%**
  - Significantly improved recall and precision for the minority class.

### Confusion Matrices

| Without Oversampling | With Oversampling  |
|-----------------------|--------------------|
| ![Confusion Matrix 1](path/to/matrix1.png) | ![Confusion Matrix 2](path/to/matrix2.png) |

## Key Insights

1. **Age** is the strongest predictor of stroke occurrences.
2. **Hypertension**, **heart disease**, and **glucose levels** are significant risk factors.
3. The use of oversampling with SMOTE greatly improved the model's ability to classify minority cases accurately.
