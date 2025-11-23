# Student Depression Prediction using SVM

A machine learning project that uses Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel to predict student depression based on various academic, lifestyle, and demographic factors.

## 📋 Overview

This project aims to build a classification model to identify students at risk of depression using a comprehensive dataset. The model employs an SVM classifier with RBF kernel and evaluates feature importance to understand which factors most significantly impact student mental health.

## 🎯 Objectives

- Build an SVM model to classify students with or without depression
- Evaluate model performance using ROC AUC score
- Analyze feature importance to identify key factors affecting student depression
- Visualize results and provide insights into depression risk factors

## 📊 Dataset

The dataset (`student_depression_dataset.csv`) contains information about students including:
- **Demographics**: Age, Gender, City
- **Academic**: CGPA, Academic Pressure, Study Satisfaction, Degree
- **Lifestyle**: Sleep Hours, Dietary Habits, Work/Study Hours
- **Mental Health**: Suicidal Thoughts, Family History of Mental Illness
- **Financial**: Financial Stress
- **Work**: Work Pressure, Job Satisfaction
- **Target Variable**: Depression (Binary: 0 = No Depression, 1 = Depression)

## 🔧 Methodology

### Data Preprocessing
1. **Data Cleaning**:
   - Removed ID column (non-predictive)
   - Converted "Sleep Duration" text to numeric "Sleep Hours"
   - Handled missing values using median imputation for numeric features
   - Converted categorical variables to numeric format

2. **Feature Engineering**:
   - **Numeric Features**: Standardized using StandardScaler
   - **Categorical Features**: One-hot encoded
   - Constant features automatically detected and removed

3. **Data Splitting**:
   - Train-Test Split: 80-20
   - Stratified sampling to maintain class distribution
   - Random state: 42 for reproducibility

### Model
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Hyperparameters**:
  - C = 0.1
  - gamma = 0.01
  - max_iter = 10000
  - probability = True (for ROC AUC calculation)

### Feature Importance Analysis
- **Permutation Importance**: Measures the decrease in ROC AUC when each feature is randomly shuffled
- **Mean Difference Analysis**: Determines the direction of relationship (Higher Risk vs Lower Risk) by comparing mean feature values between depression and no-depression groups

## 📈 Results

### Model Performance
- **Training Accuracy**: 84.86%
- **Test Accuracy**: 84.30%
- **Training ROC AUC**: 0.9224
- **Test ROC AUC**: 0.9186

### Classification Report
```
              precision    recall  f1-score   support

           0       0.83      0.78      0.81      2313
           1       0.85      0.88      0.87      3268

    accuracy                           0.84      5581
```

### Key Findings
The model identifies several important factors for depression prediction:
- Academic and financial pressures
- Suicidal thoughts history
- Age and work/study hours
- Dietary habits
- Study satisfaction

## 🚀 Installation

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 💻 Usage

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Project.ipynb
   ```

2. **Run the cells sequentially**:
   - Cell 0: Data loading and initial preprocessing
   - Cell 1: Feature definition and constant feature handling
   - Cell 2: Preprocessing pipeline setup
   - Cell 3: Train-test split and preprocessing
   - Cell 4: SVM model training (with optimized parameters)
   - Cell 5: ROC AUC evaluation and visualization
   - Cell 6-8: Feature importance analysis and visualization

## 📁 Project Structure

```
code/
├── Project.ipynb                    # Main Jupyter notebook
├── student_depression_dataset.csv    # Dataset
├── README.md                        # Project documentation
└── .gitignore                       # Git ignore file
```

## 🔍 Key Features

- **Robust Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **Model Evaluation**: Comprehensive metrics including accuracy, ROC AUC, classification report, and confusion matrix
- **Feature Analysis**: Permutation importance with direction analysis (Higher Risk vs Lower Risk)
- **Visualizations**: ROC curves and feature importance bar charts

## 📝 Notes

- The model uses stratified train-test split to ensure balanced class distribution
- Missing values are handled through the preprocessing pipeline to prevent data leakage
- Constant features are automatically detected and removed
- The RBF kernel SVM provides good performance for this non-linear classification problem

## 👥 Author

Student Project - SDSC3006

## 📄 License

This project is for educational purposes.

