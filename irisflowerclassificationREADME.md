# Iris Flower Classification using Machine Learning

## Overview

This project demonstrates a complete machine learning workflow using the famous Iris dataset. The objective is to classify iris flowers into one of three species based on their sepal and petal measurements.

The project covers:

* Loading and exploring the dataset
* Creating and preprocessing a Pandas DataFrame
* Splitting data into training and testing sets
* Training machine learning models
* Evaluating model performance

Two classification algorithms are implemented:

1. Logistic Regression
2. Random Forest Classifier

---

## Dataset

The project uses the Iris dataset available through Scikit-learn.

### Dataset Characteristics

* Total Samples: 150
* Number of Features: 4
* Number of Classes: 3

### Features

| Feature      | Description           |
| ------------ | --------------------- |
| Sepal Length | Length of sepal in cm |
| Sepal Width  | Width of sepal in cm  |
| Petal Length | Length of petal in cm |
| Petal Width  | Width of petal in cm  |

### Target Classes

| Class Label | Species         |
| ----------- | --------------- |
| 0           | Iris Setosa     |
| 1           | Iris Versicolor |
| 2           | Iris Virginica  |

---

## Technologies Used

* Python
* Pandas
* Scikit-learn

### Libraries

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
```

---

## Project Workflow

### 1. Load Dataset

The Iris dataset is loaded using:

```python
iris = load_iris()
```

---

### 2. Create DataFrame

A Pandas DataFrame is created using the dataset features:

```python
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
```

---

### 3. Add Target Variable

The target labels are added as a new column:

```python
df['target'] = iris.target
```

---

### 4. Feature and Target Separation

```python
X = df.drop('target', axis=1)
y = df['target']
```

---

### 5. Train-Test Split

Dataset split:

* Training Data: 80%
* Testing Data: 20%

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 6. Logistic Regression Model

Model Training:

```python
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
```

Model Evaluation:

```python
accuracy = accuracy_score(y_test, y_pred)
```

### Result

```text
Accuracy = 1.00
```

---

### 7. Random Forest Model

Model Training:

```python
rand_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rand_forest.fit(X_train, y_train)
```

Model Evaluation:

```python
classification_report(
    y_test,
    y_pred_rf
)
```

### Classification Report

```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

---

## Results Summary

| Model                    | Accuracy |
| ------------------------ | -------- |
| Logistic Regression      | 100%     |
| Random Forest Classifier | 100%     |

Both models achieved perfect classification accuracy on the testing dataset.

---

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Install dependencies:

```bash
pip install pandas scikit-learn
```

3. Run the Jupyter Notebook or Python script.

---

## Project Structure

```text
Iris-Classification/
│
├── iris_classification.ipynb
├── README.md
└── requirements.txt
```

---

## Learning Outcomes

Through this project, the following machine learning concepts were explored:

* Data loading and preprocessing
* Exploratory data handling with Pandas
* Train-test data splitting
* Classification using Logistic Regression
* Ensemble learning using Random Forest
* Model evaluation using Accuracy Score and Classification Report

---

## Author

Developed as part of a Machine Learning laboratory exercise using the Iris Flower Dataset and Scikit-learn.
