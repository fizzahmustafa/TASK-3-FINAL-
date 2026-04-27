# Task 3: Heart Disease Prediction

## Objective
Build a binary classification model to predict whether a person is **at risk of heart disease** based on clinical health data. Evaluate model performance using accuracy, ROC curve, and confusion matrix.

---

## Dataset
**Heart Disease UCI Dataset** — Available on [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) and the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease).

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia type |
| `target` | **Target** — 1 = Heart disease, 0 = No heart disease |

---

## Requirements

Install the required libraries before running:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Project Structure

```
task3_heart_disease/
│
├── heart.csv                   # Dataset (download from Kaggle)
├── heart_disease_model.py      # Main script
├── README.md                   # This file
└── plots/
    ├── eda_distributions.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## Step-by-Step Instructions

### Step 1 — Load and Clean the Dataset

```python
import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')

print("Shape:", df.shape)
print("\nFirst rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nClass distribution:")
print(df['target'].value_counts())
```

**Handling Missing Values (if any):**

```python
# Fill numeric columns with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values after cleaning:", df.isnull().sum().sum())
```

### Step 2 — Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of key features by target class
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
features_to_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'cp']

for ax, feature in zip(axes.flatten(), features_to_plot):
    for target_val, label in [(0, 'No Disease'), (1, 'Heart Disease')]:
        subset = df[df['target'] == target_val]
        ax.hist(subset[feature], bins=20, alpha=0.6, label=label)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend()

plt.suptitle('Feature Distributions by Heart Disease Status', fontsize=14)
plt.tight_layout()
plt.savefig('plots/eda_distributions.png', dpi=150)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.show()
```

### Step 3 — Prepare Features and Split Data

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
```

### Step 4A — Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['No Disease', 'Heart Disease']))
```

### Step 4B — Decision Tree Model

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print("=== Decision Tree ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt, target_names=['No Disease', 'Heart Disease']))
```

### Step 5 — Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (model_name, y_pred) in zip(axes, [
    ("Logistic Regression", y_pred_lr),
    ("Decision Tree", y_pred_dt)
]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Disease', 'Heart Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{model_name} — Confusion Matrix')

plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.show()
```

### Step 6 — ROC Curve

```python
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(8, 6))

for model_name, y_prob in [
    ("Logistic Regression", y_prob_lr),
    ("Decision Tree", y_prob_dt)
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Heart Disease Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.show()
```

### Step 7 — Feature Importance

```python
# Logistic Regression coefficients
lr_coef = pd.Series(np.abs(lr.coef_[0]), index=X.columns)
lr_coef = lr_coef.sort_values(ascending=False)

# Decision Tree importances
dt_importance = pd.Series(dt.feature_importances_, index=X.columns)
dt_importance = dt_importance.sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=lr_coef.values, y=lr_coef.index, palette='coolwarm', ax=axes[0])
axes[0].set_title('Logistic Regression — Feature Coefficients')
axes[0].set_xlabel('|Coefficient|')

sns.barplot(x=dt_importance.values, y=dt_importance.index, palette='viridis', ax=axes[1])
axes[1].set_title('Decision Tree — Feature Importance')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()
```

---

## Key Observations

- **`thalach`** (max heart rate) and **`cp`** (chest pain type) are consistently among the strongest predictors of heart disease.
- **`ca`** (number of major vessels) and **`oldpeak`** (ST depression) also show high importance.
- A higher maximum heart rate surprisingly correlates with **lower** heart disease risk in this dataset.
- Logistic Regression and Decision Tree tend to achieve **80–88% accuracy** on this dataset.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Percentage of correct predictions overall |
| **Precision** | Of predicted positives, how many are truly positive |
| **Recall** | Of actual positives, how many were correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve — higher is better (max = 1.0) |
| **Confusion Matrix** | Full breakdown of TP, TN, FP, FN |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, evaluation, and metrics |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualization |

---

## How to Run

1. Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
2. Place it in the project folder
3. Run the script:

```bash
python heart_disease_model.py
```

All evaluation plots will be saved to the `plots/` folder.
