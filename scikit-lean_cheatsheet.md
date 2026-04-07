# Scikit-Learn Cheatsheet

> [!TIP]
> Scikit-learn (`sklearn`) is Python's go-to library for classical Machine Learning. Every function here follows the same pattern: **fit → transform/predict**. Master that and everything else clicks.

---

## 📋 Table of Contents
1. [Imports & Setup](#1-imports--setup)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Train / Test Split](#3-train--test-split)
4. [Model Selection — Picking the Right Model](#4-model-selection--picking-the-right-model)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Pipelines](#6-pipelines)
7. [Evaluation Metrics Quick Reference](#7-evaluation-metrics-quick-reference)
8. [Saving & Loading Models](#8-saving--loading-models)
9. [Q&A — Test Yourself!](#9-qa--test-yourself)

---

## 1. Imports & Setup

```python
# Core sklearn modules (import only what you need)
from sklearn import datasets                   # Built-in toy datasets
from sklearn.model_selection import (
    train_test_split,                          # Split data
    cross_val_score,                           # K-Fold cross-validation
    GridSearchCV, RandomizedSearchCV           # Hyperparameter tuning
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler,              # Feature scaling
    LabelEncoder, OneHotEncoder,               # Categorical encoding
    PolynomialFeatures,                        # Feature engineering
    SimpleImputer                              # Fill missing values
)
from sklearn.pipeline import Pipeline          # Chain steps together
from sklearn.metrics import (
    accuracy_score, confusion_matrix,          # Classification
    mean_squared_error, r2_score               # Regression
)
import joblib                                  # Save/load models
```

---

## 2. Data Preprocessing

Preprocessing transforms raw data into a form that algorithms can learn from effectively.

---

### 2.1 Handling Missing Values — `SimpleImputer`

> **Why?** Most algorithms crash on `NaN`. `SimpleImputer` fills gaps with a chosen strategy.

```python
from sklearn.preprocessing import SimpleImputer
import numpy as np

X = np.array([[1, 2],
              [np.nan, 3],
              [7, np.nan]])

# Strategy options: 'mean', 'median', 'most_frequent', 'constant'
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)          # Learn the mean of each column from training data
X_clean = imputer.transform(X)

# X_clean:
# [[1. , 2. ],
#  [4. , 3. ],   ← NaN replaced by column mean (1+7)/2 = 4
#  [7. , 2.5]]   ← NaN replaced by column mean (2+3)/2 = 2.5
```

| Strategy | Replaces NaN with |
|---|---|
| `mean` | Column average (numeric) |
| `median` | Column median (numeric, robust to outliers) |
| `most_frequent` | Most common value (works on text too) |
| `constant` | A fixed value you specify |

> [!NOTE]
> Always `fit` on **training data only**, then `transform` both train and test. Fitting on the full dataset leaks information.

---

### 2.2 Feature Scaling

Algorithms that rely on **distance** (KNN, SVM, PCA, gradient descent) are sensitive to feature magnitudes. Scaling puts all features on a comparable range.

#### `StandardScaler` — Zero mean, unit variance (Z-score normalisation)

**Formula:**

$$z = \frac{x - \mu}{\sigma}$$

- $\mu$ = column mean, $\sigma$ = column standard deviation
- Result: ~68% of values fall between -1 and +1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform in one step
X_test_scaled  = scaler.transform(X_test)         # ONLY transform (never re-fit on test)

# scaler.mean_  → learned means
# scaler.scale_ → learned std devs
```

#### `MinMaxScaler` — Squeezes values into [0, 1]

**Formula:**

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

mm_scaler = MinMaxScaler()                        # feature_range=(0, 1) by default
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)
```

| Scaler | Use when |
|---|---|
| `StandardScaler` | Data is roughly Gaussian; SVM, Logistic Reg, PCA |
| `MinMaxScaler` | You need values in a fixed range (e.g. image pixels, neural networks) |

---

### 2.3 Encoding Categorical Variables

Algorithms work on **numbers**, not strings. Encoding converts categories to numbers.

#### `LabelEncoder` — Ordinal integer mapping (for target `y`)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = ['cat', 'dog', 'cat', 'bird']
y_encoded = le.fit_transform(y)   # [1, 2, 1, 0]

# le.classes_ → ['bird', 'cat', 'dog']
# Reverse: le.inverse_transform([1, 2]) → ['cat', 'dog']
```

> [!WARNING]
> Do **not** use `LabelEncoder` on input features `X` — it implies a false ordering (0 < 1 < 2). Use `OneHotEncoder` instead.

#### `OneHotEncoder` — Creates binary columns per category (for features `X`)

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse_output=False)   # sparse_output=False → dense array
X_cat = [['red'], ['blue'], ['green'], ['red']]
X_encoded = enc.fit_transform(X_cat)

# Result (columns: blue, green, red):
# [[0, 0, 1],
#  [1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# drop='first' removes one column to avoid multicollinearity
enc_drop = OneHotEncoder(drop='first', sparse_output=False)
```

---

### 2.4 Polynomial Features — `PolynomialFeatures`

> **Why?** Turns a linear model non-linear by adding feature interactions and powers.

If $X = [x_1, x_2]$ and `degree=2`:

$$\text{output} = [1,\ x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2]$$

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)   # X shape (n,2) → X_poly shape (n,5)

# poly.get_feature_names_out() → shows what each column represents
```

> [!NOTE]
> Higher degree = more features = risk of **overfitting**. Pair with regularisation.

---

## 3. Train / Test Split

Splitting data ensures you evaluate a model on data it has **never seen** during training.

### `train_test_split`

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducible split
    stratify=y            # Preserve class proportions (classification)
)

# Shapes (example with 1000 rows):
# X_train: (800, n_features), X_test: (200, n_features)
```

> [!IMPORTANT]
> Always split **before** any preprocessing. Fit scalers/imputers on training data only — never on the full dataset.

---

### K-Fold Cross-Validation — `cross_val_score`

> **Why?** A single train/test split is noisy. K-Fold trains and evaluates on K different folds and averages the score — much more reliable.

```
Data → |  Fold 1  |  Fold 2  |  Fold 3  |  Fold 4  |  Fold 5  |
Run 1:  [TEST]      [TRAIN]    [TRAIN]    [TRAIN]    [TRAIN]
Run 2:  [TRAIN]    [TEST]      [TRAIN]    [TRAIN]    [TRAIN]
Run 3:  [TRAIN]    [TRAIN]    [TEST]      [TRAIN]    [TRAIN]
...
Final score = average of 5 scores
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(scores)           # [0.91, 0.88, 0.93, 0.90, 0.92]
print(scores.mean())    # 0.908  ← use this as your performance estimate
print(scores.std())     # 0.017  ← low std = stable model
```

| cv value | Typical use |
|---|---|
| 5 | General default |
| 10 | When you want more reliable estimates |
| `StratifiedKFold` | Classification with imbalanced classes |

---

## 4. Model Selection — Picking the Right Model

### 4.1 Common Model Imports

```python
# --- Regression ---
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# --- Classification ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --- Clustering (Unsupervised) ---
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# --- Dimensionality Reduction ---
from sklearn.decomposition import PCA
```

---

### 4.2 The Universal API — fit / predict

Every sklearn model follows the same 3-step pattern:

```python
# Step 1: Instantiate with hyperparameters
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 2: Train
model.fit(X_train, y_train)

# Step 3: Predict
y_pred = model.predict(X_test)          # Hard class labels
y_prob = model.predict_proba(X_test)    # Probability per class (if supported)
```

---

### 4.3 Comparing Multiple Models with Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000),
    'Random Forest'       : RandomForestClassifier(n_estimators=100),
    'SVM'                 : SVC(kernel='rbf'),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name:25s} → {scores.mean():.3f} ± {scores.std():.3f}")

# Output example:
# Logistic Regression       → 0.891 ± 0.012
# Random Forest             → 0.923 ± 0.008
# SVM                       → 0.911 ± 0.010
# → Pick Random Forest, then tune it
```

---

### 4.4 Model Cheat-Sheet: When to Use What

| Model | Type | Best when | Watch out for |
|---|---|---|---|
| `LinearRegression` | Regression | Numeric output, linear relationship | Sensitive to outliers |
| `Ridge` / `Lasso` | Regression | Linear + regularisation needed | Lasso zeros out features |
| `LogisticRegression` | Classification | Binary/multi-class, fast baseline | Requires scaling |
| `DecisionTree` | Both | Interpretable, non-linear | Overfits easily |
| `RandomForest` | Both | Strong general performance | Slow on very large datasets |
| `GradientBoosting` | Both | High accuracy competitions | Many hyperparameters to tune |
| `SVM` | Classification | High-dimensional data | Slow on large datasets |
| `KNN` | Both | Small datasets, no training phase | Slow at prediction time |
| `KMeans` | Clustering | Grouping unlabelled data | Must specify K in advance |
| `PCA` | Dim. Reduction | Reduce features, visualise data | Loses interpretability |

---

## 5. Hyperparameter Tuning

**Hyperparameters** are settings you choose *before* training (e.g. `n_estimators`, `C`, `max_depth`). Tuning finds the combination that maximises performance.

---

### 5.1 `GridSearchCV` — Exhaustive search

> **Why?** Tries **every combination** of hyperparameters, picks the best via cross-validation.

```
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [3, 5, None]
}
Total combinations = 3 × 3 = 9 models × 5 folds = 45 fits
```

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [3, 5, None],
    'min_samples_split': [2, 5]
}

gs = GridSearchCV(
    estimator  = RandomForestClassifier(random_state=42),
    param_grid = param_grid,
    cv         = 5,              # 5-fold cross-validation
    scoring    = 'accuracy',
    n_jobs     = -1,             # Use all CPU cores
    verbose    = 1
)

gs.fit(X_train, y_train)

print(gs.best_params_)    # {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 100}
print(gs.best_score_)     # 0.927  ← CV score of best model

# Best model is already re-fitted on full training data
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
```

---

### 5.2 `RandomizedSearchCV` — Random sample search

> **Why?** When the grid is huge, random search samples N combinations instead of all — often finds near-optimal results much faster.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators'     : randint(50, 500),   # Randomly sample integers 50-500
    'max_depth'        : randint(2, 20),
    'min_samples_split': randint(2, 10),
}

rs = RandomizedSearchCV(
    estimator          = RandomForestClassifier(random_state=42),
    param_distributions= param_dist,
    n_iter             = 50,          # Try 50 random combinations
    cv                 = 5,
    scoring            = 'accuracy',
    random_state       = 42,
    n_jobs             = -1
)

rs.fit(X_train, y_train)

print(rs.best_params_)
print(rs.best_score_)
```

| | `GridSearchCV` | `RandomizedSearchCV` |
|---|---|---|
| Search space | All combinations | N random samples |
| Speed | Slow (exhaustive) | Fast |
| Best when | Small grid | Large/continuous search space |

---

### 5.3 Common Hyperparameters by Model

```python
# Logistic Regression
LogisticRegression(
    C=1.0,            # Inverse regularisation strength. Smaller = stronger regularisation
    penalty='l2',     # 'l1', 'l2', 'elasticnet', None
    max_iter=100      # Increase if it doesn't converge
)

# Random Forest
RandomForestClassifier(
    n_estimators=100,       # Number of trees — more is usually better
    max_depth=None,         # Max depth per tree (None = grow fully)
    min_samples_split=2,    # Min samples to split a node
    min_samples_leaf=1,     # Min samples in a leaf
    max_features='sqrt'     # Features to consider per split
)

# SVM
SVC(
    C=1.0,           # Regularisation — larger C = less regularisation
    kernel='rbf',    # 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale'    # Kernel coefficient ('scale', 'auto', or float)
)

# KNN
KNeighborsClassifier(
    n_neighbors=5,       # K — the most important hyperparameter
    weights='uniform',   # 'uniform' or 'distance'
    metric='minkowski'   # Distance metric
)
```

---

## 6. Pipelines

> **Why?** A `Pipeline` chains preprocessing steps and a model into a single object. It prevents data leakage and makes deployment clean.

```
Raw Data → [Imputer] → [Scaler] → [Model] → Predictions
               ↑ all steps fit only on training data ↑
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SimpleImputer, StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),     # Step 1: fill NaN
    ('scaler',  StandardScaler()),                    # Step 2: scale
    ('model',   LogisticRegression(max_iter=1000))    # Step 3: train
])

# One-liner: fit entire pipeline on training data
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Works seamlessly with GridSearchCV!
param_grid = {
    'model__C'      : [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2']
}
# Note: use '<step_name>__<param>' syntax

gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

### `ColumnTransformer` — Different preprocessing per column type

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

numeric_features  = ['age', 'salary']
categorical_features = ['city', 'job_title']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                        numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'),  categorical_features)
])

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(n_estimators=100))
])

full_pipeline.fit(X_train, y_train)
y_pred = full_pipeline.predict(X_test)
```

---

## 7. Evaluation Metrics Quick Reference

### 7.1 Classification

```python
from sklearn.metrics import (
    accuracy_score,        # (TP+TN) / Total
    precision_score,       # TP / (TP+FP)  — how accurate positive predictions are
    recall_score,          # TP / (TP+FN)  — how many actual positives were found
    f1_score,              # Harmonic mean of precision & recall
    confusion_matrix,      # Full breakdown of predictions
    classification_report  # Precision, recall, F1 for each class
)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Confusion Matrix Layout:**

```
                 Predicted
                  Pos   Neg
Actual  Pos   [ TP  |  FN ]
        Neg   [ FP  |  TN ]
```

### 7.2 Regression

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)   # Average absolute error — same units as y
mse  = mean_squared_error(y_test, y_pred)    # Penalises large errors more
rmse = np.sqrt(mse)                          # Back to same units as y
r2   = r2_score(y_test, y_pred)              # 1.0 = perfect, 0 = predicts mean, <0 = worse than mean

print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")
```

### 7.3 Most Used `scoring` Strings for CV

| Task | `scoring` |
|---|---|
| Binary classification | `'accuracy'`, `'f1'`, `'roc_auc'` |
| Multi-class | `'accuracy'`, `'f1_macro'`, `'f1_weighted'` |
| Regression | `'r2'`, `'neg_mean_squared_error'`, `'neg_mean_absolute_error'` |

> [!NOTE]
> Regression metrics are **negated** in sklearn (e.g. `neg_mean_squared_error`) because GridSearchCV always **maximises** the score. So the "most negative" MSE = the lowest MSE.

---

## 8. Saving & Loading Models

> **Why?** Training is expensive. Save your model to disk and reload it for inference without re-training.

```python
import joblib

# Save the model (or entire pipeline)
joblib.dump(best_model, 'model.pkl')

# Load it later
loaded_model = joblib.load('model.pkl')
y_pred = loaded_model.predict(X_test)
```

> [!TIP]
> Always save the **full pipeline** (not just the model) so that the scaler and imputer are also preserved.
> ```python
> joblib.dump(full_pipeline, 'pipeline.pkl')
> ```

---

## 9. Q&A — Test Yourself!

> [!IMPORTANT]
> Spend **at least 5 minutes** on each question before checking the answer. Write your answers down first.

---

### 🟢 Warm-Up (Recall)

**Q1.** What is the single most important habit when using scalers like `StandardScaler`? Why does it matter?

<details>
<summary>▶ Answer</summary>

Always `fit` the scaler **only on training data**, then `transform` both training and test data separately. If you fit on the full dataset, the scaler learns statistics (mean, std) from the test set, which means your model has *seen* the test data — this is data leakage and gives you falsely optimistic evaluation scores.

</details>

---

**Q2.** Why should you **not** use `LabelEncoder` on input features `X` for a column like `['red', 'green', 'blue']`?

<details>
<summary>▶ Answer</summary>

`LabelEncoder` assigns integers like `blue=0, green=1, red=2`. This implies an **ordinal relationship** (red > green > blue), which is mathematically false for nominal categories. The model would treat `red` as "twice as much" as `green`. Use `OneHotEncoder` instead to create separate binary columns per category.

</details>

---

**Q3.** You train a model and get 99% accuracy on the training set but only 61% on the test set. What is happening and how do you typically fix this?

<details>
<summary>▶ Answer</summary>

This is **overfitting** — the model has memorised the training data instead of learning generalisable patterns. Fixes:
- Reduce model complexity (lower `max_depth`, fewer `n_estimators`)
- Add regularisation (increase `C⁻¹` or `alpha`, use Ridge/Lasso)
- Get more training data
- Use cross-validation rather than a single train/test split for more reliable estimates
- Use `PolynomialFeatures` with a lower degree

</details>

---

### 🟡 Intermediate (Understanding)

**Q4.** A colleague runs this code:

```python
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X)   # Scales ALL data

X_train, X_test = train_test_split(X_all_scaled, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

What is the **critical mistake** and how do you fix it?

<details>
<summary>▶ Answer</summary>

**Mistake:** The scaler was fitted on all data (including the test set) *before* splitting. This means the test-set statistics (mean, std) influenced the scaling of training data — data leakage.

**Fix:** Split first, scale second:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)         # transform ONLY
```
Or use a `Pipeline` — it handles this automatically.

</details>

---

**Q5.** Explain the difference between `GridSearchCV` and `RandomizedSearchCV`. When would you choose one over the other?

<details>
<summary>▶ Answer</summary>

| | GridSearchCV | RandomizedSearchCV |
|---|---|---|
| How | Tries every combination in the grid | Randomly samples N combinations |
| Speed | Slower (exhaustive) | Faster |

**Choose GridSearchCV** when: you have a small, well-defined hyperparameter grid (e.g. 2-3 parameters with 3-4 values each).

**Choose RandomizedSearchCV** when: the search space is large or includes continuous ranges. It often finds near-optimal results with far fewer fits (set `n_iter` to control the budget).

</details>

---

**Q6.** You have a dataset with both numeric columns (`age`, `income`) and categorical columns (`city`, `job_type`). Which sklearn tool lets you apply different preprocessing to different column types inside a Pipeline?

<details>
<summary>▶ Answer</summary>

`ColumnTransformer`. It lets you specify a list of `(name, transformer, columns)` tuples, applying each transformer to the specified subset of columns. The results are concatenated horizontally. Combined with `Pipeline`, you get a clean, leak-proof end-to-end workflow.

</details>

---

### 🔴 Advanced (Application)

**Q7.** Write the complete code to:
1. Load the `breast_cancer` dataset from sklearn
2. Split it 80/20 (stratified)
3. Build a pipeline: `SimpleImputer(mean)` → `StandardScaler` → `LogisticRegression`
4. Use `GridSearchCV` to tune `C ∈ [0.01, 0.1, 1, 10]` and `penalty ∈ ['l1', 'l2']` (use solver `liblinear`) with 5-fold CV
5. Print the best params and test accuracy

<details>
<summary>▶ Answer</summary>

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SimpleImputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(solver='liblinear', max_iter=1000))
])

# 4. GridSearchCV
param_grid = {
    'model__C'      : [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2']
}
gs = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

# 5. Results
print("Best params:", gs.best_params_)
print("CV accuracy:", round(gs.best_score_, 4))
print("Test accuracy:", round(accuracy_score(y_test, gs.predict(X_test)), 4))
```

Expected output (approximate):
```
Best params: {'model__C': 0.1, 'model__penalty': 'l2'}
CV accuracy: 0.9802
Test accuracy: 0.9737
```

</details>

---

**Q8.** You are using `cross_val_score` and get these 5-fold scores:

```
[0.91, 0.90, 0.91, 0.55, 0.90]
```

What could cause the dramatic dip in fold 4, and what should you investigate?

<details>
<summary>▶ Answer</summary>

A single fold performing dramatically worse than the others is a red flag. Possible causes:
- **Class imbalance**: Fold 4 might have very few positive class examples in training, or the test fold might be unrepresentative. Fix: use `StratifiedKFold` or set `stratify=y` in `cross_val_score`.
- **Data ordering**: If the dataset is sorted by class or time, a non-shuffled split could produce a homogeneous fold. Fix: shuffle data before splitting.
- **Outliers or data quality issues**: A cluster of bad/corrupted rows landed in one fold.

The mean (0.834) would be misleading here. Investigate the fold composition before trusting any averaged score.

</details>

---

**Q9.** What does `scoring='neg_mean_squared_error'` mean in the context of `GridSearchCV`? Why is it negative?

<details>
<summary>▶ Answer</summary>

`GridSearchCV` always **maximises** its scoring function. Since MSE is a loss (lower is better), sklearn negates it so that "less negative" = lower MSE = better. Therefore:

- `neg_mean_squared_error = -MSE`
- The "best" model has the **highest** (least negative) `neg_mean_squared_error`
- To recover the actual MSE: `actual_mse = -gs.best_score_`

This is purely a convention to keep the API consistent (always maximise).

</details>

---

> [!TIP]
> **Full Workflow Summary** — the steps every ML project follows:
>
> ```
> 1. Load & explore data
> 2. Split → train_test_split (stratify if classification)
> 3. Preprocess → Imputer, Scaler, Encoder (fit on train ONLY)
> 4. Baseline model → fit, cross_val_score
> 5. Compare models → loop cross_val_score over candidates
> 6. Tune best model → GridSearchCV or RandomizedSearchCV
> 7. Final evaluation → predict on held-out test set
> 8. Save → joblib.dump(pipeline)
> ```
