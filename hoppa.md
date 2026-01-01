# Lesson 1: Linear Regression & Logistic Regression

> **Goal of this lesson**
> By the end of this lesson, the student should:

* Understand what supervised learning is and how it differs from unsupervised learning
* Understand linear regression and logistic regression intuitively **and mathematically**
* Be able to derive and interpret cost functions
* Understand gradient descent, learning rate, and how all components work together
* Implement both models from scratch in Python
* Reason about conceptual and mathematical questions (tests)

---

## 1. Brief Introduction to Machine Learning

**Machine Learning (ML)** is a subfield of AI that focuses on building systems that learn patterns from data instead of being explicitly programmed.

At its core, ML is about:

```
Data  →  Model  →  Predictions  →  Error  →  Improvement
```

The model improves by minimizing error according to a mathematical objective.

---

## 2. Supervised vs Unsupervised Learning

### 2.1 Supervised Learning

**Definition:**
Supervised learning uses **labeled data**, meaning each input has a known correct output.

**General form:**

```
(x, y) pairs
```

Where:

* `x` = input features
* `y` = target (label)

#### Examples

| Problem                | Input (x)      | Output (y)      |
| ---------------------- | -------------- | --------------- |
| House price prediction | size, location | price           |
| Spam detection         | email text     | spam / not spam |
| Medical diagnosis      | symptoms       | disease         |

**Common supervised algorithms:**

* Linear Regression
* Logistic Regression
* k-NN
* Decision Trees
* Neural Networks

---

### 2.2 Unsupervised Learning

**Definition:**
Unsupervised learning uses **unlabeled data**. The model tries to discover hidden structure.

```
(x) only
```

#### Examples

| Problem               | Goal                         |
| --------------------- | ---------------------------- |
| Customer segmentation | Group similar customers      |
| Topic modeling        | Discover themes in documents |
| Anomaly detection     | Find unusual behavior        |

**Common unsupervised algorithms:**

* K-Means Clustering
* PCA
* Autoencoders

---

### Key Difference (Intuition)

| Supervised                     | Unsupervised            |
| ------------------------------ | ----------------------- |
| Has correct answers            | No correct answers      |
| Learns mapping x → y           | Learns structure in x   |
| Error can be measured directly | No explicit error label |

---

## 3. Linear Regression

### 3.1 Problem Definition

Linear regression predicts a **continuous value**.

Example:

> Predict house price based on size

---

### 3.2 Model Representation

For **one feature**:

[ \hat{y} = wx + b ]

For **multiple features**:

[ \hat{y} = w^T x + b ]

Where:

* `w` = weights
* `b` = bias
* `x` = input vector
* `ŷ` = prediction

---

### 3.3 Geometric Interpretation

* In 1D → a **line**
* In 2D → a **plane**
* In nD → a **hyperplane**

The goal is to find the best-fitting line.

---

### 3.4 Cost Function (Mean Squared Error)

[ J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 ]

**Why squared error?**

* Penalizes large errors
* Smooth and differentiable
* Convex → one global minimum

**Intuition:**

> Measure how far predictions are from actual values

---

### 3.5 Gradient Descent

We want to minimize `J(w, b)`.

**Update rules:**

[ w := w - \alpha \frac{\partial J}{\partial w} ]
[ b := b - \alpha \frac{\partial J}{\partial b} ]

Where:

* `α` = learning rate

**Partial derivatives:**

[ \frac{\partial J}{\partial w} = \frac{1}{m} \sum (\hat{y} - y)x ]
[ \frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{y} - y) ]

---

### 3.6 Learning Rate

| Learning Rate | Behavior              |
| ------------- | --------------------- |
| Too small     | Very slow convergence |
| Too large     | Divergence            |
| Just right    | Fast convergence      |

---

### 3.7 Linear Regression From Scratch (Code)

```python
import numpy as np

X = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

w, b = 0.0, 0.0
alpha = 0.01
m = len(X)

for _ in range(1000):
    y_hat = w * X + b
    dw = (1/m) * np.sum((y_hat - y) * X)
    db = (1/m) * np.sum(y_hat - y)
    w -= alpha * dw
    b -= alpha * db

print(w, b)
```

---

## 4. Logistic Regression

### 4.1 Problem Definition

Logistic regression predicts **probabilities for binary classification**.

Example:

> Spam (1) or Not Spam (0)

---

### 4.2 Why Not Linear Regression?

Linear regression outputs values from `(-∞, +∞)`

But probabilities must be in:

[ [0, 1] ]

---

### 4.3 Sigmoid Function

[ \sigma(z) = \frac{1}{1 + e^{-z}} ]

Where:

[ z = w^T x + b ]

**Interpretation:**

* Output = probability of class 1

---

### 4.4 Logistic Model

[ \hat{y} = \sigma(w^T x + b) ]

Decision boundary:

[ \hat{y} = 0.5 ]

---

### 4.5 Cost Function (Binary Cross-Entropy)

[ J(w,b) = -\frac{1}{m} \sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] ]

**Why this cost?**

* Penalizes confident wrong predictions heavily
* Derived from maximum likelihood

---

### 4.6 Gradient Descent for Logistic Regression

Update rules are **same structure** as linear regression:

[ w := w - \alpha \frac{\partial J}{\partial w} ]

Where:

[ \frac{\partial J}{\partial w} = \frac{1}{m} \sum (\hat{y} - y)x ]

---

### 4.7 Logistic Regression From Scratch (Code)

```python
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

w, b = 0.0, 0.0
alpha = 0.1
m = len(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for _ in range(1000):
    z = X.dot(w) + b
    y_hat = sigmoid(z)
    dw = (1/m) * np.sum((y_hat - y) * X.flatten())
    db = (1/m) * np.sum(y_hat - y)
    w -= alpha * dw
    b -= alpha * db

print(w, b)
```

---

## 5. How Everything Connects (Big Picture)

```
Model → Prediction → Cost → Gradient → Update → Better Model
```

* **Model** defines hypothesis
* **Cost function** measures error
* **Gradient descent** minimizes cost
* **Learning rate** controls step size

Same pipeline, different functions.

---

## 6. Conceptual & Mathematical Tests

### Test 1 (Conceptual)

Why does Mean Squared Error create a convex optimization problem for linear regression?

---

### Test 2 (Math)

Derive the gradient of MSE with respect to `w`.

---

### Test 3 (Intuition)

What happens if learning rate → infinity?

---

### Test 4 (Logistic Regression)

Why does MSE perform poorly for classification?

---

### Test 5 (Implementation)

Modify the logistic regression code to support **multiple features**.

---

## End of Lesson 1

**Next Lesson Preview:**

* Feature scaling
* Normalization vs standardization
* Polynomial regression
* Overfitting & underfitting
