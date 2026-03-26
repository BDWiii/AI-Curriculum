# Decision Boundaries & Regularization

## 📚 Overview

In this lesson we cover two essential ideas that shape how and why we choose our ML models:

1. **Decision Boundaries** — the line (or curve) a classifier draws to separate classes. Understanding it tells you *which algorithm* to reach for.
2. **Regularization** — a technique to stop a model from memorising the training data so it generalizes to new examples.

---

# Part 1 — Decision Boundaries

## 🎯 What Is a Decision Boundary?

When a classifier predicts a class for a new point it is essentially asking: *"Which side of the fence does this point fall on?"*

The **decision boundary** is that fence — the rule the model has learned to separate classes.

**Simple intuition:**

| Region | Model predicts |
|--------|---------------|
| Left of boundary | Class 0 (e.g. "Not spam") |
| Right of boundary | Class 1 (e.g. "Spam") |

The shape of the boundary is determined by the algorithm and the data. Some boundaries are straight lines; others are curves or complex shapes.

---

## 📐 Linear Decision Boundaries

A **linear boundary** is a straight line (2D), a plane (3D), or a hyperplane (higher dimensions).

**When does it appear?**  
Any model whose hypothesis is a linear combination of features draws a linear boundary.

### Logistic Regression

$$h_{w,b}(x) = \sigma(w^Tx + b) = 0.5 \;\text{ when }\; w^Tx + b = 0$$

The boundary is exactly the set of points where $w^Tx + b = 0$ — a straight line.

```
     x₂
      |      ✦ ✦
      |   ✦ ✦        (Class 1)
      |  /
   ---/--------->  x₁
      / ✗ ✗
     /       ✗       (Class 0)
```

> **Works well when** the two classes can be cleanly separated (or approximately so) by a straight line.

**Example use-cases:** Email spam vs. not-spam (based on word counts), pass/fail prediction from study hours.

---

## 🌀 Non-Linear Decision Boundaries

Real data is rarely linearly separable. When classes interleave or form concentric rings, a straight line will always misclassify many points.

**Two main approaches to get non-linear boundaries:**

### Approach 1 — Polynomial Features (still Logistic Regression)

Add engineered features like $x_1^2$, $x_2^2$, $x_1 x_2$. The model is still logistic regression but the boundary in the original space becomes a curve.

$$h_{w,b}(x) = \sigma(w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2 + b)$$

Boundary equation: $w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_2^2 + b = 0$ — this is an **ellipse or circle** in the original feature space.

```
     x₂
      |    ✗ ✗ ✗
      |  ✗       ✗
      |  ✗   ✦   ✗   (✦ = class 1, inner ring)
      |  ✗       ✗
      |    ✗ ✗ ✗
      +-----------> x₁
```

### Approach 2 — Using a Different Algorithm Altogether

If polynomial features become unwieldy (too many terms) or you need a fundamentally different boundary shape, switch algorithms.

---

## 🗺️ Matching Algorithms to Boundary Types

This is the key skill: **look at your data, then choose the algorithm whose boundary shape fits**.

```
Dataset Shape              →  Algorithm to Try
------------------------------------------------------
Linearly separable         →  Logistic Regression (linear)
Circular / elliptical      →  Logistic Regression + polynomial features
                              OR Kernel SVM (RBF kernel)
Complex / irregular        →  Decision Tree / Random Forest
Any shape (local regions)  →  K-Nearest Neighbors (KNN)
```

### K-Nearest Neighbors (KNN)

KNN draws **no explicit boundary formula**. For a new point it simply looks at the $k$ closest training points and takes a majority vote.

- Small $k$ → very jagged, complex boundary (fits training data closely)
- Large $k$ → smoother, more rounded boundary (more robust)

```
     x₂
      |  ✦  ✦   ✗         k=1: boundary hugs every point
      |    ✦  ✗  ✗         k=5: boundary smooths out
      |  ✦   ?   ✗
      |    ✦    ✗
      +-----------> x₁
      ? = classify by majority of k nearest neighbors
```

---

## 🧭 Quick Algorithm Selection Guide

| Boundary needed | Algorithm(s) |
|----------------|--------------|
| Straight line / hyperplane | Logistic Regression |
| Smooth curve (circle, ellipse) | Logistic Reg. + polynomial features |
| Flexible curved boundary | SVM with RBF Kernel |
| Arbitrary / region-based | KNN |
| Hierarchical splits | Decision Tree |
| Ensemble of splits | Random Forest |

> [!NOTE]
> There is no single "best" algorithm. You pick based on the data shape, the dataset size, interpretability requirements, and computational budget. As you learn more algorithms in this course, add them to your mental toolkit.

---

## 🧪 Think-Through Questions — Decision Boundaries

**Q1.** You have a 2-feature dataset where class 0 forms a ring around class 1 (like a bullseye). Logistic regression with only the original two features gives you 58 % accuracy. What would you try next, and why?

<details>
<summary>💡 Hint</summary>

Think about what shape the boundary needs to be to separate a ring from its center. Can a straight line ever do it? What features would you add?
</details>

<details>
<summary>✅ Answer</summary>

A straight line can **never** separate a bullseye pattern. You need a circular/elliptical boundary.

Add polynomial features $x_1^2$ and $x_2^2$. The model can then learn a boundary like:

$$w_1 x_1^2 + w_2 x_2^2 + b = 0$$

which is a circle or ellipse in the original space. Alternatively, try KNN (small $k$) or an SVM with an RBF kernel — both handle this shape naturally.
</details>

---

**Q2.** A friend says: *"I'll always use KNN with k=1 because it perfectly separates my training data."* What is wrong with this reasoning?

<details>
<summary>✅ Answer</summary>

k=1 KNN memorises every training point. The boundary is so jagged that it fits noise and outliers. On unseen data the model will likely perform poorly — this is **overfitting**.

A larger $k$ smooths the boundary and generalises better, even if training accuracy drops slightly. The goal is good performance on *new* data, not perfect performance on training data.
</details>

---

**Q3.** Code challenge — plot two datasets and the logistic regression boundary for each:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# --- Dataset A: linearly separable ---
X_A = np.vstack([
    np.random.randn(50, 2) + [2, 2],   # class 1
    np.random.randn(50, 2) + [-2, -2]  # class 0
])
y_A = np.array([1]*50 + [0]*50)

# --- Dataset B: circular (not linearly separable) ---
angles = np.linspace(0, 2*np.pi, 50)
X_B = np.vstack([
    np.c_[np.cos(angles) * 3, np.sin(angles) * 3],  # class 0 (outer ring)
    np.random.randn(50, 2) * 0.5                      # class 1 (center)
])
y_B = np.array([0]*50 + [1]*50)

def plot_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dataset A — plain logistic regression
model_A = LogisticRegression().fit(X_A, y_A)
plot_boundary(axes[0], model_A, X_A, y_A,
              f"Dataset A (linear)\nAccuracy: {model_A.score(X_A, y_A):.0%}")

# Dataset B — try plain logistic regression first
# TODO: after observing the poor accuracy, add polynomial features and retrain.
# What accuracy do you get? What does the new boundary look like?
model_B = LogisticRegression().fit(X_B, y_B)
plot_boundary(axes[1], model_B, X_B, y_B,
              f"Dataset B (circular)\nAccuracy: {model_B.score(X_B, y_B):.0%}")

plt.tight_layout()
plt.show()
```

**Your task:**
1. Run the code and observe both accuracy values.
2. Fix Dataset B by adding polynomial features (`PolynomialFeatures(degree=2)`) before fitting logistic regression.
3. Plot the new boundary.  What changed?

<details>
<summary>✅ Solution for step 2 & 3</summary>

```python
# Polynomial logistic regression for Dataset B
poly = PolynomialFeatures(degree=2, include_bias=False)
X_B_poly = poly.fit_transform(X_B)

model_B_poly = LogisticRegression(max_iter=1000).fit(X_B_poly, y_B)

# To plot: we need to transform the mesh grid the same way
fig, ax = plt.subplots(figsize=(6, 5))
x_min, x_max = X_B[:, 0].min() - 1, X_B[:, 0].max() + 1
y_min, y_max = X_B[:, 1].min() - 1, X_B[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_poly = poly.transform(np.c_[xx.ravel(), yy.ravel()])
Z = model_B_poly.predict(grid_poly).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
ax.scatter(X_B[:, 0], X_B[:, 1], c=y_B, cmap='bwr', edgecolors='k')
ax.set_title(f"Dataset B + Polynomial Features\nAccuracy: {model_B_poly.score(X_B_poly, y_B):.0%}")
plt.show()
```

The boundary becomes **circular**, matching the data structure, and accuracy jumps significantly.
</details>

---

# Part 2 — Regularization

## 🤔 The Problem: Overfitting

Before explaining regularization, we need to understand the problem it solves.

Consider fitting a polynomial regression to 6 data points:

- **Degree 1** (straight line): misses the trend → **underfitting**
- **Degree 5** (wiggly curve): passes through every point exactly → **overfitting**
- **Degree 2–3**: fits the trend without memorising noise → **just right**

```
  Overfit model (degree 5):        Underfit model (degree 1):
  y                                y
  |  *                             |         *
  |   \  *  /\                     |  *  *  /
  | *  \/  /  *                    | ______/  *
  +-----------> x                  +-----------> x
  Wiggles through every point      Completely misses the pattern
```

**Overfitting signs:**
- Very low training error, high validation/test error
- Large weight values ($w_j$ with huge magnitudes)
- Model memorises noise rather than learning the signal

---

## 💡 What Is Regularization?

**Regularization** adds a penalty to the cost function that discourages large weight values. Large weights are the signature of an overfit model — they let the model create wild oscillations to hit every training point.

By penalising large weights we are telling the model:  
*"Fit the data well, but don't go crazy doing it."*

**General regularized cost function:**

$$J_{\text{reg}}(w,b) = J(w,b) + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{regularization penalty}}$$

Where:
- $J(w,b)$ is the original cost (MSE or log-loss)
- $\lambda$ (lambda) is the **regularization strength** — a hyperparameter you tune
- $n$ is the number of features
- The bias $b$ is **not** regularized (standard practice)

> [!IMPORTANT]
> **The role of λ (lambda):**
> - $\lambda = 0$: No regularization — original model, prone to overfitting
> - Small $\lambda$: Light regularization — weights may still grow large
> - Large $\lambda$: Heavy regularization — weights are forced toward zero (may underfit)
> - **You tune $\lambda$ with a validation set**

---

## 📐 L2 Regularization (Ridge)

**L2** penalises the **squared** magnitude of each weight:

$$\text{Penalty}_{L2} = \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

### Polynomial Regression with L2

Original MSE cost:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

Regularized version:

$$J_{\text{ridge}}(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

**Gradient update** (how gradient descent changes with L2):

$$w_j := w_j - \alpha \left[ \frac{1}{m}\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

$$w_j := w_j \left(1 - \frac{\alpha \lambda}{m}\right) - \frac{\alpha}{m}\sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

Notice the factor $\left(1 - \frac{\alpha\lambda}{m}\right)$: it **shrinks** $w_j$ slightly every step toward zero, before the normal gradient update is applied. This is called **weight decay**.

### Logistic Regression with L2

Original log-loss:

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

Regularized version:

$$J_{\text{ridge}}(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

The gradient update is identical in form to polynomial regression above:

$$w_j := w_j\left(1 - \frac{\alpha\lambda}{m}\right) - \frac{\alpha}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x_j^{(i)}$$

**Key property of L2:** Weights shrink toward zero but **rarely become exactly zero**. All features stay in the model with small coefficients.

---

## 📐 L1 Regularization (Lasso)

**L1** penalises the **absolute** magnitude of each weight:

$$\text{Penalty}_{L1} = \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

### Cost function with L1

$$J_{\text{lasso}}(w,b) = J(w,b) + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

(works the same way with both MSE and log-loss)

**Gradient update** (the subgradient of $|w_j|$ is $\text{sign}(w_j)$):

$$w_j := w_j - \alpha\left[\frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\text{sign}(w_j)\right]$$

Where $\text{sign}(w_j) = +1$ if $w_j > 0$, $-1$ if $w_j < 0$, $0$ if $w_j = 0$.

**Key property of L1:** Some weights are pushed **exactly to zero**, effectively removing those features from the model. L1 performs **automatic feature selection**.

---

## 🔍 L1 vs L2 at a Glance

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|-----------|-----------|
| Penalty term | $\frac{\lambda}{m}\sum\|w_j\|$ | $\frac{\lambda}{2m}\sum w_j^2$ |
| Effect on weights | Drives some to **exactly 0** | Shrinks all toward 0 (rarely exactly 0) |
| Feature selection | ✅ Yes (sparse model) | ❌ No (keeps all features) |
| When to prefer | Many irrelevant features | Most features are relevant |
| Sensitivity to outliers | More robust | More sensitive |

---

## 🔧 Code Example: Effect of Regularization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(0)

# Generate noisy data
X = np.sort(np.random.rand(20, 1) * 6 - 3, axis=0)
y = np.sin(X).ravel() + np.random.randn(20) * 0.3

X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["No Regularization\n(degree-10 poly)", "L2 Ridge\n(λ=1)", "L1 Lasso\n(λ=0.01)"]
models = [
    Pipeline([("poly", PolynomialFeatures(degree=10)),
              ("scale", StandardScaler()),
              ("model", LinearRegression())]),
    Pipeline([("poly", PolynomialFeatures(degree=10)),
              ("scale", StandardScaler()),
              ("model", Ridge(alpha=1.0))]),          # alpha = λ in sklearn
    Pipeline([("poly", PolynomialFeatures(degree=10)),
              ("scale", StandardScaler()),
              ("model", Lasso(alpha=0.01, max_iter=5000))]),
]

for ax, model, title in zip(axes, models, titles):
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    ax.scatter(X, y, color='black', s=30, label="Training data")
    ax.plot(X_plot, y_plot, color='blue', linewidth=2, label="Model fit")
    ax.plot(X_plot, np.sin(X_plot), color='red', linestyle='--',
            linewidth=1, label="True function")
    ax.set_ylim(-2, 2)
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

**Observe:**
- The unregularized model oscillates wildly to fit every training point
- Ridge smooths the curve significantly with all weights kept small
- Lasso produces an even sparser model (inspect `model[-1].coef_` to see zeros)

---

## 🧪 Think-Through Questions — Regularization

**Q1.** You have a polynomial regression model (degree 8) with very low training error but high test error. You add L2 regularization and increase $\lambda$ from 0 to a large value. Describe what happens to training error and test error as $\lambda$ increases.

<details>
<summary>💡 Hint</summary>

Think of two extremes: $\lambda = 0$ (original overfit model) and $\lambda \to \infty$ (what does an infinitely penalised weight look like?).
</details>

<details>
<summary>✅ Answer</summary>

| $\lambda$ | Training error | Test error |
|-----------|---------------|------------|
| 0 (no reg) | Very low (overfit) | High |
| Small | Slightly higher | Lower (better generalisation) |
| Optimal | Moderate | Minimum (best generalisation) |
| Too large | High (underfit) | Also high |

As $\lambda \to \infty$, all weights are forced to zero → the model predicts a constant (the mean of $y$) → both training and test error are high. This is **underfitting**.

The sweet spot is found by plotting validation error vs $\lambda$ — the U-shaped curve's minimum gives the best $\lambda$.
</details>

---

**Q2.** Your dataset has 100 features but you suspect only 10 of them are truly useful. Would you prefer L1 or L2 regularization? Why?

<details>
<summary>✅ Answer</summary>

**L1 (Lasso)** — because it drives irrelevant feature weights exactly to zero, automatically selecting the important 10 features and producing a simpler, more interpretable model.

L2 would keep all 100 features with small but non-zero weights, making the model harder to interpret without removing the noise from the 90 irrelevant features.
</details>

---

**Q3.** Code challenge — implement L2 regularization in gradient descent from scratch for polynomial regression:

```python
import numpy as np

def polynomial_features(X, degree):
    """Build feature matrix [1, x, x^2, ..., x^degree] for 1D X."""
    return np.column_stack([X**d for d in range(1, degree + 1)])  

def predict(X_poly, w, b):
    return X_poly @ w + b

def compute_cost_l2(X_poly, y, w, b, lam):
    m = len(y)
    errors = predict(X_poly, w, b) - y
    mse    = (1 / (2 * m)) * np.sum(errors**2)
    # TODO: add the L2 penalty term here
    penalty = ???
    return mse + penalty

def gradient_descent_l2(X_poly, y, w, b, alpha, lam, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        errors = predict(X_poly, w, b) - y
        dw = (1/m) * X_poly.T @ errors + ???   # TODO: add regularization gradient
        db = (1/m) * np.sum(errors)             # bias is NOT regularized
        w  = w - alpha * dw
        b  = b - alpha * db
        cost_history.append(compute_cost_l2(X_poly, y, w, b, lam))
    return w, b, cost_history

# --- Test ---
np.random.seed(1)
X = np.linspace(-2, 2, 30)
y = X**2 + np.random.randn(30) * 0.5   # true relation: quadratic

degree = 5
X_poly = polynomial_features(X, degree)

w = np.zeros(degree)
b = 0.0
lam   = 0.1     # try 0, 0.1, 1, 10
alpha = 0.01
iters = 2000

w_trained, b_trained, costs = gradient_descent_l2(X_poly, y, w, b, alpha, lam, iters)
print("Trained weights:", np.round(w_trained, 3))
```

<details>
<summary>✅ Solution</summary>

```python
def compute_cost_l2(X_poly, y, w, b, lam):
    m = len(y)
    errors  = predict(X_poly, w, b) - y
    mse     = (1 / (2 * m)) * np.sum(errors**2)
    penalty = (lam / (2 * m)) * np.sum(w**2)   # ← L2 penalty
    return mse + penalty

def gradient_descent_l2(X_poly, y, w, b, alpha, lam, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        errors = predict(X_poly, w, b) - y
        dw = (1/m) * X_poly.T @ errors + (lam/m) * w   # ← regularization term
        db = (1/m) * np.sum(errors)
        w  = w - alpha * dw
        b  = b - alpha * db
        cost_history.append(compute_cost_l2(X_poly, y, w, b, lam))
    return w, b, cost_history
```

**Experiment:** run with `lam = 0` (overfit), `lam = 0.1` (good fit), `lam = 10` (underfit) and observe how `w_trained` values change.
</details>

---

**Q4.** Without running any code, what are the trained weights of a Ridge model as $\lambda \to \infty$? What does the model's predictions look like?

<details>
<summary>✅ Answer</summary>

As $\lambda \to \infty$, the penalty term dominates the cost function completely. The only way to minimize cost is to make all $w_j \to 0$.

With $w = 0$ and only $b$ remaining, the model predicts $\hat{y} = b$ for all inputs — a constant equal to the mean of the training $y$ values (since that minimises MSE when $w=0$).

The model is completely underfitted: it ignores all features and always predicts the same value.
</details>

---

## 🎓 Key Takeaways

> [!IMPORTANT]
> **Decision Boundaries**
> - The decision boundary separates the classes; its shape determines which algorithm fits the data
> - Linear boundaries → Logistic Regression / Linear SVM
> - Non-linear boundaries → polynomial features, KNN, kernel SVM, Decision Trees
> - Always visualise your data before picking an algorithm

> [!IMPORTANT]
> **Regularization**
> - Regularization penalises large weights to prevent overfitting
> - **L2 (Ridge)**: penalty $= \frac{\lambda}{2m}\sum w_j^2$ → shrinks all weights, keeps all features
> - **L1 (Lasso)**: penalty $= \frac{\lambda}{m}\sum |w_j|$ → drives some weights to zero, performs feature selection
> - $\lambda$ is a hyperparameter: too small → overfit, too large → underfit
> - The bias $b$ is **never** regularized
> - Both L1 and L2 apply to the *same* gradient descent loop — you only change the penalty term in the cost and gradient

---

*Next lesson → Bias-Variance Trade-off: a formal framework for understanding overfitting and underfitting.*
