# 📝 Assignment 2: Implementing Linear & Polynomial Regression from Scratch

> **Goal**: Build linear regression and polynomial regression from scratch using only NumPy.  
> You will implement each core component as a function, then wire them all together in a complete example.

---

## 📐 Quick Math Recap

Before coding, make sure these equations are clear in your mind:

| Component | Scalar Form | Vectorized Form |
|-----------|-------------|-----------------|
| **Prediction** | $\hat{y} = wx + b$ | $\hat{y} = Xw + b$ |
| **Cost (MSE)** | $J = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$ | $J = \frac{1}{2m}\lVert Xw + b - y \rVert^2$ |
| **Gradient w** | $\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x^{(i)}$ | $\frac{\partial J}{\partial w} = \frac{1}{m} X^T(Xw + b - y)$ |
| **Gradient b** | $\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$ | $\frac{\partial J}{\partial b} = \frac{1}{m}\sum(Xw + b - y)$ |
| **Update w** | $w \leftarrow w - \alpha \frac{\partial J}{\partial w}$ | same |
| **Update b** | $b \leftarrow b - \alpha \frac{\partial J}{\partial b}$ | same |

---

# Part 1 — Linear Regression (Scalar / Loop Form)

> These functions use plain Python loops — no matrix math yet.  
> This mirrors the math exactly and is great for building intuition.

---

## 🔧 Problem 1.1 — Prediction Function

The prediction function computes $\hat{y} = wx + b$ for every sample.

```python
import numpy as np

def predict(X, w, b):
    """
    Compute predictions: y_hat = w * x + b  (for each sample)

    Parameters
    ----------
    X : list or 1-D array of shape (m,)  — input feature values
    w : float                            — weight (slope)
    b : float                            — bias  (intercept)

    Returns
    -------
    predictions : list of floats, shape (m,)
    """
    predictions = []
    for x in X:
        # TODO: compute w * x + b and append to predictions
        # -----------------------------------------------
        pass
        # -----------------------------------------------
    return predictions
```

<details>
<summary>💡 Hint</summary>

Multiply `w` by `x`, then add `b`.  
In math: $\hat{y} = wx + b$

</details>

<details>
<summary>✅ Solution</summary>

```python
def predict(X, w, b):
    predictions = []
    for x in X:
        y_hat = w * x + b      # core formula: wx + b
        predictions.append(y_hat)
    return predictions
```

</details>

---

## 🔧 Problem 1.2 — Cost Function (MSE)

MSE measures the average squared difference between predictions and true values.

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

```python
def compute_cost(X, y, w, b):
    """
    Compute Mean Squared Error cost.

    Parameters
    ----------
    X : list or 1-D array, shape (m,)
    y : list or 1-D array, shape (m,)  — true target values
    w : float
    b : float

    Returns
    -------
    cost : float
    """
    m = len(y)
    total_error = 0.0

    for i in range(m):
        y_hat = predict([X[i]], w, b)[0]   # prediction for sample i
        # TODO: compute squared error (y_hat - y[i])^2 and add to total_error
        # -----------------------------------------------
        pass
        # -----------------------------------------------

    # TODO: return the average: total_error / (2 * m)
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

- Error for one sample = $(\hat{y} - y)$
- Squared error = $(\hat{y} - y)^2$
- Sum all squared errors, then divide by $2m$

</details>

<details>
<summary>✅ Solution</summary>

```python
def compute_cost(X, y, w, b):
    m = len(y)
    total_error = 0.0

    for i in range(m):
        y_hat = w * X[i] + b
        total_error += (y_hat - y[i]) ** 2   # squared error

    cost = total_error / (2 * m)             # average
    return cost
```

</details>

---

## 🔧 Problem 1.3 — Gradient Descent

Gradient descent tweaks $w$ and $b$ step-by-step to reduce the cost.

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)}) \cdot x^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$$

$$w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}, \quad b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}$$

```python
def gradient_descent(X, y, w, b, alpha, iterations):
    """
    Run gradient descent to learn w and b.

    Parameters
    ----------
    X          : list or 1-D array, shape (m,)
    y          : list or 1-D array, shape (m,)
    w          : float  — initial weight
    b          : float  — initial bias
    alpha      : float  — learning rate
    iterations : int    — number of update steps

    Returns
    -------
    w, b        : learned parameters (float, float)
    cost_history: list of cost at each iteration
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):

        # --- compute gradients ---
        dw = 0.0
        db = 0.0
        for i in range(m):
            y_hat = w * X[i] + b
            error = y_hat - y[i]

            # TODO: accumulate gradient for w  →  dw += error * X[i]
            # -----------------------------------------------
            pass
            # -----------------------------------------------

            # TODO: accumulate gradient for b  →  db += error
            # -----------------------------------------------
            pass
            # -----------------------------------------------

        dw /= m   # average over all samples
        db /= m

        # TODO: update w using the gradient and learning rate
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        # TODO: update b using the gradient and learning rate
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history
```

<details>
<summary>💡 Hint</summary>

- `dw += error * X[i]`  ← this is $\sum (\hat{y} - y) \cdot x$
- `db += error`         ← this is $\sum (\hat{y} - y)$
- After dividing by `m`, subtract: `w = w - alpha * dw`

</details>

<details>
<summary>✅ Solution</summary>

```python
def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        dw = 0.0
        db = 0.0
        for i in range(m):
            y_hat = w * X[i] + b
            error = y_hat - y[i]
            dw += error * X[i]     # gradient w
            db += error            # gradient b

        dw /= m
        db /= m

        w = w - alpha * dw         # update w
        b = b - alpha * db         # update b

        cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history
```

</details>

---

## 🧪 Example 1 — Putting It All Together (Scalar)

> Now let's use all three functions you just built on a real example.

**Scenario**: We have house sizes (in hundreds of sq ft) and their prices (in $1000s).  
We want to learn the line that best fits this data.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Dataset ---
X = [1.0, 2.0, 3.0, 4.0, 5.0]   # house size (hundreds of sq ft)
y = [1.5, 3.5, 5.0, 6.5, 8.0]   # price ($1000s)

# --- Initialize parameters ---
w = 0.0
b = 0.0
alpha = 0.01
iterations = 1000

# --- Train ---
w, b, cost_history = gradient_descent(X, y, w, b, alpha, iterations)
print(f"Learned  w = {w:.4f},  b = {b:.4f}")

# --- Final cost ---
final_cost = compute_cost(X, y, w, b)
print(f"Final cost (MSE) = {final_cost:.6f}")

# --- Predict ---
new_size = 3.5
predicted_price = predict([new_size], w, b)[0]
print(f"Predicted price for size {new_size}: ${predicted_price:.2f}k")

# --- Plot: Data vs Fitted Line ---
x_line = [min(X), max(X)]
y_line = predict(x_line, w, b)
plt.plot(X, y, 'bo', label='Data')
plt.plot(x_line, y_line, 'r-', label='Fitted line')
plt.xlabel('Size (100 sq ft)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression — Scalar Form')
plt.legend()
plt.show()

# --- Plot: Cost over iterations ---
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(w,b)')
plt.title('Cost Decreasing over Time')
plt.show()
```

**Expected output (approximately)**:
```
Learned  w = 1.6250,  b = -0.1250
Final cost (MSE) = 0.015625
Predicted price for size 3.5: $5.56k
```

> **Think**: What happens if you set `alpha = 1.0`? Try it and observe! 🔍

---

---

# Part 2 — Linear Regression (Vectorized Form with NumPy)

> Vectorized code replaces Python loops with NumPy matrix operations.  
> It is much faster and is how real implementations work.

**Key NumPy operations you need:**

| Operation | NumPy |
|-----------|-------|
| Matrix multiply | `A @ B` or `np.dot(A, B)` |
| Element-wise multiply | `A * B` |
| Sum all elements | `np.sum(arr)` |
| Transpose | `A.T` |

---

## 🔧 Problem 2.1 — Vectorized Prediction

$$\hat{y} = Xw + b \quad \text{(all samples at once)}$$

Here $X$ is a matrix of shape $(m, n)$, $w$ is a vector of shape $(n,)$, and $b$ is a scalar.

```python
def predict_vec(X, w, b):
    """
    Vectorized prediction: y_hat = X @ w + b

    Parameters
    ----------
    X : np.ndarray, shape (m, n)  — m samples, n features
    w : np.ndarray, shape (n,)    — weight vector
    b : float                     — bias

    Returns
    -------
    y_hat : np.ndarray, shape (m,)
    """
    # TODO: compute X @ w + b  (single line)
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

Use the `@` operator for matrix multiplication.  
`X @ w` gives a vector of shape $(m,)$, then add `b`.

</details>

<details>
<summary>✅ Solution</summary>

```python
def predict_vec(X, w, b):
    return X @ w + b     # (m, n) @ (n,) + scalar  →  (m,)
```

</details>

---

## 🔧 Problem 2.2 — Vectorized Cost (MSE)

$$J(w,b) = \frac{1}{2m} \lVert \hat{y} - y \rVert^2 = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$

```python
def compute_cost_vec(X, y, w, b):
    """
    Vectorized MSE cost.

    Parameters
    ----------
    X : np.ndarray, shape (m, n)
    y : np.ndarray, shape (m,)
    w : np.ndarray, shape (n,)
    b : float

    Returns
    -------
    cost : float
    """
    m = len(y)

    # TODO: step 1 — get predictions using predict_vec
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 2 — compute errors = y_hat - y
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 3 — return  sum(errors^2) / (2 * m)
    #        Hint: np.dot(errors, errors) gives sum of squares
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

- `errors = y_hat - y`  → a vector of shape $(m,)$
- `np.dot(errors, errors)` computes $\sum errors^2$  (same as `errors @ errors`)
- Divide by `2 * m`

</details>

<details>
<summary>✅ Solution</summary>

```python
def compute_cost_vec(X, y, w, b):
    m = len(y)
    y_hat  = predict_vec(X, w, b)        # (m,)
    errors = y_hat - y                   # (m,)
    cost   = np.dot(errors, errors) / (2 * m)
    return cost
```

</details>

---

## 🔧 Problem 2.3 — Vectorized Gradient Descent

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y), \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum(\hat{y} - y)$$

```python
def gradient_descent_vec(X, y, w, b, alpha, iterations):
    """
    Vectorized gradient descent.

    Parameters
    ----------
    X          : np.ndarray, shape (m, n)
    y          : np.ndarray, shape (m,)
    w          : np.ndarray, shape (n,)   — initial weights
    b          : float                    — initial bias
    alpha      : float                    — learning rate
    iterations : int

    Returns
    -------
    w, b, cost_history
    """
    m = len(y)
    cost_history = []

    for _ in range(iterations):

        # TODO: step 1 — compute predictions  →  y_hat
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        # TODO: step 2 — compute errors = y_hat - y
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        # TODO: step 3 — compute dw = (1/m) * X.T @ errors
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        # TODO: step 4 — compute db = (1/m) * np.sum(errors)
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        # TODO: step 5 — update w and b
        # -----------------------------------------------
        pass
        # -----------------------------------------------

        cost_history.append(compute_cost_vec(X, y, w, b))

    return w, b, cost_history
```

<details>
<summary>💡 Hint</summary>

Steps in order:
1. `y_hat = predict_vec(X, w, b)`
2. `errors = y_hat - y`
3. `dw = (1/m) * X.T @ errors`     ← shape: $(n, m) \cdot (m,) = (n,)$
4. `db = (1/m) * np.sum(errors)`
5. `w = w - alpha * dw`
6. `b = b - alpha * db`

</details>

<details>
<summary>✅ Solution</summary>

```python
def gradient_descent_vec(X, y, w, b, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        y_hat  = predict_vec(X, w, b)          # (m,)
        errors = y_hat - y                     # (m,)
        dw     = (1/m) * (X.T @ errors)        # (n,)
        db     = (1/m) * np.sum(errors)        # scalar
        w      = w - alpha * dw
        b      = b - alpha * db
        cost_history.append(compute_cost_vec(X, y, w, b))

    return w, b, cost_history
```

</details>

---

## 🧪 Example 2 — Putting It All Together (Vectorized)

> Same house-price dataset, but now with multiple features and vectorized code.

**Features**: `[size_100sqft, num_rooms]`

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Dataset  (m=5 samples, n=2 features) ---
X = np.array([
    [1.0, 1],
    [2.0, 2],
    [3.0, 2],
    [4.0, 3],
    [5.0, 3],
])
y = np.array([1.5, 3.5, 5.0, 6.5, 8.0])

# --- Initialize ---
m, n = X.shape
w = np.zeros(n)    # weight vector, one per feature
b = 0.0
alpha = 0.01
iterations = 1000

# --- Train ---
w, b, cost_history = gradient_descent_vec(X, y, w, b, alpha, iterations)
print(f"Learned  w = {w},  b = {b:.4f}")

# --- Evaluate ---
final_cost = compute_cost_vec(X, y, w, b)
print(f"Final cost = {final_cost:.6f}")

# --- Predict for a new house: size=3.5, rooms=2 ---
new_house = np.array([3.5, 2])
price = predict_vec(new_house.reshape(1, -1), w, b)[0]
print(f"Predicted price: ${price:.2f}k")

# --- Plot cost ---
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(w,b)')
plt.title('Vectorized Linear Regression — Cost Curve')
plt.show()
```

> **Think**: Why do we initialize `w = np.zeros(n)` instead of a single number now? 🤔

---

---

# Part 3 — Polynomial Regression

> Sometimes data follows a **curve**, not a straight line.  
> Polynomial regression handles this by creating new features: $x^2, x^3, \ldots, x^d$.

$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + \cdots + w_d x^d + b$$

This is still **linear regression** internally — we are linear in the *weights* $w$.  
The trick is in how we build the feature matrix.

---

## 📐 Feature Expansion

For a single input $x$ and degree $d$, we create a row:

$$\phi(x) = [x,\ x^2,\ x^3,\ \ldots,\ x^d]$$

For $m$ samples, we get a matrix of shape $(m, d)$.

---

## 🔧 Problem 3.1 — Build Polynomial Features

```python
def polynomial_features(X, degree):
    """
    Expand a 1-D feature array into polynomial features.

    Example: X = [2, 3],  degree = 3
             output = [[2, 4, 8],
                       [3, 9, 27]]   (each row: x, x^2, x^3)

    Parameters
    ----------
    X      : np.ndarray, shape (m,)  — original single feature
    degree : int                     — highest power to include

    Returns
    -------
    X_poly : np.ndarray, shape (m, degree)
    """
    # TODO: for powers 1, 2, ..., degree  → stack columns  X^1, X^2, ..., X^degree
    # Hint: use np.column_stack([X**p for p in range(1, degree+1)])
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

`X**p` raises every element of `X` to the power `p`.  
`np.column_stack([...])` places several 1-D arrays side by side as columns.

</details>

<details>
<summary>✅ Solution</summary>

```python
def polynomial_features(X, degree):
    return np.column_stack([X**p for p in range(1, degree + 1)])
    # Creates columns: X^1, X^2, ..., X^degree  →  shape (m, degree)
```

</details>

---

## 🔧 Problem 3.2 — Feature Normalization (Z-score)

> With high-degree features, values like $x^3$ can be enormous.  
> Normalization keeps all features on the same scale → faster, stable training.

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

```python
def normalize(X):
    """
    Z-score normalize each column of X.
    Returns normalized X, plus mean and std (needed to transform future data).

    Parameters
    ----------
    X : np.ndarray, shape (m, n)

    Returns
    -------
    X_norm : np.ndarray, shape (m, n)
    mu     : np.ndarray, shape (n,)   — mean of each column
    sigma  : np.ndarray, shape (n,)   — std  of each column
    """
    # TODO: compute mean and std along axis=0 (column-wise)
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: normalize:  X_norm = (X - mu) / sigma
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    return X_norm, mu, sigma
```

<details>
<summary>💡 Hint</summary>

- `np.mean(X, axis=0)` → mean of each column, shape `(n,)`
- `np.std(X, axis=0)`  → std of each column, shape `(n,)`
- Then simply: `(X - mu) / sigma`

</details>

<details>
<summary>✅ Solution</summary>

```python
def normalize(X):
    mu    = np.mean(X, axis=0)     # column means
    sigma = np.std(X, axis=0)      # column std deviations
    X_norm = (X - mu) / sigma      # z-score each column
    return X_norm, mu, sigma
```

</details>

---

## 🔧 Problem 3.3 — Train Polynomial Regression

> Now combine `polynomial_features`, `normalize`, and `gradient_descent_vec`  
> to train a polynomial regression model.

```python
def train_polynomial_regression(X, y, degree, alpha, iterations):
    """
    Full pipeline: expand features → normalize → train with gradient descent.

    Parameters
    ----------
    X          : np.ndarray, shape (m,)  — raw single feature
    y          : np.ndarray, shape (m,)
    degree     : int    — polynomial degree
    alpha      : float  — learning rate
    iterations : int

    Returns
    -------
    w, b, cost_history, mu, sigma
    """
    # TODO: step 1 — expand X to polynomial features using polynomial_features()
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 2 — normalize the expanded features using normalize()
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 3 — initialize w (zeros, shape=(degree,)) and b=0.0
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 4 — call gradient_descent_vec() and return results + mu, sigma
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

1. `X_poly = polynomial_features(X, degree)`      → shape `(m, degree)`
2. `X_norm, mu, sigma = normalize(X_poly)`
3. `w = np.zeros(degree)`,  `b = 0.0`
4. `w, b, cost_history = gradient_descent_vec(X_norm, y, w, b, alpha, iterations)`
5. `return w, b, cost_history, mu, sigma`

</details>

<details>
<summary>✅ Solution</summary>

```python
def train_polynomial_regression(X, y, degree, alpha, iterations):
    X_poly           = polynomial_features(X, degree)
    X_norm, mu, sigma = normalize(X_poly)
    w                = np.zeros(degree)
    b                = 0.0
    w, b, cost_history = gradient_descent_vec(X_norm, y, w, b, alpha, iterations)
    return w, b, cost_history, mu, sigma
```

</details>

---

## 🔧 Problem 3.4 — Prediction with Polynomial Model

> To predict new values, we must apply the same feature expansion and normalization.

```python
def predict_polynomial(X_new, w, b, degree, mu, sigma):
    """
    Predict with a trained polynomial model.

    Parameters
    ----------
    X_new  : np.ndarray, shape (m,)  — new raw input values
    w      : np.ndarray, shape (degree,)
    b      : float
    degree : int
    mu     : np.ndarray, shape (degree,)  — from training normalization
    sigma  : np.ndarray, shape (degree,)  — from training normalization

    Returns
    -------
    y_hat : np.ndarray, shape (m,)
    """
    # TODO: step 1 — expand X_new to polynomial features
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 2 — normalize using the TRAINING mu and sigma (not new ones!)
    # -----------------------------------------------
    pass
    # -----------------------------------------------

    # TODO: step 3 — call predict_vec and return
    # -----------------------------------------------
    pass
    # -----------------------------------------------
```

<details>
<summary>💡 Hint</summary>

- Same two steps as training (expand then normalize), **but use `mu` and `sigma` from training** — never recompute them on new data.
- Then `predict_vec(X_norm_new, w, b)`

</details>

<details>
<summary>✅ Solution</summary>

```python
def predict_polynomial(X_new, w, b, degree, mu, sigma):
    X_poly    = polynomial_features(X_new, degree)
    X_norm    = (X_poly - mu) / sigma          # use training stats!
    return predict_vec(X_norm, w, b)
```

</details>

---

## 🧪 Example 3 — Fitting a Curved Relationship

> A car's fuel consumption follows a curve: it drops steeply at first, then levels off.  
> A straight line won't capture this — but a polynomial will!

```python
import numpy as np
import matplotlib.pyplot as plt

# --- Dataset: speed (km/h) vs fuel consumption (L/100km) ---
X = np.array([20., 40., 60., 80., 100., 120., 140.])
y = np.array([8.5, 6.0, 5.2, 5.5,  6.3,  7.8,  9.5])

# --- Hyperparameters ---
degree     = 3
alpha      = 0.1
iterations = 5000

# --- Train ---
w, b, cost_history, mu, sigma = train_polynomial_regression(
    X, y, degree, alpha, iterations
)
print(f"Learned weights: {w}")
print(f"Learned bias:    {b:.4f}")
print(f"Final cost:      {cost_history[-1]:.6f}")

# --- Smooth curve for plotting ---
X_plot = np.linspace(20, 140, 200)
y_plot = predict_polynomial(X_plot, w, b, degree, mu, sigma)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', zorder=5, label='Data points')
plt.plot(X_plot, y_plot, color='red', label=f'Degree-{degree} polynomial')
plt.xlabel('Speed (km/h)')
plt.ylabel('Fuel Consumption (L/100km)')
plt.title('Polynomial Regression — Fuel vs Speed')
plt.legend()
plt.grid(True)
plt.show()

# --- Cost curve ---
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost over Training')
plt.show()

# --- Predict at a new speed ---
new_speed = np.array([90.])
pred = predict_polynomial(new_speed, w, b, degree, mu, sigma)
print(f"\nPredicted fuel at 90 km/h: {pred[0]:.2f} L/100km")
```

**Expected output (approximately)**:
```
Learned weights: [ ... ]
Learned bias:    5.7xxx
Final cost:      ~0.01
Predicted fuel at 90 km/h: ~5.7 L/100km
```

---

## 🧠 Thinking Questions

Answer these on your own before checking:

1. **Degree matters**: Train the fuel model with `degree=1` (straight line) and `degree=5`. What do you observe? Which is better?

2. **Why normalize?** What would happen to training if we did NOT normalize the polynomial features? Try it.

3. **Overfitting check**: With `degree=6` and only 7 data points, will the model generalize? What does the cost look like?

4. **Reuse mu/sigma**: Why is it *wrong* to call `normalize()` again on new test data instead of reusing `mu` and `sigma` from training?

<details>
<summary>💡 Answers</summary>

1. `degree=1` won't capture the curve — high cost. `degree=5` might overfit (memorizes data). `degree=3` is usually a good balance here.

2. Without normalization, $x^3$ can reach values in the millions, causing gradients to explode. Training will diverge or be extremely slow.

3. With more parameters than data points, the model can fit training data perfectly (cost ≈ 0) but will perform badly on unseen data — this is **overfitting**.

4. Using training `mu/sigma` ensures consistent scaling. If you re-normalize test data independently, the scale will shift and predictions will be wrong.

</details>

---

## 📊 Summary: What You Built

| Function | Used In | What It Does |
|----------|---------|--------------|
| `predict()` | Part 1 | $\hat{y} = wx + b$ (loop) |
| `compute_cost()` | Part 1 | MSE cost (loop) |
| `gradient_descent()` | Part 1 | Trains w, b (loop) |
| `predict_vec()` | Part 2 & 3 | $\hat{y} = Xw + b$ (vectorized) |
| `compute_cost_vec()` | Part 2 & 3 | MSE cost (vectorized) |
| `gradient_descent_vec()` | Part 2 & 3 | Trains w, b (vectorized) |
| `polynomial_features()` | Part 3 | Expands $x$ to $[x, x^2, \ldots, x^d]$ |
| `normalize()` | Part 3 | Z-score normalization |
| `train_polynomial_regression()` | Part 3 | Full poly training pipeline |
| `predict_polynomial()` | Part 3 | Predict with trained poly model |

> ✅ The vectorized functions (Part 2) are **reused** in Part 3 — no code duplication needed!
