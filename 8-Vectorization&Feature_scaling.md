# Vectorization & Feature Scaling

## 1. Linear Regression with Multiple Features

In basic linear regression, we often start with a single variable (feature). However, in real-world problems, we usually have multiple features that influence the outcome.

### Notation

Let's define our notation for multiple features:

- $n$: number of features.
- $m$: number of training examples.
- $x^{(i)}$: input (features) of the $i^{th}$ training example.
- $x^{(i)}_j$: value of feature $j$ in the $i^{th}$ training example.

### Hypothesis Function

With multiple features, our hypothesis function changes from a simple line equation to a linear combination of all features:

$$ h_w(x) = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n $$

For convenience, we define $x_0 = 1$ (intercept term). This allows us to write:

$$ h_w(x) = w_0x_0 + w_1x_1 + \dots + w_nx_n $$

---

## 2. Vectorization

Vectorization is the process of converting an algorithm that operates on a single value at a time into an operation that operates on a set of values (vectors) at once.

### The Vector View
Instead of treating $w$ and $x$ as separate lists of numbers, we view them as **Vectors**:

$$ x = \begin{bmatrix} x_0 \\ x_1 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^{n+1} \quad \text{and} \quad w = \begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_n \end{bmatrix} \in \mathbb{R}^{n+1} $$

Using linear algebra, our hypothesis function typically becomes a dot product:

$$ h_w(x) = w^T x $$

### Why Vectorize?
1.  **Code Conciseness**: Mathematical notation translates directly to code.
2.  **Computational Efficiency**: Modern CPUs and GPUs use **SIMD** (Single Instruction, Multiple Data) to perform operations on vectors in parallel, largely outperforming explicit `for` loops.

### Code Example: Loop vs. Vectorization

Here is a Python comparison using NumPy (Numerical Python).

```python
import numpy as np
import time

# Create two large vectors with 1 million elements
n = 1000000
x = np.random.rand(n)
w = np.random.rand(n)

# --- 1. Non-Vectorized Version (Explicit Loop) ---
start_time = time.time()
prediction_loop = 0
for i in range(n):
    prediction_loop += w[i] * x[i]
end_time = time.time()

print(f"Loop Result: {prediction_loop}")
print(f"Loop Time: {end_time - start_time:.6f} seconds")

# --- 2. Vectorized Version ---
start_time = time.time()
prediction_vector = np.dot(w, x) # This happens in parallel on the CPU
end_time = time.time()

print(f"Vectorized Result: {prediction_vector}")
print(f"Vectorized Time: {end_time - start_time:.6f} seconds")
```

**Outcome**: You will typically observe that the vectorized version is orders of magnitude faster (e.g., 300x faster) than the loop version.

---

## 3. Feature Scaling

Feature scaling is a method used to normalize the range of independent variables or features of data.

### Why do we need it?
If you have one feature $x_1$ (e.g., size of house) ranging from 0 to 2000, and another feature $x_2$ (e.g., number of bedrooms) ranging from 1 to 5, the cost function $J(w)$ contours will be skewed (elliptical).

- **Without Scaling**: Gradient Descent will oscillate and take a long time to reach the global minimum.
- **With Scaling**: The contours become more circular, allowing Gradient Descent to take a direct path to the minimum.

### Method 1: Mean Normalization
Rescales features so that they have approximately zero mean.

$$ x_i := \frac{x_i - \mu_i}{max(x) - min(x)} $$

Where $\mu_i$ is the average of all values for feature (i). The resulting values usually range between -0.5 and 0.5.

### Method 2: Z-score Normalization (Standardization)
Rescales features so that they have a mean of 0 and a standard deviation of 1.

$$ x_i := \frac{x_i - \mu_i}{\sigma_i} $$

Where $\sigma_i$ is the standard deviation.

### Code Snippet

```python
import numpy as np

features = np.array([2000, 1500, 1000, 2500]) # e.g., house sizes

# Mean Normalization
mean_val = np.mean(features)
range_val = np.max(features) - np.min(features)
normalized = (features - mean_val) / range_val

# Z-score Normalization
std_dev = np.std(features)
z_scored = (features - mean_val) / std_dev

print(f"Original: {features}")
print(f"Mean Normalized: {normalized}")
print(f"Z-Score Normalized: {z_scored}")
```

---

## 4. Practice & Tests

Take your time to think through these problems.

### Problem 1: Vector Dimension
**Question**: You implement linear regression with $n=5$ features ($x_1$ to $x_5$). You include the intercept term $x_0 = 1$. What is the dimension of the parameter vector $w$?
<details>
<summary>Click to reveal Solution</summary>
The dimension is $n+1 = 6$. The vector is $[w_0, w_1, w_2, w_3, w_4, w_5]$.
</details>

---

### Problem 2: Computational Logic
**Question**: Why does the vectorized implementation `np.dot(w, x)` run faster than a `for` loop in Python?
<details>
<summary>Click to reveal Solution</summary>
Python loops are interpreted and generally slow. The NumPy library calls optimized, pre-compiled C/C++ functions and utilizes SIMD (Single Instruction, Multiple Data) hardware capabilities to perform mathematical operations on arrays in parallel.
</details>

---

### Problem 3: Scaling Calculation
**Scenario**: You have a dataset for a feature "Age".
- Mean ($\mu$) = 40
- Standard Deviation ($\sigma$) = 10
- Max = 70
- Min = 10

**Question**: Calculate the **Z-score normalized** value for a person with Age = 60.

**Work your solution**:
$$ x_{new} = \frac{x - \mu}{\sigma} $$

<details>
<summary>Click to reveal Solution</summary>
$$ x_{new} = \frac{60 - 40}{10} = \frac{20}{10} = 2.0 $$
The normalized value is 2.0.
</details>
