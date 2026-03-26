# Bias, Variance & the Bias-Variance Tradeoff

## 📚 Overview

Every machine learning model makes errors on data it has never seen. Understanding **where those errors come from** is the key to fixing them.

In this lesson we will:
- Define **bias** and **variance** precisely
- Show what harm each one causes to your model
- Identify them with numerical examples and diagnostic plots
- Learn the strategies to address both problems

---

## 🎯 The Big Picture: Sources of Model Error

When a model's predictions are wrong, the error always comes from one or more of three sources:

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Source | What it is | Can we fix it? |
|--------|-----------|---------------|
| **Bias²** | Error from wrong assumptions | ✅ Yes — more complex model |
| **Variance** | Error from sensitivity to training data | ✅ Yes — simpler model / more data |
| **Irreducible Noise** | Random noise in the data itself | ❌ No — it is part of reality |

> [!NOTE]
> **Irreducible noise** comes from unmeasured factors, measurement errors, or genuine randomness. No matter how perfect your model is, it cannot remove this. Our job is to minimise bias and variance while accepting the noise floor.

---

## 🔍 What Is Bias?

**Bias** is the error caused by a model being too simple to capture the true pattern in the data.

A **high-bias** model makes strong (wrong) assumptions about the data. It consistently predicts in the wrong direction — it doesn't just miss randomly, it misses *systematically*.

### Analogy: The Stubborn Archer

Imagine an archer who always aims at the same fixed spot regardless of where the target moves. Every arrow clusters in the same wrong spot. This is **bias**: a systematic, consistent error.

```
True target:  ●

Predictions:
   ✗  ✗
    ✗ ✗      ← always missing left — consistent, systematic error
```

### Formal definition

If we could train the model on many different datasets drawn from the same distribution, **bias** is the difference between the *average* prediction and the true value:

$$\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)$$

Where:
- $\hat{f}(x)$ is the model's prediction
- $f(x)$ is the true underlying function
- $\mathbb{E}[\cdot]$ is the expected value (average over many training sets)

**High bias ≡ Underfitting**

Signs:
- High training error
- High test/validation error
- Both errors are similar (the model is equally bad on everything)

---

## 🔍 What Is Variance?

**Variance** is the error caused by a model being too sensitive to the specific training data it was given. A **high-variance** model learns the noise in the training data as if it were a real pattern.

### Analogy: The Jittery Archer

Now imagine an archer whose aim jumps all over the place. Sometimes they hit the target, sometimes wildly miss. Each shot is in a completely different place. This is **variance**: predictions are inconsistent and unpredictable.

```
True target:  ●

Predictions:
         ✗
  ✗           ✗         ← spread out everywhere, not consistent
           ✗
    ✗
```

### Formal definition

**Variance** measures how much the model's prediction changes when trained on different datasets:

$$\text{Variance}[\hat{f}(x)] = \mathbb{E}\left[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\right]$$

It is the expected squared deviation of a prediction from its own average.

**High variance ≡ Overfitting**

Signs:
- Very low training error
- Much higher test/validation error
- Large **gap** between training and test error

---

## 🎯 The Bias-Variance Tradeoff

Bias and variance pull in **opposite directions**. As you make a model more complex:

- Bias **decreases** (the model can fit subtler patterns)
- Variance **increases** (the model becomes more sensitive to noise)

```
Error
 ▲
 │ \
 │  \        Total Error      ← U-shaped curve
 │   \      / ‾‾‾‾‾‾‾‾\
 │    \___/             \
 │     ← Bias²           \
 │            Variance →   \
 └──────────────────────────► Model Complexity
         ↑
    Sweet spot (optimal complexity)
```

The **goal** is to find the sweet spot where total error (bias² + variance) is minimised.

| Model complexity | Bias | Variance | Training Error | Test Error |
|-----------------|------|----------|---------------|-----------|
| Too simple | High | Low | High | High |
| Just right | Low | Low | Moderate | Moderate (minimum) |
| Too complex | Low | High | Very low | High |

---

## 🔢 Numerical Example: Polynomial Regression

Suppose the **true relationship** is: $f(x) = 0.5x^2 + 2$

We have 10 training points with some noise:

```
x:  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,   6
y:   6.8, 3.9,  2.6, 2.1, 2.4, 4.1, 6.7, 9.9, 14.6, 19.8
     (true values + small random noise)
```

We try three model complexities:

### Model A — Degree 1 (straight line)
Fits: $\hat{y} = wx + b$

Best fit line: $\hat{y} = 2.3x + 3.1$

| x | True $y$ | Predicted $\hat{y}$ | Error |
|---|----------|---------------------|-------|
| -3 | 6.8 | -3.8 | -10.6 |
| 0 | 2.1 | 3.1 | +1.0 |
| 6 | 19.8 | 16.9 | -2.9 |

Training MSE ≈ **18.4** — Test MSE ≈ **19.1** → **High bias. Both errors are large and similar.**

### Model B — Degree 2 (quadratic — matches true function)
Fits: $\hat{y} = w_2 x^2 + w_1 x + b$

Best fit: $\hat{y} = 0.49x^2 + 0.03x + 2.05$

| x | True $y$ | Predicted $\hat{y}$ | Error |
|---|----------|---------------------|-------|
| -3 | 6.8 | 6.6 | -0.2 |
| 0 | 2.1 | 2.1 | 0.0 |
| 6 | 19.8 | 19.8 | 0.0 |

Training MSE ≈ **0.09** — Test MSE ≈ **0.12** → **Both errors are low. Good model.**

### Model C — Degree 9 (over-parameterised)
The polynomial passes *exactly* through every training point.

Training MSE ≈ **0.00** — Test MSE ≈ **142.7** → **Massive gap. High variance (overfit).**

```
Visual summary:

  y │           * Model C (degree 9): passes every point but wild on new data
    │          /\
    │  *      /  \        * Model A (degree 1): misses the curve entirely
    │   \    /    \_      _ _ _ _ _ _ ← flat ish line
    │    * *         \  /
    │                 *
    └────────────────────► x
                    
    ●●●●● Model B (degree 2): follows the true shape closely
```

---

## ⚠️ The Harm They Cause

### Harm of High Bias (Underfitting)
- The model **misses the real relationship** in the data
- Adding more training data barely helps (the model is too rigid to use it)
- Leads to poor performance in production → wrong decisions
- Example: predicting house prices with just one feature when many matter

### Harm of High Variance (Overfitting)
- The model works great on training data but **fails on new data**
- Can't be trusted in production
- Wastes computational resources on a model that doesn't generalise
- Example: memorising training patients' exact lab values → useless for new patients

---

### How to read learning curves

| Pattern | Diagnosis |
|---------|-----------|
| Both curves are high and converge together | **High Bias** — model is too simple |
| Training error is very low, validation is much higher, large gap | **High Variance** — model too complex |
| Both curves are low and close together | **Good fit** |
| Validation error still decreasing at max training size | Need **more data** |

---

## 🛠️ How to Fix Bias (Underfitting)

| Strategy | Why it helps |
|----------|-------------|
| Use a more complex model (higher polynomial degree, more layers) | Gives the model more capacity to learn the pattern |
| Add more features | Supply more information about the data |
| Reduce regularization ($\lambda$) | Allows weights to grow and capture more of the pattern |
| Train for more iterations | Ensures the model has converged |

> Adding more training data does **NOT** fix high bias. A simple model will make the same systematic error whether it sees 100 or 1,000,000 examples.

---

## 🛠️ How to Fix Variance (Overfitting)

| Strategy | Why it helps |
|----------|-------------|
| **Get more training data** | More data makes it harder to memorise noise |
| **Add regularization (L1 or L2)** | Discourages large weights that create wild oscillations |
| **Reduce model complexity** | Fewer parameters → less capacity to memorise noise |
| **Feature selection / dimensionality reduction** | Remove noisy, irrelevant features |
| **Cross-validation** | Gives a reliable estimate of how bad overfitting is |
| **Early stopping** (advanced) | Stop training before the model starts memorising |

---

## 📊 Complete Comparison Table

| Aspect | High Bias | High Variance |
|--------|-----------|---------------|
| Other name | Underfitting | Overfitting |
| Training error | High | Very low |
| Test / val error | High | Much higher than training |
| Gap (test − train) | Small | Large |
| Model complexity | Too simple | Too complex |
| More data helps? | ❌ No | ✅ Yes |
| Regularization helps? | ❌ Not directly | ✅ Yes |
| Reduce complexity helps? | ❌ No | ✅ Yes |
| Increase complexity helps? | ✅ Yes | ❌ No |

---

## 🧩 Putting It All Together: Workflow

When you deploy a model and performance is bad, follow this decision tree:

```
        Bad model performance
               │
    ┌──────────┴──────────┐
    │                     │
Training error         Training error
  also high              is low
    │                     │
High BIAS             High VARIANCE
(Underfitting)        (Overfitting)
    │                     │
    ▼                     ▼
• More complex model  • More training data
• More features       • Add regularization
• Less regularization • Simpler model
• More iterations     • Feature selection
```

---

## 🧪 Q&A — Test Your Understanding

### Q1. Concept Check

A student trains a logistic regression model on a spam dataset. The results are:
- Training accuracy: **72 %**
- Validation accuracy: **70 %**

Is this model suffering from high bias, high variance, or neither? What should they try?

<details>
<summary>💡 Hint</summary>

Look at both the absolute values and the gap between training and validation accuracy.
</details>

<details>
<summary>✅ Answer</summary>

**High Bias (Underfitting).**

Both training and validation accuracy are low (72 % and 70 %) and they are very close together — there is almost no gap. This is the textbook signature of underfitting: the model is not complex enough to capture the pattern.

**What to try:**
- Add more features (e.g., bigrams, email metadata)
- Try a more complex model (polynomial features, or a different algorithm)
- Reduce regularization if any is being used
- Do **not** add more data — it won't help a model that is already failing on training data
</details>

---

### Q2. Concept Check

A second team trains a polynomial regression (degree 10) on the same spam dataset. Their results:
- Training accuracy: **99 %**
- Validation accuracy: **61 %**

What is happening? What should they try?

<details>
<summary>✅ Answer</summary>

**High Variance (Overfitting).**

The model achieves near-perfect accuracy on training data but performs much worse on the validation set. The gap (99 % vs 61 % = **38 %**) is very large. The model has memorised the training examples including their noise.

**What to try:**
- Reduce model complexity (lower polynomial degree)
- Add L2 regularization (Ridge) — increase $\lambda$
- Collect more training data
- Remove noisy/irrelevant features
</details>

---

### Q3. Numerical Reasoning

You train three models on the same dataset. Using 5-fold cross-validation you get these MSE values:

| Model | Train MSE | Validation MSE |
|-------|-----------|----------------|
| A | 85.2 | 87.1 |
| B | 1.3 | 2.1 |
| C | 0.4 | 94.8 |

1. Identify whether each model is underfitting, overfitting, or a good fit.
2. Which model would you deploy?
3. What would you do to improve Model C?

<details>
<summary>✅ Answer</summary>

1. **Model A**: Both errors are high and close → **Underfitting (high bias)**  
   **Model B**: Both errors are low and close → **Good fit ✅**  
   **Model C**: Training error is near-zero, validation error is huge → **Overfitting (high variance)**

2. **Deploy Model B** — it generalises well (low, balanced errors).

3. To improve Model C:
   - Add L2 (Ridge) regularization — increase $\lambda$ to penalise large weights
   - Reduce model complexity (fewer parameters / polynomial degree)
   - Collect more training data so the model can't memorise all examples
   - Apply cross-validation to select the best regularization strength
</details>

---

### Q4. Learning Curve Interpretation

You plot the learning curve for your model and observe:

```
MSE
 ▲
 │
 │ ●─────────────────────── Validation error  (stays high ~50)
 │
 │
 │ ●───────────────────────── Training error  (stays high ~48)
 │
 └─────────────────────────────────────────► Training set size
```

Both curves are high and flat, converging near each other at a high MSE.

1. What does this tell you?
2. If you doubled the training data, what would happen?
3. What would you change in the model?

<details>
<summary>✅ Answer</summary>

1. **High bias (underfitting)**. The model is too simple to learn the underlying pattern. Training error is already high, which means it fails even on data it has seen.

2. **Doubling training data would not help** in a high-bias situation. Both curves are already converged — more data just gives the model more examples it can't properly learn from. The validation error won't decrease.

3. **Increase model complexity**: use polynomial features, add more features, switch to a more powerful algorithm (e.g., decision tree instead of linear model), or reduce regularization if any is applied.
</details>

---

### Q5. Connecting Regularization (previous lesson) to Bias-Variance

In the previous lesson you learned that increasing $\lambda$ (regularization strength) shrinks model weights toward zero.

1. Does increasing $\lambda$ increase or decrease model bias?
2. Does increasing $\lambda$ increase or decrease model variance?
3. If a model is currently overfitting, what does increasing $\lambda$ do to the bias-variance tradeoff?

<details>
<summary>✅ Answer</summary>

1. **Increases bias** — by forcing weights toward zero, the model is prevented from fitting the training data as closely. It makes more systematic errors.

2. **Decreases variance** — smaller weights mean the model changes less when the training data changes. It becomes more stable and generalises better.

3. With an overfitting model (high variance), increasing $\lambda$ **trades some variance for bias**. The model becomes slightly less flexible (bias rises a little) but much more stable on unseen data (variance drops a lot). The net effect on total error is **positive**: total error decreases if the model was truly overfitting.

This is the fundamental mechanism by which regularization helps: it pushes the model toward the sweet spot of the bias-variance tradeoff.
</details>

---

### Q6. Code Challenge

Fill in the blanks to compute bias and variance empirically by training on multiple different random splits of a dataset:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

np.random.seed(0)

# True function
def true_f(x):
    return 0.5 * x**2 + 2

# Full dataset
X_all = np.sort(np.random.rand(500) * 10 - 2)
y_all = true_f(X_all) + np.random.randn(500) * 1.5   # add noise

# Fixed test point to evaluate at
x_test = np.array([[3.0]])

n_experiments = 50
train_size    = 30
predictions   = []

for _ in range(n_experiments):
    # Randomly sample a training set
    idx = np.random.choice(len(X_all), size=train_size, replace=False)
    X_train = X_all[idx].reshape(-1, 1)
    y_train = y_all[idx]

    degree = 2   # try 1, 2, 9 and compare results
    model = Pipeline([
        ("poly", PolynomialFeatures(degree)),
        ("lr",   LinearRegression())
    ])
    model.fit(X_train, y_train)

    pred = model.predict(x_test)[0]
    predictions.append(pred)

predictions = np.array(predictions)
true_value  = true_f(x_test[0, 0])

# TODO: compute bias and variance from the 50 predictions
mean_pred = ???           # average prediction across all experiments
bias      = ???           # difference between mean prediction and true value
variance  = ???           # average squared deviation from mean prediction
mse       = ???           # total error (should ≈ bias^2 + variance + noise)

print(f"True value     : {true_value:.3f}")
print(f"Mean prediction: {mean_pred:.3f}")
print(f"Bias           : {bias:.3f}")
print(f"Bias²          : {bias**2:.3f}")
print(f"Variance       : {variance:.3f}")
print(f"Bias² + Var    : {bias**2 + variance:.3f}")
print(f"Observed MSE   : {mse:.3f}")
```

<details>
<summary>✅ Solution</summary>

```python
mean_pred = predictions.mean()
bias      = mean_pred - true_value
variance  = ((predictions - mean_pred)**2).mean()
mse       = ((predictions - true_value)**2).mean()

# Note: mse ≈ bias² + variance + irreducible noise
# Try degree=1 → high bias, low variance
# Try degree=2 → low bias, low variance  (best)
# Try degree=9 → low bias (on average), high variance (predictions scattered)
```

**Expected findings:**

| Degree | Bias² | Variance | MSE |
|--------|-------|----------|-----|
| 1 | High | Low | High |
| 2 | Low | Low | Low (best) |
| 9 | Near 0 | High | High |

This numerically demonstrates the bias-variance tradeoff: degree 9 actually achieves near-zero bias at a single test point *on average*, but its predictions are wildly scattered — the variance alone makes it a bad model.
</details>

---

## 🎓 Key Takeaways

> [!IMPORTANT]
> **Bias**
> - Error from wrong assumptions — the model is *systematically* wrong
> - Symptom: high training error AND high test error with a small gap
> - Fix: increase model complexity, add features, reduce regularization

> [!IMPORTANT]
> **Variance**
> - Error from sensitivity to the specific training data — the model memorises noise
> - Symptom: very low training error, much higher test error (large gap)
> - Fix: more data, add regularization (L1/L2), reduce model complexity

> [!IMPORTANT]
> **The Tradeoff**
> - Increasing complexity → bias ↓, variance ↑
> - Decreasing complexity → bias ↑, variance ↓
> - Goal: find the sweet spot that minimises **bias² + variance** on unseen data
> - Use **learning curves** to diagnose which problem you have before choosing a fix
> - Regularization is the most powerful lever for controlling this tradeoff

---

*Next lesson → Model Evaluation & Selection: metrics, cross-validation, and how to choose the right model.*
