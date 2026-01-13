# Unsupervised Learning: Clustering & Anomaly Detection

## Table of Contents
1. [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)
2. [Clustering & K-Means](#clustering--k-means)
3. [K-Means Algorithm Deep Dive](#k-means-algorithm-deep-dive)
4. [Initialization & Choosing K](#initialization--choosing-k)
5. [Anomaly Detection](#anomaly-detection)
6. [Gaussian Distribution](#gaussian-distribution)
7. [Assessment Questions](#assessment-questions)

---

## Introduction to Unsupervised Learning

### What is Unsupervised Learning?

Unlike supervised learning where we have labeled data $(x, y)$, in unsupervised learning we only have inputs $x$ **without labels**.

**Goal:** Discover hidden patterns, structures, or relationships in data.

### Supervised vs Unsupervised

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| **Data** | $(x, y)$ pairs | Only $x$ |
| **Goal** | Predict $y$ from $x$ | Find structure in $x$ |
| **Examples** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Evaluation** | Compare to true labels | No ground truth |

### Main Types of Unsupervised Learning

1. **Clustering**: Group similar data points together
   - K-Means, Hierarchical Clustering, DBSCAN

2. **Dimensionality Reduction**: Reduce number of features
   - PCA, t-SNE, Autoencoders

3. **Anomaly Detection**: Identify unusual patterns
   - Gaussian models, Isolation Forest

4. **Association**: Find relationships between variables
   - Market basket analysis

This lesson focuses on **clustering (K-Means)** and **anomaly detection**.

---

## Clustering & K-Means

### What is Clustering?

**Clustering** is the task of grouping data points so that:
- Points in the **same cluster** are similar
- Points in **different clusters** are dissimilar

### Real-World Applications

📊 **Customer Segmentation**: Group customers by behavior  
🧬 **Gene Analysis**: Group genes with similar expression patterns  
📰 **Document Organization**: Group similar articles  
🖼️ **Image Compression**: Reduce colors by clustering  
🏢 **Market Research**: Identify market segments

### K-Means: The Most Popular Clustering Algorithm

**K-Means** partitions data into $K$ clusters by:
- Assigning each point to the nearest cluster center (centroid)
- Updating centroids as the mean of assigned points
- Repeating until convergence


---

## K-Means Algorithm Deep Dive

### Algorithm Steps

**Input:**
- Dataset: $\{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$ where $x^{(i)} \in \mathbb{R}^n$
- Number of clusters: $K$

**Process:**

1. **Initialize** $K$ cluster centroids $\mu_1, \mu_2, ..., \mu_K \in \mathbb{R}^n$ randomly

2. **Repeat until convergence:**
   
   **Step A - Cluster Assignment:**
   For each data point $i = 1, ..., m$:
   $$c^{(i)} := \arg\min_{k} \|x^{(i)} - \mu_k\|^2$$
   
   Assign $x^{(i)}$ to the closest centroid
   
   **Step B - Move Centroids:**
   For each cluster $k = 1, ..., K$:
   $$\mu_k := \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$$
   
   Where $C_k$ is the set of points assigned to cluster $k$

3. **Stop** when centroids don't change (or change is minimal)

### Cost Function (Distortion)

K-Means minimizes the **within-cluster sum of squares**:

$$J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$

Where:
- $c^{(i)}$ = index of cluster to which $x^{(i)}$ is assigned
- $\mu_k$ = centroid of cluster $k$
- $\mu_{c^{(i)}}$ = centroid of the cluster to which $x^{(i)}$ is assigned

**Interpretation:** Average squared distance from each point to its assigned centroid.

### How K-Means Optimizes the Cost

**Cluster Assignment Step** minimizes $J$ with respect to $c^{(1)}, ..., c^{(m)}$:
- For each point, choose the closest centroid
- This is the optimal assignment given current centroids

**Move Centroids Step** minimizes $J$ with respect to $\mu_1, ..., \mu_K$:
- For each cluster, the mean is the point that minimizes sum of squared distances
- Proof: Taking derivative and setting to zero gives the mean

**Guarantee:** Each iteration either decreases $J$ or keeps it the same. **K-Means always converges!**

However, it may converge to a **local minimum**, not global minimum.

### Code Example: K-Means from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, K, max_iters=100):
    """
    K-Means clustering from scratch
    
    Args:
        X: (m, n) data matrix
        K: number of clusters
        max_iters: maximum iterations
    
    Returns:
        centroids: (K, n) final centroids
        assignments: (m,) cluster assignments
        costs: list of cost at each iteration
    """
    m, n = X.shape
    
    # Random initialization
    centroids = X[np.random.choice(m, K, replace=False)]
    costs = []
    
    for iteration in range(max_iters):
        # Step 1: Assign points to nearest centroid
        distances = np.zeros((m, K))
        for k in range(K):
            distances[:, k] = np.sum((X - centroids[k])**2, axis=1)
        
        assignments = np.argmin(distances, axis=1)
        
        # Compute cost
        cost = np.mean([distances[i, assignments[i]] for i in range(m)])
        costs.append(cost)
        
        # Step 2: Update centroids
        new_centroids = np.zeros((K, n))
        for k in range(K):
            points_in_cluster = X[assignments == k]
            if len(points_in_cluster) > 0:
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize randomly
                new_centroids[k] = X[np.random.choice(m)]
        
        # Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration}")
            break
            
        centroids = new_centroids
    
    return centroids, assignments, costs

# Example usage
np.random.seed(42)
X = np.random.randn(300, 2)
X[:100] += [2, 2]
X[100:200] += [-2, 2]
X[200:] += [0, -2]

centroids, assignments, costs = kmeans(X, K=3)

print(f"Final cost: {costs[-1]:.4f}")
print(f"Converged in {len(costs)} iterations")
```

### Using Scikit-Learn

```python
from sklearn.cluster import KMeans

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42, max_iter=100)
kmeans.fit(X)

# Get results
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_  # Same as our cost function J

print(f"Inertia (cost): {inertia:.4f}")
print(f"Iterations: {kmeans.n_iter_}")
```

### 🤔 Think About It
*Why does K-Means always converge? What prevents it from oscillating forever?*

---

## Initialization & Choosing K

### The Initialization Problem

K-Means can get stuck in local minima. Different initializations lead to different results!

**Bad initialization example:**


### Random Initialization (Standard Method)

**Algorithm:**
1. Randomly pick $K$ training examples
2. Set $\mu_1, ..., \mu_K$ equal to these examples

```python
def random_initialization(X, K):
    """Randomly select K points as initial centroids"""
    m = X.shape[0]
    random_indices = np.random.choice(m, K, replace=False)
    return X[random_indices]
```

**Problem:** Can lead to local minima depending on initial choice.

### Solution: Multiple Random Initializations

**Algorithm:**
```
For i = 1 to 50-100:
    Randomly initialize K-Means
    Run K-Means to convergence
    Compute cost J
    
Choose the clustering with lowest cost J
```

**Code Example:**

```python
def kmeans_multiple_init(X, K, n_init=50):
    """Run K-Means multiple times, return best result"""
    best_cost = float('inf')
    best_centroids = None
    best_assignments = None
    
    for _ in range(n_init):
        centroids, assignments, costs = kmeans(X, K)
        final_cost = costs[-1]
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_centroids = centroids
            best_assignments = assignments
    
    return best_centroids, best_assignments, best_cost

# Scikit-learn does this automatically!
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
```

### Choosing the Number of Clusters K

This is often the hardest part! No definitive answer, but several methods help:

#### Method 1: Elbow Method

Plot cost $J$ vs number of clusters $K$:


**Look for "elbow"**: Point where cost stops decreasing rapidly.

**Code Example:**

```python
costs = []
K_range = range(1, 11)

for K in K_range:
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    kmeans.fit(X)
    costs.append(kmeans.inertia_)

plt.plot(K_range, costs, 'bo-')
plt.xlabel('Number of Clusters K')
plt.ylabel('Cost (Inertia)')
plt.title('Elbow Method')
plt.show()

# Look for elbow visually
```

**Limitation:** Elbow not always clear!

#### Method 2: Silhouette Score

Measures how similar a point is to its own cluster vs other clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance to points in same cluster
- $b(i)$ = average distance to points in nearest other cluster

**Range:** $s(i) \in [-1, 1]$
- **+1**: Point is far from other clusters (good)
- **0**: Point is on cluster boundary
- **-1**: Point might be in wrong cluster (bad)

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for K in range(2, 11):
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={K}, Silhouette Score: {score:.3f}")

# Choose K with highest silhouette score
```

#### Method 3: Domain Knowledge

Often the best approach! Choose $K$ based on:
- Business requirements (e.g., 3 customer segments for marketing)
- Interpretability needs
- Downstream task requirements

### 🤔 Think About It
*Why does the cost J always decrease as K increases? Why not just use K=m (number of samples)?*

---

## Anomaly Detection

### What is Anomaly Detection?

**Anomaly Detection** identifies data points that are unusual or don't fit the normal pattern.

**Applications:**
- 🔒 **Fraud Detection**: Unusual credit card transactions
- 🏭 **Manufacturing**: Defective products
- 🖥️ **System Monitoring**: Server failures
- 🏥 **Healthcare**: Unusual patient vitals
- 🔐 **Security**: Network intrusion detection

### Problem Setup

**Training:** Given dataset $\{x^{(1)}, ..., x^{(m)}\}$ of **normal** examples

**Test:** For new example $x_{test}$, determine if it's:
- **Normal** (similar to training data)
- **Anomalous** (very different from training data)

### The Gaussian (Normal) Distribution Approach

Model the probability that a point is normal using Gaussian distribution.

---

## Gaussian Distribution

### Univariate Gaussian Distribution

A random variable $X$ has a Gaussian distribution if:

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

**Parameters:**
- $\mu$ = mean (center of distribution)
- $\sigma^2$ = variance (spread of distribution)
- $\sigma$ = standard deviation

**Notation:** $X \sim \mathcal{N}(\mu, \sigma^2)$

### Properties

- **Symmetric** around mean $\mu$
- **Bell-shaped** curve
- **68-95-99.7 rule:**
  - 68% of data within $\mu \pm \sigma$
  - 95% within $\mu \pm 2\sigma$
  - 99.7% within $\mu \pm 3\sigma$

### Parameter Estimation

Given data $\{x^{(1)}, ..., x^{(m)}\}$:

**Mean:**
$$\mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$

**Variance:**
$$\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2$$

### Code Example: Gaussian Distribution

```python
import numpy as np
from scipy.stats import norm

# Generate data from Gaussian
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

# Estimate parameters
mu = data.mean()
sigma = data.std()

print(f"Estimated μ: {mu:.2f}")
print(f"Estimated σ: {sigma:.2f}")

# Compute probability density for new point
x_new = 8.5
probability = norm.pdf(x_new, loc=mu, scale=sigma)
print(f"p(x={x_new}): {probability:.6f}")
```

### Multivariate Gaussian Distribution

For data with $n$ features, $x \in \mathbb{R}^n$:

$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

**Parameters:**
- $\mu \in \mathbb{R}^n$ = mean vector
- $\Sigma \in \mathbb{R}^{n \times n}$ = covariance matrix

**For independence assumption** (simpler model):

$$p(x) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) \times ... \times p(x_n; \mu_n, \sigma_n^2)$$

$$p(x) = \prod_{j=1}^{n} \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(x_j - \mu_j)^2}{2\sigma_j^2}\right)$$

### Anomaly Detection Algorithm

**Training:**

1. Choose features $x_j$ that might indicate anomalies
2. Fit parameters $\mu_1, ..., \mu_n$ and $\sigma_1^2, ..., \sigma_n^2$:
   $$\mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)}$$
   $$\sigma_j^2 = \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2$$

**Testing:**

For new example $x$:
1. Compute $p(x)$:
   $$p(x) = \prod_{j=1}^{n} p(x_j; \mu_j, \sigma_j^2)$$

2. Classify as anomaly if $p(x) < \epsilon$
   - $\epsilon$ is a threshold (chosen using validation set)

### Code Example: Anomaly Detection

```python
import numpy as np
from scipy.stats import norm

class AnomalyDetector:
    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
        self.mu = None
        self.sigma = None
    
    def fit(self, X):
        """Fit Gaussian parameters on normal data"""
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self
    
    def predict_proba(self, X):
        """Compute probability for each sample"""
        # Compute probability for each feature
        probs = np.ones(X.shape[0])
        for j in range(X.shape[1]):
            probs *= norm.pdf(X[:, j], loc=self.mu[j], scale=self.sigma[j])
        return probs
    
    def predict(self, X):
        """Predict: 1 = anomaly, 0 = normal"""
        probs = self.predict_proba(X)
        return (probs < self.epsilon).astype(int)
    
    def decision_function(self, X):
        """Return anomaly scores (lower = more anomalous)"""
        return self.predict_proba(X)

# Example usage
np.random.seed(42)

# Normal data
X_train = np.random.randn(1000, 2) * 2 + [5, 10]

# Test data with some anomalies
X_test_normal = np.random.randn(50, 2) * 2 + [5, 10]
X_test_anomalies = np.random.randn(10, 2) * 0.5 + [15, 3]  # Far from training
X_test = np.vstack([X_test_normal, X_test_anomalies])

# Train detector
detector = AnomalyDetector(epsilon=0.001)
detector.fit(X_train)

# Predict
predictions = detector.predict(X_test)
probabilities = detector.predict_proba(X_test)

print(f"Detected {predictions.sum()} anomalies out of {len(X_test)} samples")
print(f"Last 10 samples (actual anomalies): {predictions[-10:]}")
```

### Choosing Threshold ε

Use a **validation set with labeled anomalies**:

1. Train on normal examples (no labels needed)
2. Validate on dataset with known normal + anomaly labels
3. Try different $\epsilon$ values
4. Choose $\epsilon$ that maximizes F1-score or precision/recall

```python
from sklearn.metrics import f1_score

# Assume X_val and y_val (0=normal, 1=anomaly)
epsilons = np.linspace(0.0001, 0.1, 100)
best_f1 = 0
best_epsilon = 0

for eps in epsilons:
    detector.epsilon = eps
    predictions = detector.predict(X_val)
    f1 = f1_score(y_val, predictions)
    
    if f1 > best_f1:
        best_f1 = f1
        best_epsilon = eps

print(f"Best ε: {best_epsilon:.6f}")
print(f"Best F1: {best_f1:.3f}")
```

### Feature Engineering for Anomaly Detection

**Original features might not be Gaussian!**

**Transformations to make features more Gaussian:**
- $\log(x)$
- $\log(x + c)$
- $\sqrt{x}$
- $x^{1/3}$

```python
# Example: Transform skewed feature
import matplotlib.pyplot as plt

x_skewed = np.random.exponential(scale=2, size=1000)
x_transformed = np.log(x_skewed + 1)

print(f"Original skewness: {((x_skewed - x_skewed.mean())**3).mean():.2f}")
print(f"Transformed skewness: {((x_transformed - x_transformed.mean())**3).mean():.2f}")
```

### 🤔 Think About It
*Why is anomaly detection typically an unsupervised problem, even though we care about a specific outcome (finding anomalies)?*

---

## Assessment Questions

### Question 1: K-Means Cost Function

**Q:** You run K-Means with K=3 on a dataset. After convergence:
- Cluster 1 has 50 points, average distance to centroid: 2.0
- Cluster 2 has 30 points, average distance to centroid: 1.5
- Cluster 3 has 20 points, average distance to centroid: 3.0

Calculate the total cost $J$. If you increase K to 4, will the cost increase or decrease? Why?

<details>
<summary><strong>Click to reveal answer</strong></summary>

**Calculating Cost J:**

Recall: $J = \frac{1}{m} \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2$

This is average squared distance across all points.

For each cluster, contribution = (number of points) × (average squared distance)

**Cluster 1:** $50 \times 2.0^2 = 50 \times 4.0 = 200$  
**Cluster 2:** $30 \times 1.5^2 = 30 \times 2.25 = 67.5$  
**Cluster 3:** $20 \times 3.0^2 = 20 \times 9.0 = 180$

**Total:** $200 + 67.5 + 180 = 447.5$  
**m:** $50 + 30 + 20 = 100$

$$J = \frac{447.5}{100} = 4.475$$

**When K increases from 3 to 4:**

**Cost will DECREASE (or stay same, never increase).**

**Why:**
- With K=4, we have more flexibility to position centroids
- Each point can potentially be assigned to a closer centroid
- K=3 solution is a special case of K=4 (we could just not use the 4th centroid)
- Therefore: $J(K=4) \leq J(K=3)$

**In general:** Cost monotonically decreases as K increases
- $K=1$: Highest cost (all points to one centroid)
- $K=m$: Zero cost (each point is its own cluster)

**Key insight:** This is why elbow method is needed - cost alone doesn't tell us optimal K!

</details>

---

### Question 2: Initialization Impact

**Q:** You run K-Means with K=3 three times with different random initializations on the same dataset:

- **Run 1:** Converges after 12 iterations, final cost = 45.2
- **Run 2:** Converges after 8 iterations, final cost = 38.7
- **Run 3:** Converges after 15 iterations, final cost = 38.7

1. Which result should you use?
2. Why did runs converge to different costs?
3. Why didn't more iterations lead to lower cost in Run 3?

<details>
<summary><strong>Click to reveal answer</strong></summary>

**1. Which result to use:**

**Use Run 2 or Run 3** (they have the same cost of 38.7).

Since both have the same final cost, Run 2 is slightly better because it converged faster (8 vs 15 iterations), though this difference is negligible. The important thing is they both found a better solution than Run 1.

**2. Why different costs:**

**K-Means finds LOCAL minima, not global minimum.**

- Different initializations lead to different optimization paths
- Run 1 got stuck in a worse local minimum (cost = 45.2)
- Runs 2 and 3 found a better local minimum (cost = 38.7)
- With only 3 runs, we can't be sure 38.7 is the global minimum!

**Visual analogy:**

**3. Why more iterations didn't help Run 3:**

**K-Means convergence** guarantees cost decreases or stays same at each iteration. But:

- **Number of iterations ≠ quality of solution**
- More iterations just means it took longer to reach a local minimum
- Run 3 likely started further from its final minimum
- Both reached the same final cost, just different paths

**Example scenario:**
```
Run 2: Started close to good minimum → Fast convergence (8 iter)
Run 3: Started far from good minimum → Slow convergence (15 iter)
       But both end at same local minimum
```

**Best practice:** Run K-Means 10-100 times with different initializations and pick the lowest cost result. Scikit-learn does this automatically with `n_init` parameter!

</details>

---

### Question 3: Choosing K

**Q:** You're clustering customers for marketing. Using elbow method, you get these costs:

| K | Cost |
|---|------|
| 1 | 850 |
| 2 | 520 |
| 3 | 380 |
| 4 | 330 |
| 5 | 310 |
| 6 | 295 |

Your marketing team can handle 3-4 segments. What K would you choose and why? How would your answer change if they could handle any number of segments?

<details>
<summary><strong>Click to reveal answer</strong></summary>

**Given marketing constraint (3-4 segments):**

**Choice: K = 3**

**Reasoning:**

1. **Clear elbow at K=3:**
   - K=1→2: Cost drops 330 (39% reduction)
   - K=2→3: Cost drops 140 (27% reduction)  ← Large drop
   - K=3→4: Cost drops 50 (13% reduction)   ← Diminishing returns
   - K=4→5: Cost drops 20 (6% reduction)
   - K=5→6: Cost drops 15 (5% reduction)

2. **Cost-benefit analysis:**
   - K=3 captures most structure (cost = 380 vs 295 at K=6)
   - Going from K=3 to K=4 only saves 50 cost units
   - Simpler is better for marketing execution

3. **Actionability:**
   - 3 segments easier to manage than 4
   - Clearer customer personas
   - Simpler marketing strategies

**If marketing could handle any number:**

**Still choose K = 3 or possibly K = 4**

**Why not K=6 even though it has lowest cost?**

1. **Diminishing returns:** After K=3, improvements are small
2. **Overfitting risk:** Too many clusters might be fitting noise
3. **Interpretability:** Hard to understand/action 6 segments
4. **Stability:** More clusters = more sensitive to data changes

**Alternative analysis - Marginal cost reduction:**

```
K   Cost   Reduction   % Reduction
1   850      -            -
2   520     330         38.8%
3   380     140         26.9%    ← Elbow
4   330      50         13.2%
5   310      20          6.1%
6   295      15          4.8%
```

The elbow is clearly at K=3 where marginal improvement starts to flatten.

**Silhouette score could help decide between K=3 and K=4:**
```python
for K in [3, 4]:
    kmeans = KMeans(n_clusters=K)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={K}: Silhouette = {score:.3f}")
```

Choose whichever has higher silhouette score.

**Final recommendation:** K=3, with K=4 as backup if analysis shows significantly better cluster separation.

</details>

---

### Question 4: Gaussian Anomaly Detection

**Q:** You're detecting fraudulent transactions. Feature: transaction amount in dollars.

From normal transactions, you estimate:
- $\mu = 50$ dollars
- $\sigma = 15$ dollars

New transactions:
- Transaction A: $45
- Transaction B: $120
- Transaction C: $95

1. Calculate $p(x)$ for each transaction
2. If $\epsilon = 0.01$, which are anomalies?
3. How would you choose $\epsilon$ in practice?

<details>
<summary><strong>Click to reveal answer</strong></summary>

**1. Calculate p(x) for each transaction:**

Using Gaussian PDF: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

Given: $\mu = 50$, $\sigma = 15$, so $\sigma^2 = 225$

**Transaction A: x = 45**

$$p(45) = \frac{1}{\sqrt{2\pi \cdot 225}} \exp\left(-\frac{(45 - 50)^2}{2 \cdot 225}\right)$$

$$= \frac{1}{\sqrt{450\pi}} \exp\left(-\frac{25}{450}\right)$$

$$= \frac{1}{37.57} \exp(-0.0556)$$

$$= 0.0266 \times 0.946 = 0.0251$$

**Transaction B: x = 120**

$$p(120) = \frac{1}{37.57} \exp\left(-\frac{(120 - 50)^2}{450}\right)$$

$$= \frac{1}{37.57} \exp\left(-\frac{4900}{450}\right)$$

$$= 0.0266 \times \exp(-10.89)$$

$$= 0.0266 \times 0.0000186 = 4.95 \times 10^{-7}$$

**Transaction C: x = 95**

$$p(95) = \frac{1}{37.57} \exp\left(-\frac{(95 - 50)^2}{450}\right)$$

$$= \frac{1}{37.57} \exp\left(-\frac{2025}{450}\right)$$

$$= 0.0266 \times \exp(-4.5)$$

$$= 0.0266 \times 0.0111 = 2.95 \times 10^{-4}$$

**Summary:**
- **A ($45)**: $p(x) = 0.0251$ ✓ Normal (close to mean)
- **B ($120)**: $p(x) = 4.95 \times 10^{-7}$ ⚠ Very unusual!
- **C ($95)**: $p(x) = 2.95 \times 10^{-4}$ ⚠ Unusual

**2. With ε = 0.01, which are anomalies?**

Classify as anomaly if $p(x) < \epsilon = 0.01$:

- **Transaction A**: $0.0251 > 0.01$ → **NORMAL**
- **Transaction B**: $4.95 \times 10^{-7} < 0.01$ → **ANOMALY** ✓
- **Transaction C**: $2.95 \times 10^{-4} < 0.01$ → **ANOMALY** ✓

**Interpretation:**
- $45 is within one standard deviation of mean ($50 ± $15) → normal
- $120 is ~4.7σ away → extremely anomalous
- $95 is ~3σ away → anomalous but less extreme

**3. How to choose ε in practice:**

**Step-by-step process:**

**A. Create validation set with labels:**
- Collect examples of known fraud (anomalies)
- Include normal transactions
- Example: 1000 normal + 50 fraud cases

**B. Try multiple ε values:**
```python
epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

for eps in epsilons:
    predictions = (p_val < eps).astype(int)
    
    # Compute metrics
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f"ε={eps:.1e}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
```

**C. Choose based on business objective:**

**If false positives are expensive** (flagging normal as fraud annoys customers):
- Optimize for **precision**
- Use lower ε (only flag extremely unusual)

**If false negatives are expensive** (missing fraud is costly):
- Optimize for **recall**
- Use higher ε (flag more liberally)

**Balanced approach:**
- Optimize **F1-score**
- Trade-off between precision and recall

**D. Example decision:**
```
ε       Precision  Recall   F1     Choice
1e-6    0.95      0.40    0.56   Too conservative
1e-4    0.85      0.75    0.80   Good balance ✓
1e-2    0.60      0.90    0.72   Too many false positives
```

Choose ε = 1e-4

**Pro tip:** Plot precision-recall curve and choose operating point based on business requirements!

</details>

---

### Question 5: Feature Engineering for Anomaly Detection

**Q:** You're building an anomaly detector for server monitoring with two features:
- **CPU usage** (0-100%)
- **Memory usage** (0-100%)

Most servers run at ~60% CPU and ~70% memory. A broken server might have:
- High CPU (95%) + Low memory (20%), OR
- Low CPU (10%) + High memory (95%)

Training a simple Gaussian model, you find it doesn't catch these anomalies well. Why? What feature would you add?

<details>
<summary><strong>Click to reveal answer</strong></summary>

**Why the simple model fails:**

**The independence assumption breaks down!**

Standard anomaly detection with independent features:
$$p(x) = p(x_1) \times p(x_2)$$

This evaluates each feature separately:
- CPU high (95%) → low probability
- Memory low (20%) → low probability
- **Product → very low probability → correctly flagged!** ✓

**BUT ALSO:**
- CPU moderate (60%) → high probability
- Memory moderate (70%) → high probability
- **Product → high probability → NOT flagged**

**The problem:** It doesn't detect the **combination** of values is unusual, even if individual values seem normal.

**Example scenario:**

```
Normal pattern:         Anomaly pattern:
CPU: 60%, Mem: 70%     CPU: 95%, Mem: 20%

Feature view:
p(CPU=95%) = low       ✓ Flagged
p(Mem=20%) = low       ✓ Flagged

But what about:
CPU: 60%, Mem: 95%?    Both values seen before individually!
                       Independent model might miss this!
```

**What feature to add:**

**Add a composite feature that captures the relationship!**

**Option 1: Ratio feature**
$$x_3 = \frac{\text{CPU}}{\text{Memory}}$$

Normal servers: $\frac{60}{70} \approx 0.86$  
Anomaly 1: $\frac{95}{20} = 4.75$ ← Very unusual!  
Anomaly 2: $\frac{10}{95} = 0.11$ ← Very unusual!

**Option 2: Difference feature**
$$x_3 = |\text{CPU} - \text{Memory}|$$

Normal: $|60 - 70| = 10$  
Anomaly 1: $|95 - 20| = 75$ ← Large!  
Anomaly 2: $|10 - 95| = 85$ ← Large!

**Option 3: Distance from normal operating point**
$$x_3 = \sqrt{(\text{CPU} - 60)^2 + (\text{Memory} - 70)^2}$$

Normal: $\sqrt{0^2 + 0^2} = 0$  
Anomaly 1: $\sqrt{35^2 + 50^2} = 61$ ← Large!  
Anomaly 2: $\sqrt{50^2 + 25^2} = 56$ ← Large!

**Implementation:**

```python
# Original features
X_train = np.array([[60, 70], [62, 72], [58, 68], ...])

# Add ratio feature
cpu_mem_ratio = X_train[:, 0] / (X_train[:, 1] + 1e-5)
X_train_enhanced = np.column_stack([X_train, cpu_mem_ratio])

# Now train anomaly detector
detector = AnomalyDetector()
detector.fit(X_train_enhanced)

# Test on anomalies
X_test = np.array([[95, 20], [10, 95], [60, 70]])
X_test_ratio = X_test[:, 0] / (X_test[:, 1] + 1e-5)
X_test_enhanced = np.column_stack([X_test, X_test_ratio])

predictions = detector.predict(X_test_enhanced)
print(predictions)  # [1, 1, 0] - first two flagged as anomalies!
```

**Alternative: Full covariance Gaussian**

Instead of assuming independence, use **multivariate Gaussian with full covariance matrix**:

$$p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$$

This captures correlations between features automatically!

```python
from sklearn.covariance import EmpiricalCovariance

cov = EmpiricalCovariance()
cov.fit(X_train)

# Mahalanobis distance (accounts for covariance)
distances = cov.mahalanobis(X_test)
anomalies = distances > threshold
```

**Key insight:** Feature engineering is crucial for anomaly detection! The model is only as good as the features you give it.

</details>

---

## Congratulations! 🎉

You've mastered unsupervised learning fundamentals! You now understand:
- How K-Means clusters data through iterative optimization
- The importance of initialization and choosing K
- How to detect anomalies using Gaussian distributions

### Key Takeaways

1. **K-Means** finds patterns without labels through iterative refinement
2. **Always run multiple initializations** to avoid bad local minima
3. **Choosing K** requires domain knowledge + elbow method + silhouette scores
4. **Anomaly detection** models normal behavior, flags deviations
5. **Gaussian distributions** provide probabilistic framework for anomaly detection
6. **Feature engineering** is critical for anomaly detection success

### Next Steps
- Apply K-Means to real datasets (customer segmentation, image compression)
- Build anomaly detectors for real problems
- Explore other clustering methods (DBSCAN, Hierarchical)
- Learn dimensionality reduction (PCA, t-SNE)

*"Unsupervised learning finds the hidden structure in data—the patterns we didn't know to look for!"* 🔍
