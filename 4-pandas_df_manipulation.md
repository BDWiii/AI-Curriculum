# Pandas DataFrame Manipulation for Machine Learning

## Table of Contents
1. [Introduction to Pandas](#introduction-to-pandas)
2. [Loading DataFrames](#loading-dataframes)
3. [Why Pandas for ML/DL](#why-pandas-for-mldl)
4. [Data Inspection & Exploration](#data-inspection--exploration)
5. [Data Cleaning Techniques](#data-cleaning-techniques)
6. [Normalization & Scaling](#normalization--scaling)
7. [Encoding Categorical Data](#encoding-categorical-data)
8. [Time Series Handling](#time-series-handling)
9. [Feature Engineering](#feature-engineering)
10. [Practice Exercises](#practice-exercises)

---

## Introduction to Pandas

**Pandas** is a powerful Python library for data manipulation and analysis. It provides two primary data structures:

- **Series**: 1-dimensional labeled array
- **DataFrame**: 2-dimensional labeled data structure (like a table)

### Why Pandas?

- Fast and efficient data manipulation
- Handles missing data elegantly
- Powerful grouping and aggregation
- Time series functionality
- Easy integration with ML libraries (scikit-learn, TensorFlow, PyTorch)

### Basic DataFrame Creation

```python
import pandas as pd
import numpy as np

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 75000, 55000]
}
df = pd.DataFrame(data)
print(df)
```

**Output:**
```
      name  age  salary
0    Alice   25   50000
1      Bob   30   60000
2  Charlie   35   75000
3    David   28   55000
```

---

## Loading DataFrames

### From CSV Files

```python
# Basic loading
df = pd.read_csv('data.csv')

# With specific parameters
df = pd.read_csv(
    'data.csv',
    sep=',',              # Delimiter
    header=0,             # Row number to use as column names
    index_col=0,          # Column to use as row labels
    na_values=['NA', '?'] # Additional strings to recognize as NA
)
```

### From Excel Files

```python
# Single sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Multiple sheets
excel_file = pd.ExcelFile('data.xlsx')
df1 = excel_file.parse('Sheet1')
df2 = excel_file.parse('Sheet2')
```

### From JSON Files

```python
# From JSON file
df = pd.read_json('data.json')

# From JSON string
json_str = '{"name": ["Alice", "Bob"], "age": [25, 30]}'
df = pd.read_json(json_str)
```

### From SQL Databases

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')

# Read SQL query into DataFrame
df = pd.read_sql_query("SELECT * FROM users", conn)

# Or read entire table
df = pd.read_sql_table('users', conn)
```

### From URLs

```python
url = 'https://example.com/data.csv'
df = pd.read_csv(url)
```

---

## Why Pandas for ML/DL?

### 1. **Data Preprocessing Pipeline**

Machine learning models require clean, structured data. Pandas provides:

- **Missing value handling**: Critical for model performance
- **Feature scaling**: Ensures features contribute equally
- **Encoding**: Converts categorical data to numerical format
- **Feature engineering**: Creates new meaningful features

### 2. **Integration with ML Libraries**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Pandas DataFrame → NumPy arrays (seamless)
X = df[['feature1', 'feature2', 'feature3']].values
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
```

### 3. **Exploratory Data Analysis (EDA)**

Understanding data before modeling:

```python
# Statistical summary
print(df.describe())

# Correlation matrix
correlation = df.corr()

# Identify relationships between features
print(correlation['target'].sort_values(ascending=False))
```

### 4. **Memory Efficiency**

Pandas optimizes memory usage, crucial for large datasets:

```python
# Check memory usage
print(df.memory_usage(deep=True))

# Optimize data types
df['age'] = df['age'].astype('int8')  # Instead of int64
df['category'] = df['category'].astype('category')  # Categorical dtype
```

---

## Data Inspection & Exploration

### Basic Information

```python
# First/last rows
print(df.head())      # First 5 rows
print(df.tail(10))    # Last 10 rows

# Shape and size
print(df.shape)       # (rows, columns)
print(df.size)        # Total elements

# Column information
print(df.info())      # Data types, non-null counts
print(df.columns)     # Column names
print(df.dtypes)      # Data types
```

### Statistical Summary

```python
# Numerical columns
print(df.describe())

# Include all columns
print(df.describe(include='all'))

# Specific statistics
print(df['age'].mean())
print(df['salary'].median())
print(df['score'].std())
print(df['category'].value_counts())
```

### Missing Data Detection

```python
# Count missing values
print(df.isnull().sum())

# Percentage of missing values
print((df.isnull().sum() / len(df)) * 100)

# Visualize missing data pattern
import missingno as msno
msno.matrix(df)
```

---

## Data Cleaning Techniques

### 1. Handling Missing Values

#### **Why it matters**: Missing data can bias models or cause errors during training.

#### **Detection**

```python
# Boolean mask of missing values
missing_mask = df.isnull()

# Rows with any missing values
rows_with_missing = df[df.isnull().any(axis=1)]
```

#### **Strategy 1: Removal**

```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows where specific column is missing
df_clean = df.dropna(subset=['important_column'])

# Drop columns with >50% missing values
threshold = len(df) * 0.5
df_clean = df.dropna(thresh=threshold, axis=1)
```

#### **Strategy 2: Imputation**

```python
# Fill with constant
df['age'].fillna(0, inplace=True)

# Fill with mean (numerical)
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Fill with median (robust to outliers)
df['score'].fillna(df['score'].median(), inplace=True)

# Fill with mode (categorical)
df['category'].fillna(df['category'].mode()[0], inplace=True)

```

### 2. Handling Duplicates

```python
# Check for duplicates
print(df.duplicated().sum())

# View duplicate rows
duplicates = df[df.duplicated()]

# Remove duplicates (keep first occurrence)
df_clean = df.drop_duplicates()

# Remove duplicates based on specific columns
df_clean = df.drop_duplicates(subset=['user_id', 'date'])

# Keep last occurrence
df_clean = df.drop_duplicates(keep='last')
```

### 3. Handling Outliers

#### **Why it matters**: Outliers can skew model training and predictions.

#### **Detection using IQR (Interquartile Range)**

```python
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
df_clean = df[(df['salary'] >= lower_bound) & (df['salary'] <= upper_bound)]
```


### 4. Data Type Conversion

```python
# Convert to numeric (errors become NaN)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Convert to categorical
df['category'] = df['category'].astype('category')

# String operations
df['name'] = df['name'].str.lower()
df['name'] = df['name'].str.strip()
```

---

## Normalization & Scaling

### Why Normalize?

**Problem**: Features with different scales can dominate the learning process.

**Example**: 
- Age: 20-80 (range: 60)
- Salary: 30,000-150,000 (range: 120,000)

Without scaling, salary would have disproportionate influence on distance-based algorithms (KNN, SVM, Neural Networks).

### 1. Min-Max Normalization (Scaling to [0, 1])

**Formula**: 

$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**When to use**: When you need bounded values (0-1) and data doesn't have extreme outliers.

```python
from sklearn.preprocessing import MinMaxScaler

# Create scaler
scaler = MinMaxScaler()

# Fit and transform
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Manual implementation
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df['age_normalized'] = min_max_normalize(df['age'])
```

**Example**:
```python
# Original data
ages = [25, 30, 35, 40, 45]

# After Min-Max scaling
# (25-25)/(45-25) = 0.0
# (30-25)/(45-25) = 0.25
# (35-25)/(45-25) = 0.5
# (40-25)/(45-25) = 0.75
# (45-25)/(45-25) = 1.0
```

### 2. Standardization (Z-score Normalization)

**Formula**: 

$$X_{standardized} = \frac{X - \mu}{\sigma}$$

Where:
- $\mu$ = mean
- $\sigma$ = standard deviation

**When to use**: When features follow Gaussian distribution or when outliers are present.

```python
from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit and transform
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])

# Manual implementation
def standardize(series):
    return (series - series.mean()) / series.std()

df['age_standardized'] = standardize(df['age'])
```

**Properties**:
- Mean = 0
- Standard deviation = 1
- Values typically range from -3 to +3

### 3. Robust Scaling

**Formula**: 

$$X_{robust} = \frac{X - Q_{median}}{Q_{75} - Q_{25}}$$

**When to use**: When data contains many outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])
```

### 4. Log Transformation

**Formula**: 

$$X_{log} = \log(X + 1)$$

**When to use**: For right-skewed distributions (e.g., income, population).

```python
# Natural log
df['salary_log'] = np.log1p(df['salary'])  # log1p = log(1 + x)

# Log base 10
df['salary_log10'] = np.log10(df['salary'] + 1)
```

### Comparison Example

```python
import pandas as pd
import numpy as np

# Sample data
data = {'salary': [30000, 50000, 75000, 100000, 150000]}
df = pd.DataFrame(data)

# Min-Max
df['minmax'] = (df['salary'] - df['salary'].min()) / (df['salary'].max() - df['salary'].min())

# Standardization
df['standard'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()

# Log
df['log'] = np.log1p(df['salary'])

print(df)
```

---

## Encoding Categorical Data

### Why Encode?

Machine learning models require numerical input. Categorical variables (text labels) must be converted to numbers.

### 1. Label Encoding

**Use case**: Ordinal data (ordered categories)

**Example**: Education level (High School < Bachelor < Master < PhD)

```python
from sklearn.preprocessing import LabelEncoder

# Create encoder
le = LabelEncoder()

# Fit and transform
df['education_encoded'] = le.fit_transform(df['education'])

# Manual mapping
education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_map)
```

**Example**:
```
Original: ['High School', 'Bachelor', 'Master', 'PhD']
Encoded:  [0, 1, 2, 3]
```

> [!WARNING]
> **Don't use Label Encoding for nominal data** (categories without order like colors, countries) because it implies ordering where none exists!

### 2. One-Hot Encoding

**Use case**: Nominal data (unordered categories)

**Example**: Color (Red, Blue, Green) - no inherent order

**Formula**: Create binary column for each category

```python
# Using pandas
df_encoded = pd.get_dummies(df, columns=['color'])

# Using sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['color']])

# Create DataFrame with proper column names
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['color'])
)
```

**Example**:
```
Original DataFrame:
   name     color
0  Item1    Red
1  Item2    Blue
2  Item3    Green
3  Item4    Red

After One-Hot Encoding:
   name     color_Blue  color_Green  color_Red
0  Item1    0           0            1
1  Item2    1           0            0
2  Item3    0           1            0
3  Item4    0           0            1
```

**Drop first to avoid multicollinearity**:
```python
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
```

### 3. Binary Encoding

**Use case**: High cardinality categorical variables (many unique values)

**Advantage**: More compact than one-hot encoding

```python
import category_encoders as ce

# Binary encoder
encoder = ce.BinaryEncoder(cols=['city'])
df_encoded = encoder.fit_transform(df)
```

**Example**:
```
Original: ['New York', 'London', 'Paris', 'Tokyo', 'Berlin']
Binary encoding uses log2(5) ≈ 3 columns instead of 5 for one-hot
```

### 4. Frequency Encoding

**Use case**: When category frequency is informative

```python
# Calculate frequency
freq_map = df['category'].value_counts(normalize=True).to_dict()

# Apply encoding
df['category_freq'] = df['category'].map(freq_map)
```

### 5. Target Encoding (Mean Encoding)

**Use case**: When category relates to target variable

**Formula**: 

$$\text{Encoded}(c) = \frac{\sum_{i \in c} y_i}{|c|}$$

Where $c$ is the category and $y_i$ are target values.

```python
# Calculate mean target per category
target_mean = df.groupby('category')['target'].mean()

# Map to DataFrame
df['category_encoded'] = df['category'].map(target_mean)
```

> [!CAUTION]
> **Target encoding can cause data leakage!** Always use cross-validation or separate train/test encoding.

### Encoding Strategy Decision Tree

```
Is the data ordinal (has natural order)?
├─ YES → Use Label Encoding
└─ NO → Is it nominal?
    ├─ Few categories (<10)?
    │   └─ Use One-Hot Encoding
    └─ Many categories (>10)?
        ├─ Is frequency meaningful?
        │   └─ Use Frequency Encoding
        └─ Is target relationship strong?
            └─ Use Target Encoding (with caution)
```

---

## Time Series Handling

### Why Time Matters in ML

Time-based features capture:
- **Seasonality**: Patterns that repeat (daily, weekly, yearly)
- **Trends**: Long-term increases/decreases
- **Cyclical patterns**: Economic cycles, weather patterns

### 1. Converting to Datetime

```python
# From string
df['date'] = pd.to_datetime(df['date'])

# With specific format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Handle errors
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# From components
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
```

### 2. Extracting Time Features

```python
# Set datetime as index
df.set_index('date', inplace=True)

# Extract components
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
df['day_name'] = df.index.day_name()
df['quarter'] = df.index.quarter
df['week_of_year'] = df.index.isocalendar().week
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# Time components
df['hour'] = df.index.hour
df['minute'] = df.index.minute
```

### 3. Cyclical Encoding

**Why?** December (12) and January (1) are close in time but far numerically.

**Solution**: Sine-Cosine transformation

**Formula**:

$$\text{sin\_feature} = \sin\left(\frac{2\pi \times \text{value}}{\text{max\_value}}\right)$$

$$\text{cos\_feature} = \cos\left(\frac{2\pi \times \text{value}}{\text{max\_value}}\right)$$

```python
# Month encoding (1-12)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of week encoding (0-6)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Hour encoding (0-23)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

**Visualization**:
```
Month 12 (December): sin ≈ 0, cos ≈ 1
Month 1 (January):   sin ≈ 0.5, cos ≈ 0.87
→ Close in transformed space!
```

### 4. Time-based Aggregations

```python
# Resample to different frequencies
daily_avg = df.resample('D').mean()      # Daily average
weekly_sum = df.resample('W').sum()      # Weekly sum
monthly_max = df.resample('M').max()     # Monthly maximum

# Rolling windows
df['rolling_mean_7d'] = df['value'].rolling(window=7).mean()
df['rolling_std_30d'] = df['value'].rolling(window=30).std()

# Expanding windows (cumulative)
df['cumulative_sum'] = df['value'].expanding().sum()
```

### 5. Lag Features

**Use case**: Past values predict future (e.g., yesterday's sales predict today's)

```python
# Create lag features
df['value_lag1'] = df['value'].shift(1)   # Previous day
df['value_lag7'] = df['value'].shift(7)   # Same day last week
df['value_lag30'] = df['value'].shift(30) # Same day last month

# Lead features (future values - careful with data leakage!)
df['value_lead1'] = df['value'].shift(-1)
```

### 6. Time Differences

```python
# Time since event
df['days_since_start'] = (df.index - df.index.min()).days

# Time between events
df['time_diff'] = df['date'].diff()

# Business days
from pandas.tseries.offsets import BDay
df['business_days'] = df['date'].apply(lambda x: len(pd.bdate_range(start_date, x)))
```

### Complete Time Series Example

```python
import pandas as pd
import numpy as np

# Sample data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
df = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 1000, 365)
})

# Set index
df.set_index('date', inplace=True)

# Extract features
df['year'] = df.index.year
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Rolling features
df['sales_ma7'] = df['sales'].rolling(window=7).mean()
df['sales_ma30'] = df['sales'].rolling(window=30).mean()

# Lag features
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)

print(df.head(10))
```

---

## Feature Engineering

### What is Feature Engineering?

Creating new features from existing data to improve model performance.

### 1. Mathematical Transformations

```python
# Polynomial features
df['age_squared'] = df['age'] ** 2
df['age_cubed'] = df['age'] ** 3

# Interaction features
df['age_income_interaction'] = df['age'] * df['income']

# Ratios
df['income_per_age'] = df['income'] / df['age']
df['debt_to_income'] = df['debt'] / df['income']

# Using sklearn
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'income']])
```

### 2. Binning (Discretization)

**Why?** Convert continuous variables to categorical for non-linear patterns.

```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'])

# Custom bins
bins = [0, 18, 35, 50, 65, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels)

# Equal-frequency bins (quantiles)
df['income_quartile'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 3. Aggregation Features

```python
# Group statistics
df['avg_income_by_city'] = df.groupby('city')['income'].transform('mean')
df['max_age_by_dept'] = df.groupby('department')['age'].transform('max')
df['count_by_category'] = df.groupby('category')['id'].transform('count')

# Multiple aggregations
agg_features = df.groupby('user_id').agg({
    'purchase_amount': ['sum', 'mean', 'std', 'count'],
    'days_since_last': 'min'
})
```

### 4. Text Features

```python
# String length
df['name_length'] = df['name'].str.len()

# Word count
df['description_words'] = df['description'].str.split().str.len()

# Contains keyword
df['has_premium'] = df['description'].str.contains('premium', case=False).astype(int)

# Extract patterns
df['phone_area_code'] = df['phone'].str.extract(r'(\d{3})-')
```

### 5. Domain-Specific Features

```python
# Example: E-commerce
df['is_first_purchase'] = (df['purchase_count'] == 1).astype(int)
df['avg_order_value'] = df['total_spent'] / df['order_count']
df['days_since_registration'] = (pd.Timestamp.now() - df['registration_date']).dt.days

# Example: Finance
df['credit_utilization'] = df['credit_used'] / df['credit_limit']
df['savings_rate'] = df['savings'] / df['income']
```

---

## Practice Exercises

### Exercise 1: Data Loading and Inspection

**Task**: Load a dataset and perform initial exploration.

```python
# Create sample dataset
import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'customer_id': range(1, 101),
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 150000, 100),
    'credit_score': np.random.randint(300, 850, 100),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
    'purchased': np.random.choice([0, 1], 100)
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('customer_data.csv', index=False)
```

**Questions**:
1. Load the dataset and display the first 10 rows
2. What is the shape of the dataset?
3. What are the data types of each column?
4. Calculate the mean, median, and standard deviation of 'income'
5. How many unique cities are there?
6. What percentage of customers made a purchase?

<details>
<summary><b>Solution</b></summary>

```python
# 1. Load and display
df = pd.read_csv('customer_data.csv')
print(df.head(10))

# 2. Shape
print(f"Shape: {df.shape}")  # (100, 6)

# 3. Data types
print(df.dtypes)

# 4. Income statistics
print(f"Mean: {df['income'].mean():.2f}")
print(f"Median: {df['income'].median():.2f}")
print(f"Std: {df['income'].std():.2f}")

# 5. Unique cities
print(f"Unique cities: {df['city'].nunique()}")

# 6. Purchase percentage
purchase_rate = (df['purchased'].sum() / len(df)) * 100
print(f"Purchase rate: {purchase_rate:.2f}%")
```
</details>

---

### Exercise 2: Handling Missing Data

**Task**: Create a dataset with missing values and clean it.

```python
# Create dataset with missing values
import pandas as pd
import numpy as np

np.random.seed(42)
data = {
    'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Frank', 'Grace', 'Henry'],
    'age': [25, np.nan, 35, 28, np.nan, 45, 32, 29],
    'salary': [50000, 60000, np.nan, 55000, 70000, np.nan, 62000, 58000],
    'department': ['HR', 'IT', 'IT', 'HR', None, 'Finance', 'IT', 'HR']
}
df = pd.DataFrame(data)
```

**Questions**:
1. Identify which columns have missing values and count them
2. Calculate the percentage of missing values for each column
3. Fill missing 'age' values with the median age
4. Fill missing 'salary' values with the mean salary
5. Fill missing 'department' values with the mode
6. Drop rows where 'name' is missing
7. Verify that no missing values remain

<details>
<summary><b>Solution</b></summary>

```python
# 1. Count missing values
print(df.isnull().sum())

# 2. Percentage missing
print((df.isnull().sum() / len(df)) * 100)

# 3. Fill age with median
df['age'].fillna(df['age'].median(), inplace=True)

# 4. Fill salary with mean
df['salary'].fillna(df['salary'].mean(), inplace=True)

# 5. Fill department with mode
df['department'].fillna(df['department'].mode()[0], inplace=True)

# 6. Drop rows with missing name
df.dropna(subset=['name'], inplace=True)

# 7. Verify
print(df.isnull().sum())  # Should be all zeros
print(df)
```
</details>

---

### Exercise 3: Normalization Challenge

**Task**: Apply different normalization techniques and compare results.

```python
import pandas as pd
import numpy as np

# Create dataset
data = {
    'height_cm': [150, 160, 170, 180, 190],
    'weight_kg': [50, 60, 70, 80, 90],
    'salary_usd': [30000, 50000, 70000, 90000, 110000]
}
df = pd.DataFrame(data)
```

**Questions**:
1. Apply Min-Max normalization to all columns
2. Apply Standardization (Z-score) to all columns
3. Apply Log transformation to 'salary_usd'
4. Which normalization method would you use for a neural network? Why?
5. Create a function that applies Min-Max normalization and can reverse it

<details>
<summary><b>Solution</b></summary>

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Min-Max normalization
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler_minmax.fit_transform(df),
    columns=[f'{col}_minmax' for col in df.columns]
)
print("Min-Max Normalized:\n", df_minmax)

# 2. Standardization
scaler_standard = StandardScaler()
df_standard = pd.DataFrame(
    scaler_standard.fit_transform(df),
    columns=[f'{col}_standard' for col in df.columns]
)
print("\nStandardized:\n", df_standard)

# 3. Log transformation
df['salary_log'] = np.log1p(df['salary_usd'])
print("\nLog transformed salary:\n", df['salary_log'])

# 4. Answer: Standardization (Z-score) is often preferred for neural networks
# because it centers data around 0 with unit variance, which helps with
# gradient descent convergence and prevents vanishing/exploding gradients.

# 5. Reversible Min-Max normalization
class ReversibleMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None
    
    def fit_transform(self, data):
        self.min_ = data.min()
        self.max_ = data.max()
        return (data - self.min_) / (self.max_ - self.min_)
    
    def inverse_transform(self, normalized_data):
        return normalized_data * (self.max_ - self.min_) + self.min_

# Test
scaler = ReversibleMinMaxScaler()
normalized = scaler.fit_transform(df['height_cm'])
original = scaler.inverse_transform(normalized)
print("\nOriginal heights:", df['height_cm'].values)
print("Normalized:", normalized.values)
print("Reversed:", original.values)
```
</details>

---

### Exercise 4: Encoding Categorical Data

**Task**: Practice different encoding techniques.

```python
import pandas as pd

# Create dataset
data = {
    'product_id': [1, 2, 3, 4, 5, 6],
    'category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Clothing', 'Food'],
    'size': ['Small', 'Medium', 'Large', 'Medium', 'Small', 'Large'],
    'rating': ['Good', 'Excellent', 'Good', 'Poor', 'Excellent', 'Good'],
    'price': [100, 50, 150, 20, 60, 25]
}
df = pd.DataFrame(data)
```

**Questions**:
1. Apply One-Hot Encoding to 'category'
2. Apply Label Encoding to 'rating' (Poor=0, Good=1, Excellent=2)
3. Apply Frequency Encoding to 'size'
4. Why shouldn't you use Label Encoding for 'category'?
5. Create a custom ordinal encoding for 'size' (Small=1, Medium=2, Large=3)

<details>
<summary><b>Solution</b></summary>

```python
from sklearn.preprocessing import LabelEncoder

# 1. One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
print("One-Hot Encoded:\n", df_encoded)

# 2. Label Encoding for rating (ordinal)
rating_map = {'Poor': 0, 'Good': 1, 'Excellent': 2}
df['rating_encoded'] = df['rating'].map(rating_map)
print("\nRating encoded:\n", df[['rating', 'rating_encoded']])

# 3. Frequency Encoding
freq_map = df['size'].value_counts(normalize=True).to_dict()
df['size_freq'] = df['size'].map(freq_map)
print("\nSize frequency encoded:\n", df[['size', 'size_freq']])

# 4. Answer: Category is nominal (no inherent order). Label encoding
# would imply Electronics < Clothing < Food, which is meaningless.
# This could mislead the model into learning false relationships.

# 5. Custom ordinal encoding for size
size_map = {'Small': 1, 'Medium': 2, 'Large': 3}
df['size_encoded'] = df['size'].map(size_map)
print("\nSize ordinal encoded:\n", df[['size', 'size_encoded']])
```
</details>

---

### Exercise 5: Time Series Feature Engineering

**Task**: Extract and engineer time-based features.

```python
import pandas as pd
import numpy as np

# Create time series dataset
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {
    'date': dates,
    'sales': np.random.randint(100, 500, 100) + np.sin(np.arange(100) * 2 * np.pi / 7) * 50
}
df = pd.DataFrame(data)
```

**Questions**:
1. Extract year, month, day, and day_of_week from 'date'
2. Create a binary feature 'is_weekend'
3. Apply cyclical encoding to 'day_of_week' (sine and cosine)
4. Create a 7-day rolling average of 'sales'
5. Create lag features for 'sales' (lag 1 and lag 7)
6. Calculate the percentage change in sales from the previous day

<details>
<summary><b>Solution</b></summary>

```python
import pandas as pd
import numpy as np

# Set date as index
df.set_index('date', inplace=True)

# 1. Extract date components
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.dayofweek

# 2. Binary weekend feature
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# 3. Cyclical encoding
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 4. Rolling average
df['sales_ma7'] = df['sales'].rolling(window=7).mean()

# 5. Lag features
df['sales_lag1'] = df['sales'].shift(1)
df['sales_lag7'] = df['sales'].shift(7)

# 6. Percentage change
df['sales_pct_change'] = df['sales'].pct_change() * 100

print(df.head(10))
print("\nColumns:", df.columns.tolist())
```
</details>

---

### Exercise 6: Advanced Data Cleaning Pipeline

**Task**: Build a complete preprocessing pipeline.

```python
import pandas as pd
import numpy as np

# Create messy dataset
np.random.seed(42)
data = {
    'customer_id': [1, 2, 3, 4, 5, 5, 6, 7, 8, 9],  # Duplicate
    'age': [25, np.nan, 35, 150, 28, 28, -5, 45, 32, 29],  # Outliers
    'income': [50000, 60000, np.nan, 55000, 70000, 70000, 62000, 58000, 1000000, 52000],
    'city': ['NYC', 'LA', 'chicago', 'NYC', 'la', 'LA', 'Chicago', 'NYC', 'LA', 'Chicago'],
    'join_date': ['2023-01-15', '2023-02-20', 'invalid', '2023-03-10', '2023-04-05', 
                  '2023-04-05', '2023-05-12', '2023-06-18', '2023-07-22', '2023-08-30']
}
df = pd.DataFrame(data)
```

**Your Task**: Create a function `clean_data(df)` that:
1. Removes duplicate rows
2. Handles invalid ages (must be 0-120)
3. Removes income outliers using IQR method
4. Standardizes city names (proper capitalization)
5. Converts join_date to datetime (invalid → NaN)
6. Fills missing ages with median
7. Returns cleaned DataFrame

**Think about**:
- What order should these operations be performed?
- How do you handle edge cases?
- Should you modify the original DataFrame or create a copy?

<details>
<summary><b>Solution</b></summary>

```python
def clean_data(df):
    """
    Comprehensive data cleaning pipeline.
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    # Create copy to avoid modifying original
    df_clean = df.copy()
    
    # 1. Remove duplicates (keep first occurrence)
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {len(df) - len(df_clean)} duplicate rows")
    
    # 2. Handle invalid ages
    df_clean.loc[(df_clean['age'] < 0) | (df_clean['age'] > 120), 'age'] = np.nan
    
    # 3. Remove income outliers using IQR

    ![OUTLIERS IMAGE](images/outliers.png)


    Q1 = df_clean['income'].quantile(0.25)
    Q3 = df_clean['income'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df_clean['income'] < lower_bound) | (df_clean['income'] > upper_bound)
    print(f"Removed {outlier_mask.sum()} income outliers")
    df_clean = df_clean[~outlier_mask]
    
    # 4. Standardize city names
    df_clean['city'] = df_clean['city'].str.title()
    
    # 5. Convert join_date to datetime
    df_clean['join_date'] = pd.to_datetime(df_clean['join_date'], errors='coerce')
    
    # 6. Fill missing ages with median
    median_age = df_clean['age'].median()
    df_clean['age'].fillna(median_age, inplace=True)
    
    # Reset index
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

# Test the function
df_cleaned = clean_data(df)
print("\nCleaned Data:")
print(df_cleaned)
print("\nData Info:")
print(df_cleaned.info())
print("\nMissing Values:")
print(df_cleaned.isnull().sum())
```

**Key Insights**:
- Order matters: Remove duplicates first, then handle outliers
- Use `.copy()` to avoid SettingWithCopyWarning
- IQR method is robust for outlier detection
- `errors='coerce'` handles invalid dates gracefully
- Always verify results with `.info()` and `.isnull().sum()`
</details>

---

### Exercise 7: Feature Engineering Challenge

**Task**: Create meaningful features for a machine learning model.

```python
import pandas as pd
import numpy as np

# E-commerce transaction data
np.random.seed(42)
data = {
    'user_id': np.repeat(range(1, 21), 5),
    'transaction_date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'amount': np.random.uniform(10, 500, 100),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
    'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal'], 100)
}
df = pd.DataFrame(data)
```

**Create the following features**:
1. Total spending per user
2. Average transaction amount per user
3. Number of transactions per user
4. Days since first transaction (for each user)
5. Most frequent category per user
6. One-hot encoding for payment_method
7. Cyclical encoding for transaction month

**Bonus**: Create a feature that indicates if a user's current transaction is above their average.

<details>
<summary><b>Solution</b></summary>

```python
import pandas as pd
import numpy as np

# 1. Total spending per user
df['total_spending'] = df.groupby('user_id')['amount'].transform('sum')

# 2. Average transaction amount per user
df['avg_transaction'] = df.groupby('user_id')['amount'].transform('mean')

# 3. Number of transactions per user
df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

# 4. Days since first transaction
df['first_transaction'] = df.groupby('user_id')['transaction_date'].transform('min')
df['days_since_first'] = (df['transaction_date'] - df['first_transaction']).dt.days
df.drop('first_transaction', axis=1, inplace=True)

# 5. Most frequent category per user
most_freq_category = df.groupby('user_id')['category'].agg(lambda x: x.mode()[0])
df['most_freq_category'] = df['user_id'].map(most_freq_category)

# 6. One-hot encoding for payment_method
df = pd.get_dummies(df, columns=['payment_method'], prefix='payment')

# 7. Cyclical encoding for month
df['month'] = df['transaction_date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Bonus: Above average indicator
df['above_avg'] = (df['amount'] > df['avg_transaction']).astype(int)

print(df.head(10))
print("\nFeature columns:", df.columns.tolist())
print("\nSample user statistics:")
print(df[df['user_id'] == 1][['user_id', 'amount', 'total_spending', 
                                'avg_transaction', 'transaction_count', 
                                'above_avg']].head())
```
</details>

---

## Summary Checklist

Before using data for ML/DL, ensure you've completed:

- [ ] **Data Loading**: Successfully imported data from various sources
- [ ] **Inspection**: Understood shape, types, and statistical properties
- [ ] **Missing Values**: Detected and handled appropriately
- [ ] **Duplicates**: Identified and removed
- [ ] **Outliers**: Detected and treated (removed/capped)
- [ ] **Normalization**: Applied appropriate scaling technique
- [ ] **Encoding**: Converted categorical variables to numerical
- [ ] **Time Features**: Extracted and engineered temporal features
- [ ] **Feature Engineering**: Created domain-specific features
- [ ] **Validation**: Verified data quality and no data leakage

---

## Additional Resources

### Recommended Libraries
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Preprocessing and ML
- **category_encoders**: Advanced encoding techniques
- **missingno**: Missing data visualization

### Best Practices
1. **Always split data before preprocessing** to avoid data leakage
2. **Document your preprocessing steps** for reproducibility
3. **Validate assumptions** (e.g., normality for standardization)
4. **Monitor for data drift** in production
5. **Version your preprocessing pipelines**

### Common Pitfalls to Avoid
- ❌ Fitting scalers on entire dataset (including test set)
- ❌ Using label encoding for nominal categories
- ❌ Dropping too much data without investigation
- ❌ Ignoring domain knowledge in feature engineering
- ❌ Not handling time-based features properly for time series

---

**Congratulations!** You now have a solid foundation in pandas DataFrame manipulation for machine learning. Practice these techniques on real datasets to master them.
