import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dataset
df=sns.load_dataset("iris")

# 2. Inspect dataset structure
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# 3. Scatter plot(s) to analyze relationships
plt.figure(figsize=(6,4))
plt.scatter(df['sepal_length'], df['sepal_width'])
plt.title('Scatter: Sepal length vs Sepal width')
plt.xlabel('sepal_length (cm)')
plt.ylabel('sepal_width (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Scatter: petal_length vs petal_width
plt.figure(figsize=(6,4))
plt.scatter(df['petal_length'], df['petal_width'])
plt.title('Scatter: Petal length vs Petal width')
plt.xlabel('petal_length (cm)')
plt.ylabel('petal_width (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Histogram to examine distribution
plt.figure(figsize=(6,4))
plt.hist(df['sepal_length'], bins=10)
plt.title('Histogram: Sepal length distribution')
plt.xlabel('sepal_length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Box plot to detect outliers and spread
numeric_cols = df.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(8,5))
plt.boxplot([df[col] for col in numeric_cols], labels=numeric_cols)
plt.title('Boxplot: Numeric feature distributions')
plt.ylabel('Value (cm)')
plt.grid(True)
plt.tight_layout()
plt.show()
