# 🔹 Step 1: Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 2: Load dataset (CSV example)
df = pd.read_csv("data.csv")

# 🔹 Step 3: Quick overview
print(df.head())        # first 5 rows
print(df.info())        # column info
print(df.describe())    # summary stats

# 🔹 Step 4: Data cleaning
df = df.dropna()        # remove missing values
df['column'] = df['column'].astype(float)  # convert type

# 🔹 Step 5: Exploratory Data Analysis (EDA)
sns.histplot(df['column'], bins=20, kde=True)
plt.show()

sns.boxplot(x=df['column'])
plt.show()

# 🔹 Step 6: Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# 🔹 Step 7: Simple ML example (Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['feature1', 'feature2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Score:", model.score(X_test, y_test))
