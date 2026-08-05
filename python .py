# 🔹 Python All-in-One Data Science Workflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Create sample dataset
df = pd.DataFrame({
    "Name": ["A","B","C","D","E"],
    "Marks": [85, 90, np.nan, 70, 95],
    "Age": [20, 21, 22, 23, 24],
    "Attendance": [90, 85, 88, 92, 80],
    "Target": [88, 92, 75, 70, 96]   # Example target variable
})

# 2️⃣ Handle missing values
df["Marks"].fillna(df["Marks"].mean(), inplace=True)

# 3️⃣ Quick EDA
print("Nulls:\n", df.isnull().sum())
print("Mean Marks:", df["Marks"].mean())
print("Median Marks:", df["Marks"].median())
print(df.describe())

# 4️⃣ Visualization
sns.pairplot(df, diag_kind="kde")
plt.show()

# 5️⃣ Train simple ML model
X = df[["Marks","Age","Attendance"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
