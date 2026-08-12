# All-in-One Data Science Workflow 🚀
# Covers: Data Loading → Cleaning → EDA → ML → Evaluation → Save Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# 1️⃣ Load Data
df = pd.read_csv("data.csv")   # replace with your dataset
print("First 5 rows:\n", df.head())

# 2️⃣ Data Cleaning
df.dropna(inplace=True)        # remove missing values
df = df.drop_duplicates()      # remove duplicates

# 3️⃣ Exploratory Data Analysis (EDA)
print("\nBasic Stats:\n", df.describe())
sns.pairplot(df, diag_kind="kde")
plt.show()

# 4️⃣ Feature Engineering
X = df.drop("target", axis=1)  # features
y = df["target"]               # target column

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 7️⃣ Evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Save Model
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")
