# 📊 Python Data Science Template
# Author: Bharti
# Purpose: Quick start for data analysis & ML

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset (CSV example)
data = pd.read_csv("data.csv")

# 3. Explore data
print(data.head())        # first 5 rows
print(data.info())        # column info
print(data.describe())    # summary stats

# 4. Handle missing values
data = data.dropna()      # simple removal
# OR: data.fillna(0, inplace=True)

# 5. Data visualization
sns.countplot(x="Category", data=data)
plt.show()

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.show()

# 6. Feature engineering
data["NewFeature"] = data["Column1"] * data["Column2"]

# 7. Train/Test split
from sklearn.model_selection import train_test_split
X = data.drop("Target", axis=1)
y = data["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Build a simple ML model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. Evaluate model
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
