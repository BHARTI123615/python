
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 2. Load Dataset
df = pd.read_csv("data.csv")   # अपना dataset path दें
print("Dataset Shape:", df.shape)
print(df.head())

# 3. Data Cleaning
df.dropna(inplace=True)        # Missing values हटाना
df = df.drop_duplicates()      # Duplicate हटाना

# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Exploratory Data Analysis (EDA)
print(df.describe())           # Summary stats
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Feature & Target Split
X = df.drop("target", axis=1)  # 'target' को अपने dataset के अनुसार बदलें
y = df["target"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training (Linear Regression Example)
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Prediction
y_pred = model.predict(X_test)

# 9. Evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# अगर classification problem है तो accuracy भी:
# acc = accuracy_score(y_test, y_pred_class)
# print("Accuracy:", acc)

# 10. Visualization of Predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
