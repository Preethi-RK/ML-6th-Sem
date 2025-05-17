import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# 1. Download the salary dataset
data_dir = kagglehub.dataset_download("abhishek14398/salary-dataset-simple-linear-regression")
print("Files in download folder:", os.listdir(data_dir))

# 2. Find the CSV and load it
csv_file = next(f for f in os.listdir(data_dir) if f.endswith(".csv"))
df = pd.read_csv(os.path.join(data_dir, csv_file))

# 3. Inspect columns (optional)
# print(df.columns)  # should show ['YearsExperience', 'Salary']

# 4. Select predictor (X) and target (y)
X = df["YearsExperience"].values
y = df["Salary"].values

# 5. Closed-form linear regression
n   = len(X)
Sx  = X.sum()
Sy  = y.sum()
Sxy = (X * y).sum()
Sx2 = (X**2).sum()

a1 = (n * Sxy - Sx * Sy) / (n * Sx2 - Sx**2)
a0 = (Sy - a1 * Sx) / n

# 6. Predictions
y_pred = a0 + a1 * X

# 7. Metrics
mse    = np.mean((y - y_pred)**2)
ss_tot = np.sum((y - y.mean())**2)
ss_res = np.sum((y - y_pred)**2)
r2     = 1 - ss_res / ss_tot

print(f"Intercept (a0): {a0:.2f}")
print(f"Slope     (a1): {a1:.2f}")
print(f"MSE:            {mse:.2f}")
print(f"R² Score:       {r2:.3f}")

# 8. Visualization
plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.5, label="Data")
idx = np.argsort(X)
plt.plot(X[idx], y_pred[idx], color="red", label="Fit line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs. Experience (Simple Linear Regression)")
plt.legend()
plt.show()
