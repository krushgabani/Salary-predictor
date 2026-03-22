import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_csv('job_salary_prediction_dataset.csv')

# Prepare features
X = df.drop('salary', axis=1)
y = df['salary']

print("Encoding features...")
# One-hot encode
cols = ['education_level', 'company_size', 'industry', 'remote_work', 'job_title', 'location']
X = pd.get_dummies(X, columns=cols, drop_first=True)

print("Training model...")
# Train model with fewer trees to reduce size
model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y)

print("Saving model with maximum compression...")
# Save with maximum compression
joblib.dump(model, 'Salary_prediction.pkl', compress=9)
joblib.dump(X.columns, 'columns.pkl')

import os
file_size = os.path.getsize('Salary_prediction.pkl') / (1024 * 1024)
print(f"\nModel saved successfully!")
print(f"Model size: {file_size:.2f} MB")
print(f"Number of features: {len(X.columns)}")
