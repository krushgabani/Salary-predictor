import pandas as pd
import joblib

# Load model and columns
model = joblib.load('Salary_prediction.pkl')
feature_columns = joblib.load('columns.pkl')

# Test input - same as your website
user_input = {
    "experience_years": 17,
    "skills_count": 9,
    "certifications": 1,
}

# One-hot encode (drop_first=True)
all_education = ["Bachelor", "Diploma", "High School", "Master", "PhD"]
all_company = ["Enterprise", "Large", "Medium", "Small", "Startup"]
all_industry = ["Consulting", "Education", "Finance", "Government", "Healthcare", "Manufacturing", "Media", "Retail", "Technology", "Telecom"]
all_remote = ["Hybrid", "No", "Yes"]
all_jobs = ["AI Engineer", "Backend Developer", "Business Analyst", "Cloud Engineer", "Cybersecurity Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer", "Frontend Developer", "Machine Learning Engineer", "Product Manager", "Software Engineer"]
all_locations = ["Australia", "Canada", "Germany", "India", "Netherlands", "Remote", "Singapore", "Sweden", "UK", "USA"]

def ohe(prefix, all_vals, chosen, drop_first=True):
    cats = sorted(all_vals)
    if drop_first:
        cats = cats[1:]  # drop first alphabetically
    for c in cats:
        user_input[f"{prefix}_{c}"] = 1 if chosen == c else 0

# Apply encoding
ohe("education_level", all_education, "PhD")
ohe("company_size", all_company, "Medium")
ohe("industry", all_industry, "Finance")
ohe("remote_work", all_remote, "No")
ohe("job_title", all_jobs, "Cybersecurity Analyst")
ohe("location", all_locations, "India")

# Create dataframe and align columns
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict
prediction = model.predict(input_df)[0]
print(f"Predicted Salary: ${prediction:,.2f}")
print(f"\nInput features shape: {input_df.shape}")
print(f"Expected columns: {len(feature_columns)}")
