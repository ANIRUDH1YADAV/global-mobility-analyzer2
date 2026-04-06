import pickle
import pandas as pd

# ─── Load PKL Files ──────────────────────────────────────
print("=== Loading PKL Files ===")

with open("artifact/data_transformation/transformed_object/preprocessing.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("artifact/model_trainer/trained_model/model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"✅ Preprocessor loaded: {type(preprocessor)}")
print(f"✅ Model loaded:        {type(model)}")

# ─── Check Column Match ──────────────────────────────────
print("\n=== Checking Column Match ===")

ohe_cols   = preprocessor.named_transformers_["cat"].get_feature_names_out()
num_cols   = ["no_of_employees", "yr_of_estab", "prevailing_wage"]
pre_cols   = list(num_cols) + list(ohe_cols)
mod_cols   = list(model.feature_names_in_)

print(f"Preprocessor columns ({len(pre_cols)}): {pre_cols}")
print(f"Model columns        ({len(mod_cols)}): {mod_cols}")
print(f"Match: {pre_cols == mod_cols}")

# ─── Load Dataset ────────────────────────────────────────
print("\n=== Loading Dataset ===")

df = pd.read_csv('C:\\Users\\aniru\\globel\\global-mobility-analyzer\\dataset\\EDA_data.csv')
print(f"Total rows: {len(df)}")
print(f"Certified: {(df['case_status'] == 'Certified').sum()}")
print(f"Denied:    {(df['case_status'] == 'Denied').sum()}")

# ─── Test Known CERTIFIED Case ───────────────────────────
print("\n=== Testing Known CERTIFIED Case ===")

certified_case = df[df["case_status"] == "Certified"].iloc[0]
print(f"Input: {certified_case.drop('case_status').to_dict()}")

X_certified = pd.DataFrame([{
    "continent":             certified_case["continent"],
    "education_of_employee": certified_case["education_of_employee"],
    "has_job_experience":    certified_case["has_job_experience"],
    "requires_job_training": certified_case["requires_job_training"],
    "no_of_employees":       certified_case["no_of_employees"],
    "yr_of_estab":           certified_case["yr_of_estab"],
    "region_of_employment":  certified_case["region_of_employment"],
    "prevailing_wage":       certified_case["prevailing_wage"],
    "unit_of_wage":          certified_case["unit_of_wage"],
    "full_time_position":    certified_case["full_time_position"],
}])

transformed  = preprocessor.transform(X_certified)
prediction   = model.predict(transformed)
result       = "Certified ✅" if prediction[0] == 1 else "Denied ❌"
print(f"Raw prediction: {prediction[0]}")
print(f"Result:         {result}")

# ─── Test Known DENIED Case ──────────────────────────────
print("\n=== Testing Known DENIED Case ===")

denied_case = df[df["case_status"] == "Denied"].iloc[0]
print(f"Input: {denied_case.drop('case_status').to_dict()}")

X_denied = pd.DataFrame([{
    "continent":             denied_case["continent"],
    "education_of_employee": denied_case["education_of_employee"],
    "has_job_experience":    denied_case["has_job_experience"],
    "requires_job_training": denied_case["requires_job_training"],
    "no_of_employees":       denied_case["no_of_employees"],
    "yr_of_estab":           denied_case["yr_of_estab"],
    "region_of_employment":  denied_case["region_of_employment"],
    "prevailing_wage":       denied_case["prevailing_wage"],
    "unit_of_wage":          denied_case["unit_of_wage"],
    "full_time_position":    denied_case["full_time_position"],
}])

transformed2 = preprocessor.transform(X_denied)
prediction2  = model.predict(transformed2)
result2      = "Denied ✅" if prediction2[0] == 0 else "Certified ❌"
print(f"Raw prediction: {prediction2[0]}")
print(f"Result:         {result2}")

# ─── Overall Accuracy Test ───────────────────────────────
print("\n=== Accuracy on 200 Samples ===")

sample   = df.sample(200, random_state=42)
X_sample = sample.drop(columns=["case_status"])
y_sample = sample["case_status"].apply(lambda x: 1 if x == "Certified" else 0)

transformed_sample = preprocessor.transform(X_sample)
predictions        = model.predict(transformed_sample)

correct  = sum(predictions == y_sample.values)
print(f"Correct:  {correct}/200")
print(f"Accuracy: {correct/2}%")

print("\n=== Debug Complete ===")