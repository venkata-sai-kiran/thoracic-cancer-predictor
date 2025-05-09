import pandas as pd
import numpy as np

df=pd.read_csv("data/ThoracicCancerSurgery.csv")

missing_values=df.isnull().sum()
print(missing_values)
duplicate_values=df.duplicated().sum()
print(duplicate_values)
data_types=df.dtypes
print(data_types)
summary_stats=df.describe(include='all')
print(summary_stats)
print(df.head())

df.columns=df.columns.str.strip()
print(df.columns)

# Save original FEV1 for comparison
original_fev1 = df["FEV1"].copy()

# Calculate IQR and bounds
Q1 = df["FEV1"].quantile(0.25)
Q3 = df["FEV1"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with bounds
df["FEV1"] = df["FEV1"].apply(
    lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
)

# Compare original and modified values where changes occurred
fev1_changes = df[original_fev1 != df["FEV1"]][["FEV1"]]
fev1_changes["Original_FEV1"] = original_fev1[fev1_changes.index]
fev1_changes = fev1_changes[["Original_FEV1", "FEV1"]]
print(fev1_changes)

print(df.isnull().sum())

print(fev1_changes.head(10)) 



df.to_csv("data/ThoraricCancerSurgery.csv", index=False)
