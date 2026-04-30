"""
Titanic Dataset - Exploratory Data Analysis
Step 1: Load data and examine structure, survival rates, missing values
"""

import pandas as pd
import seaborn as sns

# Load Titanic dataset via seaborn's built-in loader
df = sns.load_dataset('titanic')

print("=" * 60)
print("1. FIRST 5 ROWS")
print("=" * 60)
print(df.head().to_string())

print("\n" + "=" * 60)
print("2. SHAPE AND COLUMN NAMES")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

print("\n" + "=" * 60)
print("3. SURVIVAL RATE")
print("=" * 60)
survival_rate = df['survived'].mean()
survival_counts = df['survived'].value_counts()
print(f"Overall survival rate: {survival_rate:.4f} ({survival_rate*100:.2f}%)")
print(f"\nSurvived (1): {survival_counts[1]} passengers")
print(f"Did not survive (0): {survival_counts[0]} passengers")
print(f"Total passengers: {len(df)}")

# Survival by class and sex
print("\nSurvival rate by Sex:")
print(df.groupby('sex')['survived'].mean().apply(lambda x: f"{x:.4f} ({x*100:.2f}%)"))
print("\nSurvival rate by Passenger Class:")
print(df.groupby('pclass')['survived'].mean().apply(lambda x: f"{x:.4f} ({x*100:.2f}%)"))

print("\n" + "=" * 60)
print("4. MISSING VALUE COUNTS")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df.to_string())
print(f"\nTotal cells: {df.shape[0] * df.shape[1]}")
print(f"Total missing cells: {df.isnull().sum().sum()}")
print(f"Overall missing rate: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
