# %%
import pandas as pd
from pathlib import Path

DATA_PATH = Path("../data/raw/ai4i2020.csv")

df = pd.read_csv(DATA_PATH)

# %%
print("Info:", df.info())
print("Description:", df.describe())
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

# Quick target checks
print("\nMachine failure distribution:")
print(df["Machine failure"].value_counts(dropna=False))
print("\nMachine failure ratio:")
print(df["Machine failure"].value_counts(normalize=True))

df.head()
# %%
