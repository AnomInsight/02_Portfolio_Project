# %%
from pathlib import Path
import sys
import pandas as pd

UTILS_DIR = Path(__file__).resolve().parents[1] / "0_utils"
sys.path.append(str(UTILS_DIR))

from io_utils import load_csv, save_parquet, save_json # type: ignore
from split_utils import random_split, split_summary # type: ignore

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "ai4i2020.csv"
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(DATA_PATH)
df = load_csv(DATA_PATH)

# %%
# Feature engineering (domain-based) — do this before splitting
df["temp_diff_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["power_W"] = (2 * 3.141592653589793 * df["Rotational speed [rpm]"] * df["Torque [Nm]"]) / 60
df["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]
# print(f"Engineered features added. df.columns now: {df.columns.tolist()}")

# %%
# Target
target_col = "Machine failure"
leakage_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
id_cols = ["UDI", "Product ID"]

# define model feature list (now includes engineered features)
model_features = [
    c for c in df.columns
    if c not in [target_col] + leakage_cols + id_cols
]

X = df.loc[:, model_features].copy()
y = df[target_col].copy()

# print("All columns:", df.columns.tolist())
# print("Model features:", model_features)
# print("Excluded from modeling only:", leakage_cols + id_cols + [target_col])

# %%
# Split the dataset into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = random_split(
    X, y, 
    val_size=0.15, 
    test_size=0.15, 
    random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
split_summary(y_train, y_val, y_test)

# %%
# Preprocessing: one-hot encode categorical column on splits
X_train = pd.get_dummies(X_train, columns=["Type"], drop_first=False)
X_val = pd.get_dummies(X_val, columns=["Type"], drop_first=False)
X_test = pd.get_dummies(X_test, columns=["Type"], drop_first=False)

# Save the processed splits to Parquet files
train_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "train.parquet"
val_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "val.parquet"
test_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.parquet"
save_parquet(pd.concat([X_train, y_train], axis=1), str(train_path))
save_parquet(pd.concat([X_val, y_val], axis=1), str(val_path))
save_parquet(pd.concat([X_test, y_test], axis=1), str(test_path))

# %%
# Save feature metadata
metadata = {
    "target_col": target_col,
    "model_features": model_features,
    "excluded_cols": leakage_cols + id_cols + [target_col],
    "split_sizes": {
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
    },
    "class_ratio": {
        "train": y_train.value_counts(normalize=True).to_dict(),
        "val": y_val.value_counts(normalize=True).to_dict(),
        "test": y_test.value_counts(normalize=True).to_dict(),
    },
}
metadata_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "metadata.json"
save_json(metadata, str(metadata_path))

print("Data preparation complete!")
# %%
