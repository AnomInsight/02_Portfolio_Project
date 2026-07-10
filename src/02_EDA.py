# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_PATH = Path("../data/raw/ai4i2020.csv")
FIG_DIR = Path("../reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# %%
# 1) Numeric distributions
num_cols = df.select_dtypes(include="number").columns.tolist()
df[num_cols].hist(figsize=(14, 10), bins=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "numeric_histograms.png", dpi=200)
plt.show()

# %%
# 2) Correlation heatmap
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=200)
plt.show()

# %%
# 3) Compare key features by failure class
key_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
for col in key_features:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="Machine failure", y=col)
    plt.title(f"{col} by Machine failure")
    plt.tight_layout()
    safe = col.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "_")
    plt.savefig(FIG_DIR / f"boxplot_{safe}.png", dpi=200)
    plt.show()
    
# %%
# 4) Pairplot of key features
sns.pairplot(df, vars=key_features, hue="Machine failure", diag_kind="kde", corner=True)
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "pairplot_key_features.png", dpi=200)
plt.show()
# %%
# 5) Pairplot of Machine failure 1
df_failure_1 = df[df["Machine failure"] == 1]
sns.pairplot(df_failure_1, vars=key_features, diag_kind="kde", corner=True, hue="Machine failure", palette=["orange"])
plt.suptitle("Pairplot of Key Features (Machine failure = 1)", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "pairplot_key_features_failure_1.png", dpi=200)
plt.show()
