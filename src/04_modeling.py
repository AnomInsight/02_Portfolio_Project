# %%
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import json


from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
)

UTILS_DIR = Path(__file__).resolve().parents[1] / "0_utils"
sys.path.append(str(UTILS_DIR))

from io_utils import load_parquet, save_json, save_pickle  # type: ignore
from experiment_utils import set_seed, Timer  # type: ignore

# %%
set_seed(42)
target_col = "Machine failure"
project_root = Path(__file__).resolve().parent.parent

train_df = load_parquet(project_root / "data" / "processed" / "train.parquet")
val_df = load_parquet(project_root / "data" / "processed" / "val.parquet")

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]

X_val = X_val.reindex(columns=X_train.columns, fill_value=0)  # Ensure same columns as training set

# %%
# Make column names safe for LightGBM (no special characters, no duplicates)
def make_lgb_safe_columns(columns):
    safe = []
    seen = {}
    for c in columns:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        if not s:
            s = "col"
        n = seen.get(s, 0)
        seen[s] = n + 1
        safe.append(f"{s}_{n}" if n else s)
    return safe

# %%
# Helper functions for threshold evaluation and selection
def eval_at_threshold(y_true, proba, threshold, pr_auc=None):
    pred = (proba >= threshold).astype(int)
    if pr_auc is None:
        pr_auc = float(average_precision_score(y_true, proba))
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": pr_auc,
    }


def sweep_thresholds(y_true, proba, start=0.01, stop=0.50, step=0.01):
    pr_auc = float(average_precision_score(y_true, proba))
    rows = []
    for t in np.arange(start, stop + 1e-12, step):
        rows.append(eval_at_threshold(y_true, proba, round(float(t), 3), pr_auc=pr_auc))
    return pd.DataFrame(rows)


def random_search_model(model_cls, param_dist, fit_kwargs=None, n_iter=20, random_state=42,
                        min_recall=0.90, coarse_step=0.02,
                        X_train_data=None, y_train_data=None, X_val_data=None, y_val_data=None):
    fit_kwargs = fit_kwargs or {}
    X_train_data = X_train if X_train_data is None else X_train_data
    y_train_data = y_train if y_train_data is None else y_train_data
    X_val_data = X_val if X_val_data is None else X_val_data
    y_val_data = y_val if y_val_data is None else y_val_data

    best_result, best_params, best_model = None, None, None

    for params in ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state):
        m = model_cls(**params)
        m.fit(X_train_data, y_train_data, **fit_kwargs)
        proba = m.predict_proba(X_val_data)[:, 1]

        sweep = sweep_thresholds(y_val_data, proba, start=0.01, stop=0.40, step=coarse_step)
        cands = sweep[sweep["recall"] >= min_recall]
        if len(cands) == 0:
            continue

        candidate = cands.sort_values("precision", ascending=False).iloc[0].to_dict()
        if best_result is None or candidate["precision"] > best_result["precision"]:
            best_result = candidate
            best_params = params
            best_model = m

    return best_model, best_params, best_result


def select_threshold_consistent(df, min_recall=0.90, p_low=0.40, p_high=0.60):
    band = df[
        (df["recall"] >= min_recall) &
        (df["precision"] >= p_low) &
        (df["precision"] <= p_high)
    ]
    if len(band) > 0:
        return band.sort_values(
            ["precision", "recall", "threshold"],
            ascending=[False, False, True]
        ).iloc[0].to_dict()

    feasible = df[df["recall"] >= min_recall]
    if len(feasible) > 0:
        return feasible.sort_values(
            ["precision", "recall", "threshold"],
            ascending=[False, False, True]
        ).iloc[0].to_dict()

    return df.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0].to_dict()






# %%
# Logistic Regression baseline
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

with Timer("LR training"):
    lr.fit(X_train, y_train)

lr_proba = lr.predict_proba(X_val)[:, 1]
lr_sweep = sweep_thresholds(y_val, lr_proba)
lr_best = select_threshold_consistent(lr_sweep)

save_pickle(lr, project_root / "models" / "logistic_regression.pkl")
save_json(lr_best, project_root / "models" / "logistic_regression_best_threshold.json")
lr_sweep.to_csv(project_root / "models" / "logistic_regression_threshold_sweep.csv", index=False)

print("LR chosen threshold result:", lr_best)





# %%
# Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

with Timer("RF training"):
    rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_val)[:, 1]
rf_sweep = sweep_thresholds(y_val, rf_proba)
rf_best = select_threshold_consistent(rf_sweep)

save_pickle(rf, project_root / "models" / "random_forest.pkl")
save_json(rf_best, project_root / "models" / "random_forest_best_threshold.json")
rf_sweep.to_csv(project_root / "models" / "random_forest_threshold_sweep.csv", index=False)

print("RF chosen threshold result:", rf_best)





# %%
# RF random search
from scipy.stats import randint, uniform

rf_param_dist = {
    "n_estimators": randint(300, 1200),
    "max_depth": [6, 8, 12, 16, None],
    "min_samples_leaf": randint(3, 30),
    "min_samples_split": randint(5, 50),
    "max_features": ["sqrt", 0.5, 0.7, None],
    "class_weight": ["balanced_subsample"],
    "random_state": [42],
    "n_jobs": [-1],
}

best_model, best_params, best_result = random_search_model(
    RandomForestClassifier,
    rf_param_dist,
    n_iter=20,
    random_state=42,
)

print("Best RF params (random search):", best_params)
print("Best RF result (coarse):", best_result)






# %%
# Gradient Boosting baseline

w_train = compute_sample_weight(class_weight="balanced", y=y_train)

gb = HistGradientBoostingClassifier(
    learning_rate=0.03,
    max_iter=800,
    max_depth=None,
    max_leaf_nodes=63,
    min_samples_leaf=5,
    random_state=42,
)

with Timer("GB training"):
    gb.fit(X_train, y_train, sample_weight=w_train)

gb_proba = gb.predict_proba(X_val)[:, 1]
gb_sweep = sweep_thresholds(y_val, gb_proba, start=0.01, stop=0.50, step=0.01)
gb_best = select_threshold_consistent(gb_sweep)

save_pickle(gb, project_root / "models" / "gradient_boosting.pkl")
save_json(gb_best, project_root / "models" / "gradient_boosting_best_threshold.json")
gb_sweep.to_csv(project_root / "models" / "gradient_boosting_threshold_sweep.csv", index=False)

print("GB chosen threshold result:", gb_best)



# %%
# Gradient Boosting random search
gb_param_dist = {
    "learning_rate": uniform(0.01, 0.14),      # [0.01, 0.15)
    "max_iter": randint(200, 1400),
    "max_depth": [None, 4, 6, 8, 12],
    "max_leaf_nodes": randint(15, 128),
    "min_samples_leaf": randint(2, 40),
    "l2_regularization": uniform(0.0, 2.0),
    "random_state": [42],
}

gb_best_model, gb_best_params, gb_best_result = random_search_model(
    HistGradientBoostingClassifier,
    gb_param_dist,
    fit_kwargs={"sample_weight": w_train},
    n_iter=30,
    random_state=42,
)

print("Best GB params:", gb_best_params)
print("GB tuned best:", gb_best_result)

if gb_best_model is not None:
    save_pickle(gb_best_model, project_root / "models" / "gradient_boosting_tuned.pkl")
    save_json(gb_best_result, project_root / "models" / "gradient_boosting_tuned_best_threshold.json")





# %%
# CatBoost classifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    print("CatBoost not installed. Install with: pip install catboost")
    CatBoostClassifier = None

if CatBoostClassifier is not None:
    cb = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.05,
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
        random_state=42,
        verbose=0,
    )

    with Timer("CatBoost training"):
        cb.fit(X_train, y_train)

    cb_proba = cb.predict_proba(X_val)[:, 1]
    cb_sweep = sweep_thresholds(y_val, cb_proba, start=0.01, stop=0.50, step=0.01)
    cb_best = select_threshold_consistent(cb_sweep)

    save_pickle(cb, project_root / "models" / "catboost.pkl")
    save_json(cb_best, project_root / "models" / "catboost_best_threshold.json")
    cb_sweep.to_csv(project_root / "models" / "catboost_threshold_sweep.csv", index=False)

    print("CatBoost chosen threshold result:", cb_best)
else:
    print("Skipping CatBoost (not installed)")
    cb_best = None
    
    

# %%
# CatBoost random search
cb_param_dist = {
    "iterations":        randint(100, 600),       # any int in [100, 600)
    "depth":             randint(4, 9),            # any int in [4, 8]
    "learning_rate":     uniform(0.01, 0.14),      # any float in [0.01, 0.15)
    "l2_leaf_reg":       uniform(1, 9),            # any float in [1, 10)
    "min_data_in_leaf":  randint(1, 21),           # any int in [1, 20]
    "subsample":         uniform(0.6, 0.4),        # any float in [0.6, 1.0)
    "scale_pos_weight":  [float(sum(y_train == 0) / sum(y_train == 1))],  # fixed
    "random_seed":       [42],                     # fixed
}

cb_best_model, cb_best_params, cb_best_result = random_search_model(
    CatBoostClassifier,
    cb_param_dist,
    fit_kwargs={"verbose": 0},
    n_iter=20,
    random_state=42,
)
print("CatBoost tuned best:", cb_best_result)





lgb_safe_cols = make_lgb_safe_columns(X_train.columns)
X_train_lgb = X_train.copy()
X_val_lgb = X_val.copy()
X_train_lgb.columns = lgb_safe_cols
X_val_lgb.columns = lgb_safe_cols


# %%
# LightGBM (baseline + random search)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Install with: pip install lightgbm")
    LGBMClassifier = None

if LGBMClassifier is not None:
    # ---- Baseline ----
    lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        random_state=42,
    )

    with Timer("LGBM training"):
        lgb.fit(X_train_lgb, y_train)

    lgb_proba = lgb.predict_proba(X_val_lgb)[:, 1]
    lgb_sweep = sweep_thresholds(y_val, lgb_proba, start=0.01, stop=0.50, step=0.01)
    lgb_best = select_threshold_consistent(lgb_sweep)

    save_pickle(lgb, project_root / "models" / "lightgbm.pkl")
    save_json(lgb_best, project_root / "models" / "lightgbm_best_threshold.json")
    lgb_sweep.to_csv(project_root / "models" / "lightgbm_threshold_sweep.csv", index=False)

    print("LightGBM chosen threshold result:", lgb_best)




# ---- Random search (expanded) ----
lgb_param_dist = {
    "n_estimators": randint(300, 1600),
    "learning_rate": uniform(0.005, 0.145),      # [0.005, 0.150)
    "num_leaves": randint(15, 192),
    "max_depth": randint(3, 15),                 # 3..14
    "min_child_samples": randint(5, 120),
    "subsample": uniform(0.6, 0.4),              # [0.6, 1.0)
    "colsample_bytree": uniform(0.6, 0.4),       # [0.6, 1.0)
    "reg_alpha": uniform(0.0, 3.0),
    "reg_lambda": uniform(0.0, 5.0),
    "scale_pos_weight": [float((y_train == 0).sum() / (y_train == 1).sum())],
    "random_state": [42],
}

lgb_best_model, lgb_best_params, lgb_best_result = random_search_model(
    LGBMClassifier,
    lgb_param_dist,
    n_iter=80,   # use 60-100 as discussed; 80 is a good default
    random_state=42,
    X_train_data=X_train_lgb,
    y_train_data=y_train,
    X_val_data=X_val_lgb,
    y_val_data=y_val,
)

print("Best LGBM params:", lgb_best_params)
print("LGBM tuned best:", lgb_best_result)

if lgb_best_model is not None:
    save_pickle(lgb_best_model, project_root / "models" / "lightgbm_tuned.pkl")
    save_json(lgb_best_result, project_root / "models" / "lightgbm_tuned_best_threshold.json")









# %%
# Fine sweep around best RF threshold WITHOUT retraining
if best_model is None:
    raise RuntimeError("No RF candidate met recall >= 0.90 in random search.")

rf_best_proba = best_model.predict_proba(X_val)[:, 1]
fine_sweep = sweep_thresholds(y_val, rf_best_proba, start=0.1, stop=0.3, step=0.005)
fine_cands = fine_sweep[fine_sweep["recall"] >= 0.90]

if len(fine_cands) > 0:
    fine_best = fine_cands.sort_values("precision", ascending=False).iloc[0].to_dict()
else:
    fine_best = fine_sweep.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0].to_dict()

print("Fine RF best:", fine_best)






# %%
# Compare all models on validation
rf_best_final = fine_best if "fine_best" in locals() else rf_best
gb_best_final = gb_best_result if "gb_best_result" in locals() and gb_best_result is not None else gb_best
cb_best_final = cb_best_result if "cb_best_result" in locals() and cb_best_result is not None else cb_best
lgb_best_final = lgb_best_result if "lgb_best_result" in locals() and lgb_best_result is not None else lgb_best

comparison = {
    "logistic_regression": lr_best,
    "random_forest": rf_best_final,
    "gradient_boosting": gb_best_final,
    "catboost": cb_best_final,
    "lightgbm": lgb_best_final,
}

save_json(comparison, project_root / "models" / "model_comparison_validation.json")
print("Validation comparison:\n" + json.dumps(comparison, indent=2))








# %%
# Candidate A freeze
candidate_a = {
    "name": "RF_tuned_candidate_A",
    "params": best_params,
    "threshold": float(fine_best["threshold"]),
    "val_metrics_at_threshold": {
        "recall": float(fine_best["recall"]),
        "precision": float(fine_best["precision"]),
        "f1": float(fine_best["f1"]),
        "pr_auc": float(fine_best["pr_auc"]),
    },
    "rule": "maximize precision subject to recall >= 0.90",
}

save_pickle(best_model, project_root / "models" / "candidate_a_model.pkl")
save_json(candidate_a, project_root / "models" / "candidate_a_spec.json")






# %%
# Candidate B = current best validation challenger
model_candidates = {
    "RF_tuned_candidate_A": rf_best_final,
    "GB_tuned_candidate_B": gb_best_final,
    "CatBoost_tuned_candidate_B": cb_best_final,
    "LightGBM_tuned_candidate_B": lgb_best_final,
}

candidate_b_name, candidate_b_metrics = max(
    model_candidates.items(),
    key=lambda item: (item[1]["precision"], item[1]["recall"], -item[1]["threshold"]),
)

candidate_b = {
    "name": candidate_b_name,
    "threshold": float(candidate_b_metrics["threshold"]),
    "val_metrics_at_threshold": {
        "recall": float(candidate_b_metrics["recall"]),
        "precision": float(candidate_b_metrics["precision"]),
        "f1": float(candidate_b_metrics["f1"]),
        "pr_auc": float(candidate_b_metrics["pr_auc"]),
    },
    "rule": "current best validation challenger under recall >= 0.90",
}

save_json(candidate_b, project_root / "models" / "candidate_b_spec.json")
print("Candidate B:\n" + json.dumps(candidate_b, indent=2))





# %%
# Final, governance-safe LightGBM flow:
# 1) pick threshold on DEV using OOF (Out-Of-Fold) probabilities
# 2) retrain once on full DEV
# 3) freeze artifacts for evaluation phase

# Dev pool = train + val
dev_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
X_dev = dev_df.drop(columns=[target_col])
y_dev = dev_df[target_col]

# Use your safe-column helper
lgb_safe_cols = make_lgb_safe_columns(X_dev.columns)
X_dev_lgb = X_dev.copy()
X_dev_lgb.columns = lgb_safe_cols

# Freeze model params (use tuned if available, else fallback)
lgb_params_frozen = (
    lgb_best_params
    if "lgb_best_params" in locals() and lgb_best_params is not None
    else {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float((y_dev == 0).sum() / (y_dev == 1).sum()),
        "random_state": 42,
    }
)

# OOF probabilities (5-fold stratified)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(X_dev_lgb), dtype=float)

for tr_idx, va_idx in skf.split(X_dev_lgb, y_dev):
    X_tr, X_va = X_dev_lgb.iloc[tr_idx], X_dev_lgb.iloc[va_idx]
    y_tr = y_dev.iloc[tr_idx]

    m = LGBMClassifier(**lgb_params_frozen)
    m.fit(X_tr, y_tr)
    oof_proba[va_idx] = m.predict_proba(X_va)[:, 1]

# Threshold from OOF ONLY
oof_sweep = sweep_thresholds(y_dev, oof_proba, start=0.01, stop=0.50, step=0.01)
oof_best = select_threshold_consistent(oof_sweep, min_recall=0.90, p_low=0.40, p_high=0.60)
frozen_threshold = float(oof_best["threshold"])
print("OOF-selected threshold:", oof_best)

# Retrain once on full dev
lgb_final = LGBMClassifier(**lgb_params_frozen)
lgb_final.fit(X_dev_lgb, y_dev)

# Save frozen model + threshold for evaluation script
save_pickle(lgb_final, project_root / "models" / "lightgbm_final_dev.pkl")
save_json(
    {
        "model": "LightGBM",
        "params_frozen": lgb_params_frozen,
        "threshold_frozen_from_oof": frozen_threshold,
        "oof_best": oof_best,
    },
    project_root / "models" / "lightgbm_final_freeze_and_holdout.json",
)


# %%
# Basic model plot: OOF threshold tradeoff (for quick portfolio visualization)
plt.figure(figsize=(8, 5))
plt.plot(oof_sweep["threshold"], oof_sweep["recall"], label="OOF recall")
plt.plot(oof_sweep["threshold"], oof_sweep["precision"], label="OOF precision")
plt.axhline(0.90, color="red", linestyle=":", label="Recall target 0.90")
plt.axvline(frozen_threshold, color="black", linestyle="--", label=f"Frozen threshold = {frozen_threshold:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("LightGBM OOF Threshold Tradeoff")
plt.legend()
plt.tight_layout()
plt.savefig(project_root / "reports" / "figures" / "oof_threshold_tradeoff.png", dpi=180)
plt.show()
