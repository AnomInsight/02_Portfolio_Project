# %%
from pathlib import Path
import sys
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
	recall_score,
	precision_score,
	f1_score,
	average_precision_score,
	confusion_matrix,
)

UTILS_DIR = Path(__file__).resolve().parents[1] / "0_utils"
sys.path.append(str(UTILS_DIR))

from io_utils import load_parquet, load_pickle, load_json, save_json, save_csv  # type: ignore
from experiment_utils import set_seed  # type: ignore


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


def eval_at_threshold(y_true, proba, threshold):
	pred = (proba >= threshold).astype(int)
	return {
		"threshold": float(threshold),
		"recall": float(recall_score(y_true, pred)),
		"precision": float(precision_score(y_true, pred, zero_division=0)),
		"f1": float(f1_score(y_true, pred, zero_division=0)),
		"pr_auc": float(average_precision_score(y_true, proba)),
	}


def build_error_table(y_true, proba, threshold):
	pred = (proba >= threshold).astype(int)
	out = pd.DataFrame({
		"y_true": y_true.values if hasattr(y_true, "values") else y_true,
		"proba": proba,
		"y_pred": pred,
	})
	out["error_type"] = np.select(
		[
			(out["y_true"] == 1) & (out["y_pred"] == 1),
			(out["y_true"] == 0) & (out["y_pred"] == 1),
			(out["y_true"] == 1) & (out["y_pred"] == 0),
		],
		["TP", "FP", "FN"],
		default="TN",
	)
	return out


def main():
	set_seed(42)
	target_col = "Machine failure"
	project_root = Path(__file__).resolve().parent.parent
	show_plots = True

	eval_dir = project_root / "reports" / "evaluation"
	eval_dir.mkdir(parents=True, exist_ok=True)

	model_path = project_root / "models" / "lightgbm_final_dev.pkl"
	freeze_path = project_root / "models" / "lightgbm_final_freeze_and_holdout.json"

	if not model_path.exists() or not freeze_path.exists():
		raise FileNotFoundError(
			"Missing frozen artifacts. Run modeling freeze step first: "
			"models/lightgbm_final_dev.pkl and models/lightgbm_final_freeze_and_holdout.json"
		)

	model = load_pickle(model_path)
	freeze = load_json(freeze_path)
	frozen_threshold = float(freeze["threshold_frozen_from_oof"])

	test_df = load_parquet(project_root / "data" / "processed" / "test.parquet")
	X_test = test_df.drop(columns=[target_col])
	y_test = test_df[target_col]

	safe_cols = make_lgb_safe_columns(X_test.columns)
	X_test_lgb = X_test.copy()
	X_test_lgb.columns = safe_cols

	proba = model.predict_proba(X_test_lgb)[:, 1]
	pred = (proba >= frozen_threshold).astype(int)

	holdout_metrics = eval_at_threshold(y_test, proba, frozen_threshold)
	cm = confusion_matrix(y_test, pred)
	tn, fp, fn, tp = [int(x) for x in cm.ravel()]

	workload = {
		"alerts_total": int(tp + fp),
		"true_failures_caught_tp": int(tp),
		"false_alerts_per_true_catch": float(fp / tp) if tp > 0 else None,
	}

	holdout_report = {
		"model": "LightGBM",
		"threshold_source": "OOF on dev (train+val), frozen before holdout",
		"holdout_metrics": holdout_metrics,
		"confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
		"workload": workload,
	}
	save_json(holdout_report, eval_dir / "holdout_report.json")

	# Confusion matrix figure
	fig, ax = plt.subplots(figsize=(5, 4))
	im = ax.imshow(cm, cmap="Blues")
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(["Pred 0", "Pred 1"])
	ax.set_yticklabels(["True 0", "True 1"])
	ax.set_title("Holdout Confusion Matrix")
	for i in range(2):
		for j in range(2):
			ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.tight_layout()
	plt.savefig(eval_dir / "confusion_matrix_holdout.png", dpi=180)
	if show_plots:
		plt.show()
	plt.close()

	# Error analysis
	errors = build_error_table(y_test, proba, frozen_threshold)
	save_csv(errors, eval_dir / "error_analysis_full.csv", index=False)
	save_csv(
		errors[errors["error_type"] == "FP"].sort_values("proba", ascending=False).head(25),
		eval_dir / "error_analysis_fp_top25.csv",
		index=False,
	)
	save_csv(
		errors[errors["error_type"] == "FN"].sort_values("proba", ascending=True).head(25),
		eval_dir / "error_analysis_fn_top25.csv",
		index=False,
	)

	# Feature importance (global)
	if hasattr(model, "feature_importances_"):
		fi = pd.DataFrame(
			{
				"feature": X_test_lgb.columns,
				"importance": model.feature_importances_,
			}
		).sort_values("importance", ascending=False)
		save_csv(fi, eval_dir / "feature_importance_global.csv", index=False)

		topn = fi.head(20).iloc[::-1]
		plt.figure(figsize=(8, 6))
		plt.barh(topn["feature"], topn["importance"])
		plt.title("Top 20 Feature Importances (Frozen Model)")
		plt.xlabel("Importance")
		plt.tight_layout()
		plt.savefig(eval_dir / "feature_importance_top20.png", dpi=180)
		if show_plots:
			plt.show()
		plt.close()

	# SHAP (optional)
	try:
		import shap  # type: ignore

		sample_n = min(1500, len(X_test_lgb))
		X_shap = X_test_lgb.sample(n=sample_n, random_state=42)
		explainer = shap.TreeExplainer(model)
		shap_values = explainer.shap_values(X_shap)

		if isinstance(shap_values, list):
			shap_values = shap_values[1]

		# 1. SHAP Bar plot (feature importance)
		plt.figure()
		shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
		plt.savefig(eval_dir / "shap_bar_importance.png", dpi=180, bbox_inches="tight")
		if show_plots:
			plt.show()
		plt.close()

		# 2. SHAP Dependence plots for top 4 features
		fi = pd.DataFrame({
			"feature": X_shap.columns,
			"importance": np.abs(shap_values).mean(axis=0),
		}).sort_values("importance", ascending=False)
		
		top_features = fi.head(4)["feature"].tolist()
		for feat in top_features:
			feat_idx = list(X_shap.columns).index(feat)
			plt.figure(figsize=(8, 5))
			shap.dependence_plot(feat_idx, shap_values, X_shap, show=False)
			plt.savefig(eval_dir / f"shap_dependence_{feat}.png", dpi=180, bbox_inches="tight")
			if show_plots:
				plt.show()
			plt.close()

		# 3. SHAP Waterfall for top TP, FP, FN examples
		errors = build_error_table(y_test, proba, frozen_threshold)
		
		for error_type in ["TP", "FP", "FN"]:
			subset = errors[errors["error_type"] == error_type].head(1)
			if len(subset) > 0:
				idx = subset.index[0]
				if idx < len(X_shap):
					plt.figure()
					shap.waterfall_plot(shap.Explanation(
						values=shap_values[idx],
						base_values=explainer.expected_value,
						data=X_shap.iloc[idx],
						feature_names=X_shap.columns
					), show=False)
					plt.savefig(eval_dir / f"shap_waterfall_{error_type}.png", dpi=180, bbox_inches="tight")
					if show_plots:
						plt.show()
					plt.close()

		print("SHAP analysis complete: bar, dependence, waterfall plots")
	except Exception as e:
		print(f"SHAP failed: {str(e)}\n  Install with: uv add shap")


	# Deployment recommendation
	go_status = "Conditional Go"
	if holdout_metrics["recall"] >= 0.90 and holdout_metrics["precision"] >= 0.40:
		go_status = "Go"

	recommendation = {
		"decision": go_status,
		"rationale": {
			"recall_target_met": holdout_metrics["recall"] >= 0.90,
			"precision_target_40pct_met": holdout_metrics["precision"] >= 0.40,
			"frozen_threshold": frozen_threshold,
		},
		"next_actions": [
			"Review FP/FN examples with maintenance team",
			"Validate alert handling capacity",
			"Monitor drift post-deployment",
		],
	}
	save_json(recommendation, eval_dir / "deployment_recommendation.json")

	print("Evaluation complete")
	print(json.dumps(holdout_report, indent=2))
 
 
 

# %%
if __name__ == "__main__":
	main()


# %%
