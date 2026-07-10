# Predictive Maintenance (AI4I) - Apex Precision Manufacturing

Interpretable predictive-maintenance decision support for CNC operations.

Primary goal: detect elevated short-term machine failure risk (24-72h) to support pre-shift maintenance planning.

## Business Targets

- Recall >= 90% (hard requirement)
- Precision between 40% and 60% (target band)
- Daily prioritized risk report for maintenance planners

## Current Status

- Business understanding complete: reports/business_understanding.md
- Data audit complete: reports/data_audit_report.md and reports/data_audit_report.pdf
- EDA complete: src/02_EDA.py
- Data preparation complete: src/03_data_preparation.py
- Modeling and holdout evaluation complete: src/04_Modeling.py

Data prep highlights:
- leakage-safe exclusions: TWF, HDF, PWF, OSF, RNF, UDI, Product ID
- engineered features: temp_diff_K, power_W, wear_torque
- stratified split: train/val/test = 70/15/15
- one-hot encoding for Type

## Holdout Result (Candidate A)

Final frozen operating point (threshold = 0.19):
- Recall: 0.9608
- Precision: 0.2952
- F1: 0.4516
- PR-AUC: 0.9416

Confusion/workload summary on holdout:
- TP: 49
- FP: 117
- FN: 2
- TN: 1332
- Alerts total: 166
- False alerts per true catch: 2.39

## Simulated Client Decision

Decision: Conditional Go for pilot (not full rollout yet).

Rationale:
- Hard KPI met: recall is above 90%.
- Risk appetite supports high recall (missed failures are costly).
- Precision is below the preferred target band, so alert load needs operational control.

Approved operating policy:
- Mode A (High Recall): threshold 0.19 for safety-critical periods.
- Mode B (Capacity Managed): use a higher threshold chosen by planner capacity.
- Daily output remains decision support; maintenance engineers keep final authority.

Pilot exit criteria (4-6 weeks):
- Maintain recall >= 0.90 in shadow/ops monitoring.
- Keep daily alert volume within planner capacity.
- Demonstrate acceptable false-alert burden for on-shift workflow.

## What To Do Next

1. Implement threshold scenario table on holdout (for example 0.19/0.22/0.25/0.28).
2. Choose Mode B threshold with Maintenance Manager using daily capacity limits.
3. Add subgroup checks by Type and drift checks train vs validation/test.
4. Add explainability summary for top alerts.
5. Prepare deployment playbook for daily batch scoring.

## Repository Layout

```text
.
├── 0_utils/                  # io_utils, split_utils, plotting, experiment helpers
├── configs/
├── data/
│   ├── raw/                  # ai4i2020.csv
│   └── processed/            # train/val/test parquet + metadata.json
├── models/                   # trained models + candidate_a_* artifacts
├── notebooks/
├── reports/
│   ├── CRISP_DM_PLAN.md
│   ├── business_understanding.md
│   ├── data_audit_report.md
│   ├── data_audit_report.pdf
│   └── figures/
├── src/
│   ├── 01_data_load.py
│   ├── 02_EDA.py
│   ├── 02b_data_audit_report.py
│   ├── 03_data_preparation.py
│   └── 04_Modeling.py
├── tests/
├── INITIAL.md
├── pyproject.toml
└── uv.lock
```

## Quick Start

```powershell
uv sync
```

Run key project steps:

```powershell
uv run python src/02b_data_audit_report.py
uv run python src/02_EDA.py
uv run python src/03_data_preparation.py
uv run python src/04_Modeling.py
```

## Key Outputs

- data/processed/train.parquet
- data/processed/val.parquet
- data/processed/test.parquet
- data/processed/metadata.json
- models/candidate_a_model.pkl
- models/candidate_a_spec.json
- models/candidate_a_holdout_report.json

Metadata includes target, selected model features, excluded columns, split sizes, and class ratios.
