# Predictive Maintenance (AI4I 2020) - Hypothetical Apex Precision Manufacturing

Self-directed portfolio case study using the synthetic AI4I 2020 dataset. The manufacturing scenario is hypothetical; this project is not a client engagement, pilot, or deployed CNC system.

Hypothetical business goal: use machine-failure risk estimates to prioritize maintenance attention before a shift. The 24-72 hour planning horizon is scenario framing, not a production forecast validated by this dataset.

## Business Targets

- Recall >= 90% (hypothetical hard requirement)
- Precision between 40% and 60% (hypothetical preferred target band)
- Daily prioritized risk report for maintenance planners (hypothetical deliverable)

## Current Status

- Business understanding complete
- Data audit complete
- EDA complete: src/02_EDA.py
- Data preparation complete: src/03_data_preparation.py
- Modeling and holdout evaluation complete: src/04_modeling.py, src/05_evaluation.py

Raw data, processed datasets, trained models, generated reports, and working notes are local artifacts and are excluded by .gitignore. The source, project configuration, tests, and utility modules are maintained in the repository.

Data prep highlights:
- leakage-safe exclusions: TWF, HDF, PWF, OSF, RNF, UDI, Product ID
- engineered features: temp_diff_K, power_W, wear_torque
- stratified split: train/val/test = 70/15/15
- one-hot encoding for Type

## Documented Holdout Result (Candidate A)

The following values are retained from the project's documented Candidate A experiment; they are not production validation.

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

Simulated decision: Conditional Go for a hypothetical pilot (not a real pilot or deployment).

Rationale:
- Hard KPI met: recall is above 90%.
- Risk appetite supports high recall (missed failures are costly).
- Precision is below the preferred target band, so alert load needs operational control.

Hypothetical operating policy:
- Mode A (High Recall): threshold 0.19 for safety-critical periods.
- Mode B (Capacity Managed): use a higher threshold chosen by planner capacity.
- Daily output remains decision support; maintenance engineers keep final authority.

Proposed exit criteria for a hypothetical pilot (4-6 weeks):
- Maintain recall >= 0.90 in shadow/ops monitoring.
- Keep daily alert volume within planner capacity.
- Demonstrate acceptable false-alert burden for on-shift workflow.

## Proposed Next Steps

1. Implement threshold scenario table on holdout (for example 0.19/0.22/0.25/0.28).
2. Choose Mode B threshold with Maintenance Manager using daily capacity limits.
3. Add subgroup checks by Type and drift checks train vs validation/test.
4. Add explainability summary for top alerts.
5. Prepare deployment playbook for daily batch scoring.

## Repository Structure

```text
.
├── 0_utils/                  # io_utils, split_utils, plotting, experiment helpers
├── data/                     # local raw and processed datasets; ignored
├── models/                   # local trained artifacts; ignored
├── reports/                  # local generated reports and figures; ignored
├── src/
│   ├── 01_data_load.py
│   ├── 02_EDA.py
│   ├── 02b_data_audit_report.py
│   ├── 03_data_preparation.py
│   ├── 04_modeling.py
│   └── 05_evaluation.py
├── tests/
├── .gitignore
├── INITIAL.md
├── LICENSE
├── pyproject.toml
└── uv.lock
```

## Quick Start

```powershell
uv sync
```

Run key project steps:

```powershell
uv run --directory src python 01_data_load.py
uv run --directory src python 02_EDA.py
uv run python src/02b_data_audit_report.py
uv run python src/03_data_preparation.py
uv run python src/04_modeling.py
uv run python src/05_evaluation.py
```

Before running the inspection steps, provide the UCI CSV at `data/raw/ai4i2020.csv`. The data-load, EDA, and audit scripts read this file; they do not download it.

The audit and inspection scripts require `data/raw/ai4i2020.csv`. Data preparation creates the processed splits consumed by modeling; modeling creates the frozen artifacts consumed by holdout evaluation.

## Key Outputs

Running the pipeline locally generates processed datasets, trained model artifacts, and metadata (target, selected features, excluded columns, split sizes, class ratios). These outputs are gitignored and not included in this repository.

## Limitations

AI4I 2020 is synthetic and does not establish performance on real CNC equipment. The business scenario is hypothetical, and the threshold and metrics above describe this experiment rather than production validation. Real deployment would require representative production data, operational validation, monitoring, calibration, and maintenance-team involvement.

## References

- [AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)