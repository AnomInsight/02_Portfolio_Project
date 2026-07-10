from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_report_lines(df: pd.DataFrame, data_path: Path) -> list[str]:
    lines: list[str] = []

    lines.append("DATA AUDIT REPORT - AI4I 2020")
    lines.append("=" * 72)
    lines.append(f"Source: {data_path}")
    lines.append("")

    lines.append("1) DATASET OVERVIEW")
    lines.append("-" * 72)
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")
    lines.append("")

    lines.append("2) SCHEMA")
    lines.append("-" * 72)
    for col, dtype in df.dtypes.items():
        lines.append(f"{col}: {dtype}")
    lines.append("")

    lines.append("3) MISSING VALUES")
    lines.append("-" * 72)
    missing = df.isna().sum().sort_values(ascending=False)
    if int(missing.sum()) == 0:
        lines.append("No missing values detected.")
    else:
        for col, n_missing in missing.items():
            if n_missing > 0:
                pct = 100.0 * n_missing / len(df)
                lines.append(f"{col}: {n_missing} ({pct:.2f}%)")
    lines.append("")

    if "Machine failure" in df.columns:
        lines.append("4) TARGET BALANCE")
        lines.append("-" * 72)
        counts = df["Machine failure"].value_counts(dropna=False).sort_index()
        for cls, n in counts.items():
            pct = 100.0 * n / len(df)
            lines.append(f"Machine failure={cls}: {n} ({pct:.2f}%)")
        lines.append("")

    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    available_modes = [c for c in failure_modes if c in df.columns]
    if available_modes:
        lines.append("5) FAILURE SUBTYPE COUNTS")
        lines.append("-" * 72)
        for col in available_modes:
            n = int(df[col].sum())
            pct = 100.0 * n / len(df)
            lines.append(f"{col}: {n} ({pct:.2f}%)")
        lines.append("")

    lines.append("6) LEAKAGE RISK REGISTER (SHORT)")
    lines.append("-" * 72)
    lines.append("TWF/HDF/PWF/OSF/RNF: HIGH - target components; exclude from modeling.")
    lines.append("UDI: LOW - identifier only; exclude from modeling.")
    lines.append("Product ID: MEDIUM - ID/serial artifact risk; exclude from baseline.")
    lines.append("Type + sensor features: LOW - valid operational predictors.")
    lines.append("")

    lines.append("7) EDA DECISIONS")
    lines.append("-" * 72)
    lines.append("Use stratified split due to class imbalance.")
    lines.append("Optimize recall first; precision constrained by operational workload.")
    lines.append("Next step: move to data prep and keep leakage columns out of the model.")

    return lines


def save_markdown(lines: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pdf(lines: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines_per_page = 45

    with PdfPages(out_path) as pdf:
        for i in range(0, len(lines), lines_per_page):
            chunk = lines[i : i + lines_per_page]
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(
                0.05,
                0.97,
                "\n".join(chunk),
                va="top",
                ha="left",
                family="monospace",
                fontsize=9,
            )
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw" / "ai4i2020.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    lines = build_report_lines(df, data_path)

    reports_dir = project_root / "reports"
    save_markdown(lines, reports_dir / "data_audit_report.md")
    save_pdf(lines, reports_dir / "data_audit_report.pdf")

    print(f"Created: {reports_dir / 'data_audit_report.md'}")
    print(f"Created: {reports_dir / 'data_audit_report.pdf'}")


if __name__ == "__main__":
    main()
