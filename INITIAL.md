# Initial Setup

This file contains setup and day-to-day commands. Keep README.md focused on project overview.

## 0) Bootstrap project folders

Run once from project root:

```powershell
@(
	"configs",
	"data/raw",
	"data/interim",
	"data/processed",
	"data/external",
	"models",
	"notebooks",
	"reports/figures",
	"src",
	"tests"
) | ForEach-Object { New-Item -ItemType Directory -Path $_ -Force | Out-Null }
```

## Quickstart (uv)

### 1) Initialize project metadata

From repo root:

```powershell
uv init
```

### 2) Create a virtual environment with a Python version

```powershell
uv venv --python 3.11
```

### 3) Sync dependencies

```powershell
uv sync
```

This creates `.venv/` (if needed) and installs dependencies from `pyproject.toml`.

### 4) Add packages as needed

```powershell
uv add pandas numpy scikit-learn matplotlib seaborn jupyter
```

Dev dependencies:

```powershell
uv add --dev pytest ruff
```

### 5) Remove packages you no longer need

```powershell
uv remove seaborn
uv remove --dev ruff
```

### 6) Useful uv commands

Inspect environment and deps:

```powershell
uv tree
uv pip list
uv lock
```

Run tools without permanent install:

```powershell
uvx ruff check .
```

### 7) Run code inside the environment

```powershell
uv run python -m src
uv run pytest
uv run jupyter lab
```

## Git Commands (Structured)

### Setup

```powershell
git init
git branch -M main
git remote add origin <repo-url>
```

### Daily workflow

```powershell
git status
git add .
git commit -m "feat: short message"
git pull --rebase origin main
git push origin main
```

### Branch workflow

```powershell
git checkout -b feature/my-change
git push -u origin feature/my-change
git checkout main
git pull --rebase origin main
git branch -d feature/my-change
```

### Inspect and troubleshoot

```powershell
git log --oneline --graph --decorate -20
git diff
git diff --staged
git stash push -m "wip"
git stash pop
```

### Undo safely

```powershell
git restore --staged <file>
git restore <file>
git revert <commit-hash>
```

## Next Steps

- Put immutable source files in `data/raw/`.
- Keep notebooks in `notebooks/` and move reusable logic into `src/`.
- Save trained artifacts in `models/` and plots in `reports/figures/`.
- Add project-specific configs under `configs/` as needed.
