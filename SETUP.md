# Setup

This project uses **uv** to manage Python and all its packages. You don't need to
install Python yourself, and you don't need to know what a virtual environment is yet
(there's a short explainer at the bottom). Just follow these steps.

---

## 1. Install uv

uv is a single tool that handles Python versions and packages for you.

**macOS / Linux** — paste this into a terminal:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** — paste this into PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal afterward, then confirm it worked:
```bash
uv --version
```
You should see something like `uv 0.11.x`. (Full docs: https://docs.astral.sh/uv/)

---

## 2. Get the code

Clone the repo, then move into it:
```bash
git clone https://github.com/childpovertyactionlab/multitenant-inspection-prediction.git
cd multitenant-inspection-prediction
```

---

## 3. Create the environment

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, downloads the right Python version, and
installs every package the project needs (pandas, scikit-learn, xgboost, JupyterLab,
and so on) into a local `.venv/` folder. It can take a minute the first time. You only
re-run `uv sync` when the dependencies change.

---

## 4. Add the data

The data is **not** in the repo (it has addresses and owner names). Copy this one file
from the shared drive into your local `_data/` folder:

```
from:  G:\Shared drives\CPAL Data Team\_Housing\_Multitenant_Inspection\_output\properties_with_acs.csv
to:    _data/properties_with_acs.csv
```

It stays on your machine — `.gitignore` makes sure it can never be committed.

---

## 5. Launch the notebook

```bash
uv run jupyter lab
```

`uv run` runs a command inside the project's environment. JupyterLab opens in your
browser; open `_code/01_replicate_bottom10.ipynb` and run the first cell. If it
prints `(2798, 33)`, everything is wired up correctly.

---

## Adding a package later

If you decide you need a library that isn't installed yet (say `plotly`):
```bash
uv add plotly
```
This installs it **and** records it in `pyproject.toml` + `uv.lock`. Commit those two
files so the next person gets it too. (Don't use `pip install` — uv won't know about it.)

---

## Everyday commands
| I want to… | Command |
|---|---|
| Install/refresh everything | `uv sync` |
| Launch the notebook | `uv run jupyter lab` |
| Run a Python script in the env | `uv run python my_script.py` |
| Add a new package | `uv add <name>` |
| Remove a package | `uv remove <name>` |

---

## What are these things? (optional reading)

- **Virtual environment (`.venv/`)** — a private, project-specific box that holds this
  project's Python and its packages, separate from anything else on your computer. So
  two projects can use different versions of the same library without fighting. uv
  creates and manages it for you; you never have to "activate" it as long as you use
  `uv run`.
- **`pyproject.toml`** — the human-edited list of which packages the project wants.
- **`uv.lock`** — the exact versions of everything (including sub-dependencies), pinned
  so that `uv sync` produces an identical environment on every machine. Always commit it.

---

## If something goes wrong

- `uv: command not found` → reopen your terminal (step 1 changed your PATH), or re-run
  the install command.
- `FileNotFoundError` when the notebook loads the CSV → the file isn't in `_data/`, or
  the name doesn't match exactly (`properties_with_acs.csv`). Re-check step 4.
- Anything else → take a screenshot of the full error and send it over Google Chat to Taylor.
