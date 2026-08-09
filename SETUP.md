# Setup and workflow guide

Everything from an empty folder to a submitted project: where to put the files,
how to create the repository, and how to work in both a terminal and Colab.

---

## Part 1 — Where to put the project

### Choosing a location

Put it somewhere short, permanent and **not** inside a syncing folder.

| Platform | Recommended | Avoid |
|---|---|---|
| Windows | `C:\Users\<you>\projects\` | OneDrive, Desktop, Documents |
| macOS | `~/projects/` | iCloud Drive, Desktop |
| Linux | `~/projects/` | any synced mount |

Three reasons this matters. Cloud-syncing folders fight with git — OneDrive and
Dropbox rewrite files under `.git/` mid-operation and produce corruption that is
tedious to unpick. Spaces in a path (`My Documents`) break shell commands that
aren't quoted carefully. And deep nesting makes every command longer than it
needs to be.

### Unpacking

```bash
mkdir -p ~/projects
cd ~/projects
unzip ~/Downloads/appliance-energy-forecasting.zip
mv project appliance-energy-forecasting
cd appliance-energy-forecasting
ls
```

You should see `README.md`, `src/`, `scripts/`, `tests/`, `notebooks/`,
`reports/`.

The zip already contains a git repository with commits, so you do not need
`git init`. Confirm:

```bash
git log --oneline
```

---

## Part 2 — Creating the GitHub repository

### Step 1: create the empty remote

On github.com, click **New repository**. Then:

- **Name:** `appliance-energy-forecasting`
- **Visibility:** Public is simplest. Private works but Colab will need a token.
- **Do not** tick "Add a README", "Add .gitignore" or "Choose a license"

That last point matters. Those options create a commit on the remote, which
conflicts with your local history and produces a rejected push on your first
attempt. Start the remote genuinely empty.

### Step 2: connect and push

```bash
cd ~/projects/appliance-energy-forecasting

git config user.name  "Your Name"
git config user.email "your.email@example.com"

git remote add origin https://github.com/YOUR_USERNAME/appliance-energy-forecasting.git
git branch -M main
git push -u origin main
```

### Step 3: authentication

GitHub stopped accepting passwords in 2021. When prompted for one, supply a
**personal access token** instead:

1. github.com → your avatar → Settings → Developer settings
2. Personal access tokens → Tokens (classic) → Generate new token
3. Tick the `repo` scope, set an expiry, generate
4. Copy it immediately — it is shown once
5. Paste it as the password at the git prompt

Store it so you are not asked repeatedly:

```bash
git config --global credential.helper store    # Linux
git config --global credential.helper osxkeychain    # macOS
# Windows: git config --global credential.helper manager
```

Treat the token as a password. Never paste it into a notebook cell you intend to
commit.

### Step 4: verify

Reload the repository page. You should see 39 files and your commit messages.
Check that `data/raw/` is **empty** — the 12 MB CSV is git-ignored deliberately,
and committing large data files is a common mark deduction.

---

## Part 3 — Running locally (terminal)

### One-time setup

```bash
cd ~/projects/appliance-energy-forecasting

python -m venv .venv

source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
```

Your prompt should now show `(.venv)`. Re-activate it every new terminal
session; forgetting is the single most common source of "module not found".

### The four commands you will actually use

```bash
pytest                             # 1. tests pass?          ~3 s
python scripts/run_pipeline.py     # 2. produce results      ~4 min
python scripts/verify.py           # 3. ready to submit?     ~30 s
python scripts/build_report.py     # 4. rebuild the PDF      ~5 s
```

Run them in that order. If step 1 fails, nothing after it is trustworthy.

### Useful variations

```bash
python scripts/run_pipeline.py --no-foundation   # skip Chronos (faster)
python scripts/run_pipeline.py --secondary       # add the 336-step experiment
python scripts/run_pipeline.py --tune            # grid search (slow)
python scripts/build_report.py CHECKLIST.md      # PDF any markdown file
pytest -v                                        # per-test detail
pytest tests/test_features.py                    # one file only
```

### Where things land

| Path | Contents |
|---|---|
| `data/raw/` | Downloaded CSV (git-ignored, ~12 MB) |
| `outputs/metrics/` | `model_comparison.csv` and diagnostics |
| `outputs/forecasts/` | `all_forecasts.csv`, 336 rows |
| `outputs/figures/` | Six PNGs |
| `reports/report.pdf` | Built from `report.md` |

### Committing your work

```bash
git status                         # what changed
git add -A
git commit -m "Write Section 8 using Chronos results"
git push
```

Commit after each meaningful chunk rather than once at the end. A history of
small, described commits is itself evidence of process.

---

## Part 4 — Running on Colab

Colab matters for one specific reason: it can reach the Chronos model host, and
it offers a free GPU. Everything else runs equally well locally.

### The workflow

1. **colab.research.google.com** → File → Upload notebook →
   `notebooks/00_colab_quickstart.ipynb`
2. Optionally: Runtime → Change runtime type → **T4 GPU**
3. In cell 1, replace `YOUR_USERNAME` with your GitHub username
4. Runtime → **Run all**
5. Run the final download cell **before closing the tab**

### Getting the code into the runtime

Uploading the notebook does not upload the project. The runtime starts empty;
cell 1 fetches the code.

**If you have pushed to GitHub** (recommended):

```python
!git clone https://github.com/YOUR_USERNAME/appliance-energy-forecasting.git /content/project
%cd /content/project
```

**If you have not pushed**, drag the zip into the Files panel on the left, then:

```python
!unzip -q -o /content/appliance-energy-forecasting.zip -d /content
%cd /content/project
```

Either way, run the **bootstrap cell** (section 2b) afterwards. It locates the
project and fixes `sys.path`, and every later cell depends on it.

### The five Colab traps

**The filesystem is wiped on disconnect.** Nothing persists. Download or push
before you close the tab, or you will re-run everything.

**`%cd` persists, `!cd` does not.** The `!` form runs in a subshell that exits
immediately. This is the cause of most "file not found" errors.

**Relative paths break across cells.** `sys.path.insert(0, "src")` only works if
`cwd` is the project root. The bootstrap cell resolves an absolute path instead,
which is why it exists.

**Never `pip install` numpy, pandas or torch.** Colab's versions work. Upgrading
forces a runtime restart that loses your state. Only `chronos-forecasting` is
genuinely missing.

**Runtime restarts reset everything in memory.** After any restart, re-run the
bootstrap cell before anything else.

### Keeping local and Colab in sync

The clean loop, using GitHub as the single source of truth:

```
edit locally  →  git push  →  Colab: git pull  →  run  →  download outputs
```

To pull fresh code into an existing Colab session:

```python
%cd /content/project
!git pull
```

To push results from Colab back — but only with a token you are willing to have
in a notebook, so prefer downloading:

```python
!git add outputs/ && git commit -m "Chronos results from Colab"
!git push https://USERNAME:TOKEN@github.com/USERNAME/appliance-energy-forecasting.git main
```

### Persisting via Drive (optional)

If you would rather not re-clone each session:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/project /content/drive/MyDrive/
```

Working directly from Drive is slower for file-heavy operations, so copy in,
work in `/content`, copy results back.

---

## Part 5 — Common errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: appliance_energy` | `cwd` is not the project root | Run the bootstrap cell; locally, check you are in the project directory with the venv active |
| `ModuleNotFoundError: statsmodels` | venv not activated | `source .venv/bin/activate` |
| `mean_squared_error() got an unexpected keyword 'squared'` | scikit-learn ≥ 1.6 | Already handled in this codebase; if you adapt outside code, use `np.sqrt(mean_squared_error(...))` |
| `remote: Repository not found` | Wrong URL, or private repo without a token | Check the URL; make public or supply a token |
| `Updates were rejected` | Remote was created with a README | `git pull --rebase origin main` then push |
| `Support for password authentication was removed` | Using a password | Use a personal access token |
| `403` fetching the Chronos model | Restricted network | Run on Colab |
| Pipeline produces different numbers on re-run | Something unseeded | Investigate before submitting; this is a reproducibility failure |
| `!cd` had no effect | Subshell | Use `%cd` |

---

## Part 6 — The order to do things

```
1. Unpack locally, create the GitHub repo, push
2. Locally:  pytest  →  run_pipeline --no-foundation  →  verify
3. Colab:    upload notebook  →  run all  →  note Chronos numbers  →  download
4. Locally:  write Section 8 and question 4 in reports/report.md
5. Locally:  rewrite the prose in your own words
6. Locally:  verify  →  build_report  →  commit  →  push
7. Clean-clone test: clone into a fresh directory and run it end to end
8. Work through CHECKLIST.md
```

Step 7 is the one people skip and the one that catches "works on my machine".
Do it from a directory that is not your working copy, in a fresh virtual
environment.
