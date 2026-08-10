"""Pre-submission verification.

Checks what a machine can check: that the repository runs from clean, that the
tests pass, that outputs exist and are internally consistent, and that no draft
placeholders remain.

    python scripts/verify.py

Exit status is 0 only if every check passes.  Checks that require judgement —
whether the prose is yours, whether the interpretation is defensible — are
listed in CHECKLIST.md and cannot be automated.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


# ---------------------------------------------------------------------------
# 1. Repository integrity
# ---------------------------------------------------------------------------

def check_structure() -> None:
    required = [
        "README.md", "requirements.txt", "pyproject.toml", ".gitignore",
        "src/appliance_energy/config.py", "src/appliance_energy/features.py",
        "src/appliance_energy/evaluation.py",
        "src/appliance_energy/models/benchmarks.py",
        "src/appliance_energy/models/sarimax.py",
        "src/appliance_energy/models/feature_models.py",
        "src/appliance_energy/models/foundation.py",
        "scripts/run_pipeline.py",
        "tests/test_features.py", "tests/test_evaluation.py",
        "reports/report.md",
    ]
    missing = [f for f in required if not (ROOT / f).exists()]
    check("Required files present", PASS if not missing else FAIL,
          "" if not missing else f"missing: {', '.join(missing)}")

    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    check("Notebooks present", PASS if len(notebooks) >= 7 else WARN,
          f"{len(notebooks)} found, 7 expected")


def check_no_large_files() -> None:
    """Raw data must not be committed."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    except Exception:
        check("Git repository initialised", WARN, "git not available or not a repo")
        return

    check("Git repository initialised", PASS, f"{len(tracked)} files tracked")

    offenders = []
    for rel in tracked:
        path = ROOT / rel
        if path.exists() and path.stat().st_size > 5_000_000:
            offenders.append(f"{rel} ({path.stat().st_size // 1_000_000} MB)")

    check("No large files committed", PASS if not offenders else FAIL,
          "" if not offenders else "; ".join(offenders))

    raw = [f for f in tracked if f.startswith("data/raw/") and f.endswith(".csv")]
    check("Raw data not committed", PASS if not raw else FAIL,
          "" if not raw else ", ".join(raw))


# ---------------------------------------------------------------------------
# 2. Tests
# ---------------------------------------------------------------------------

def check_tests() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "no output"
    check("Test suite passes", PASS if proc.returncode == 0 else FAIL, tail)


# ---------------------------------------------------------------------------
# 3. Outputs exist and are consistent
# ---------------------------------------------------------------------------

def check_outputs() -> None:
    from appliance_energy import config

    forecasts = config.FORECAST_DIR / "all_forecasts.csv"
    metrics = config.METRICS_DIR / "model_comparison.csv"

    if not forecasts.exists() or not metrics.exists():
        check("Pipeline outputs exist", FAIL,
              "run: python scripts/run_pipeline.py")
        return

    fd = pd.read_csv(forecasts, index_col=0, parse_dates=True)
    res = pd.read_csv(metrics)

    check("Pipeline outputs exist", PASS)

    check("Forecast length matches test period",
          PASS if len(fd) == config.TEST_STEPS else FAIL,
          f"{len(fd)} rows, expected {config.TEST_STEPS}")

    check("No missing forecast values",
          PASS if not fd.isna().any().any() else FAIL,
          f"{int(fd.isna().sum().sum())} NaNs")

    model_cols = {c for c in fd.columns if c != "actual"}
    check("Metrics cover every forecast column",
          PASS if model_cols == set(res["model"]) else FAIL,
          f"forecasts: {len(model_cols)}, metrics: {len(res)}")

    required_metrics = {"model", "MAE", "RMSE", "MASE", "Bias"}
    check("All four metrics reported",
          PASS if required_metrics <= set(res.columns) else FAIL,
          f"columns: {list(res.columns)}")

    check("RMSE >= MAE for every model",
          PASS if (res["RMSE"] >= res["MAE"] - 1e-9).all() else FAIL)

    check("Metrics sorted by MASE",
          PASS if res["MASE"].is_monotonic_increasing else WARN)

    # Benchmarks must be present.
    needed = {"mean", "naive", "seasonal_naive_daily",
              "seasonal_naive_weekly", "drift"}
    check("All five benchmarks evaluated",
          PASS if needed <= set(res["model"]) else FAIL,
          f"missing: {needed - set(res['model'])}")

    has_sarimax = any("sarimax" in m for m in res["model"])
    has_ml = any("feature" in m for m in res["model"])
    has_fm = any(m in ("chronos_zeroshot", "foundation_model") for m in res["model"])

    check("SARIMAX evaluated", PASS if has_sarimax else FAIL)
    check("Feature-based model evaluated", PASS if has_ml else FAIL)
    check("Foundation model evaluated", PASS if has_fm else FAIL,
          "" if has_fm else "run without --no-foundation")

    figures = list((config.FIGURE_DIR).glob("*.png"))
    check("Figures generated", PASS if len(figures) >= 4 else WARN,
          f"{len(figures)} PNG files")


# ---------------------------------------------------------------------------
# 4. Report completeness
# ---------------------------------------------------------------------------

def check_report() -> None:
    report = ROOT / "reports" / "report.md"
    text = report.read_text(encoding="utf-8")

    placeholders = re.findall(r"⟦FILL.*?⟧", text, flags=re.S)
    check("No draft placeholders remain",
          PASS if not placeholders else FAIL,
          f"{len(placeholders)} remaining")

    check("Draft notice removed",
          PASS if "DRAFT NOTE" not in text and "DRAFT NOTICE" not in text else FAIL)

    sections = re.findall(r"^## (\d+)\.", text, flags=re.M)
    check("All 11 sections present",
          PASS if len(set(sections)) >= 11 else FAIL,
          f"{len(set(sections))} numbered sections")

    # The six mandatory questions should each be answered in Section 11.
    tail = text[text.rfind("## 11."):] if "## 11." in text else ""
    answered = len(re.findall(r"^\*\*\d\.", tail, flags=re.M))
    check("Six mandatory questions answered",
          PASS if answered >= 6 else FAIL, f"{answered} found in Section 11")

    words = len(re.sub(r"⟦FILL.*?⟧", "", text, flags=re.S).split())
    check("Report length plausible", PASS if 2500 <= words <= 6000 else WARN,
          f"~{words} words (6-8 pages is roughly 3000-4500)")

    for term in ("MASE", "Diebold", "leakage", "conditional"):
        if term.lower() not in text.lower():
            check(f"Report discusses '{term}'", WARN, "not found")


# ---------------------------------------------------------------------------
# 5. Numbers in the report match the outputs
# ---------------------------------------------------------------------------

def check_numbers_match() -> None:
    """Every MASE in model_comparison.csv should appear in the report."""
    from appliance_energy import config

    metrics = config.METRICS_DIR / "model_comparison.csv"
    if not metrics.exists():
        check("Report figures match pipeline output", WARN, "no metrics file")
        return

    res = pd.read_csv(metrics)
    text = (ROOT / "reports" / "report.md").read_text(encoding="utf-8")

    missing = [
        f"{row.model} ({row.MASE:.3f})"
        for row in res.itertuples()
        if f"{row.MASE:.3f}" not in text
    ]
    check("Every model's MASE appears in the report",
          PASS if not missing else FAIL,
          "" if not missing else "; ".join(missing))


# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 68)
    print("PRE-SUBMISSION VERIFICATION")
    print("=" * 68)

    for fn in (check_structure, check_no_large_files, check_tests,
               check_outputs, check_report, check_numbers_match):
        try:
            fn()
        except Exception as exc:
            check(fn.__name__, FAIL, f"check itself errored: {exc}")

    width = max(len(n) for n, _, _ in results)
    symbol = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}

    print()
    for name, status, detail in results:
        line = f"[{symbol[status]}] {name.ljust(width)}"
        if detail:
            line += f"  {detail}"
        print(line)

    failures = sum(1 for _, s, _ in results if s == FAIL)
    warnings = sum(1 for _, s, _ in results if s == WARN)

    print()
    print("-" * 68)
    print(f"{len(results)} checks: {len(results) - failures - warnings} passed, "
          f"{warnings} warnings, {failures} failures")

    if failures:
        print("\nNOT READY TO SUBMIT. Resolve the failures above.")
    else:
        print("\nAutomated checks pass. Now work through CHECKLIST.md — "
              "the judgement-based checks are the ones that carry marks.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
