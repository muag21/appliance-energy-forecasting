"""Replace a numbered section of the report with the contents of a file.

    python scripts/apply_section.py SECTION_8_COMPLETE.md

The section number is read from the replacement file's first heading, so the
command above replaces everything from "## 8." up to (but not including)
"## 9." in reports/report.md.

A timestamped backup is written before any change, and the result is checked
for duplicated headings.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "report.md"

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/apply_section.py <section-file.md>")

# Accept a bare filename, a relative path, or a path with Windows separators.
candidate = sys.argv[1].replace("\\", "/")
paths = [ROOT / candidate, pathlib.Path(candidate),
         ROOT / pathlib.Path(candidate).name]
source = next((p for p in paths if p.exists()), None)

if source is None:
    raise SystemExit(
        f"Could not find '{sys.argv[1]}'.\n"
        f"Place it in {ROOT} and pass just the file name."
    )

replacement = source.read_text(encoding="utf-8").strip()
report = REPORT.read_text(encoding="utf-8")

# --- Which section? ---------------------------------------------------------

match = re.match(r"##\s*(\d+)\.", replacement)
if not match:
    raise SystemExit(
        "The replacement file must begin with a heading like '## 8. Title'."
    )

number = int(match.group(1))
start_pat = re.compile(rf"^##\s*{number}\.", re.M)
end_pat = re.compile(rf"^##\s*{number + 1}\.", re.M)

start_match = start_pat.search(report)
end_match = end_pat.search(report)

if not start_match:
    raise SystemExit(f"No '## {number}.' heading found in {REPORT.name}.")
if not end_match:
    raise SystemExit(f"No '## {number + 1}.' heading found in {REPORT.name}.")
if end_match.start() < start_match.start():
    raise SystemExit("Section headings are out of order; fix the file by hand.")

# --- Back up, then splice ---------------------------------------------------

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
backup = REPORT.with_name(f"report.backup-{stamp}.md")
shutil.copy2(REPORT, backup)

updated = (
    report[:start_match.start()]
    + replacement
    + "\n\n"
    + report[end_match.start():]
)

REPORT.write_text(updated, encoding="utf-8")

# --- Report what happened ---------------------------------------------------

removed = report[start_match.start():end_match.start()].count("\n")
added = replacement.count("\n")

print(f"Replaced section {number}: {removed} lines out, {added} lines in.")
print(f"Backup written to {backup.name}")

# Duplicate-heading check, which is what went wrong by hand.
headings = re.findall(r"^###?\s*\d+(?:\.\d+)?\s+.*$", updated, re.M)
duplicates = {h for h in headings if headings.count(h) > 1}

if duplicates:
    print("\nWARNING - duplicated headings remain:")
    for heading in sorted(duplicates):
        print(f"  {heading.strip()}")
else:
    print("No duplicated headings.")

remaining = updated.count("⟦FILL")
print(f"Placeholders remaining in the report: {remaining}")
