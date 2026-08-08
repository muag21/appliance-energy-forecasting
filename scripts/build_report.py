"""Render reports/report.md to a print-ready A4 PDF.

Requires: pandoc (system) and weasyprint (pip install weasyprint).
Placeholders of the form ⟦FILL: ...⟧ are rendered as highlighted spans so
that unfinished sections are visible on the page.

    python scripts/build_report.py
"""
import re, pathlib, subprocess
from weasyprint import HTML, CSS

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / "reports" / "report.md").read_text()

# --- Protect inline code spans and fenced blocks from substitution ----------
vault = []
def stash(m):
    vault.append(m.group(0))
    return f"@@V{len(vault)-1}@@"

protected = re.sub(r"```.*?```", stash, src, flags=re.S)
protected = re.sub(r"`[^`\n]+`", stash, protected)

# --- Turn placeholders into highlighted spans ------------------------------
def fill(m):
    body = " ".join(m.group(1).split())
    return f'<span class="fill">FILL: {body}</span>'

protected = re.sub(r"⟦FILL:?\s*(.*?)⟧", fill, protected, flags=re.S)

for i, original in enumerate(vault):
    protected = protected.replace(f"@@V{i}@@", original)

pathlib.Path("/tmp/pre.md").write_text(protected)

# --- Markdown -> HTML fragment ---------------------------------------------
body = subprocess.run(
    ["pandoc", "/tmp/pre.md", "-f", "markdown+pipe_tables+raw_html",
     "-t", "html5"],
    capture_output=True, text=True, check=True,
).stdout

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>{body}</body></html>"

CSS_TEXT = """
@page {
  size: A4; margin: 17mm 17mm 15mm 17mm;
  @bottom-center {
    content: counter(page) " of " counter(pages);
    font-family: "DejaVu Sans", sans-serif; font-size: 8pt; color: #6a7280;
  }
}
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 9.6pt;
       line-height: 1.36; color: #16181d; text-align: justify; hyphens: auto; }
h1 { font-family: "DejaVu Sans", sans-serif; font-size: 18pt; line-height: 1.22;
     margin: 0 0 10pt 0; text-align: left; font-weight: 700;
     string-set: doctitle content(); }
h2 { font-family: "DejaVu Sans", sans-serif; font-size: 11.5pt;
     margin: 14pt 0 5pt 0; text-align: left;
     border-bottom: 1pt solid #16181d; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-family: "DejaVu Sans", sans-serif; font-size: 10pt;
     margin: 10pt 0 4pt 0; text-align: left; break-after: avoid; }
p { margin: 0 0 6pt 0; orphans: 2; widows: 2; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f1f3f5; padding: 0.5pt 2.5pt; }
pre { background: #f1f3f5; padding: 7pt 9pt; border-left: 2.5pt solid #9aa1ac;
      font-size: 8.4pt; text-align: left; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0 12pt 0;
        font-size: 8.2pt; break-inside: avoid; }
th { background: #eceef1; border: 0.6pt solid #b8bec7; padding: 2.5pt 4pt;
     text-align: left; font-family: "DejaVu Sans", sans-serif; font-size: 8.6pt; }
td { border: 0.6pt solid #ccd1d8; padding: 2.5pt 4pt; text-align: left;
     vertical-align: top; }
ul, ol { margin: 0 0 9pt 0; padding-left: 16pt; }
li { margin-bottom: 2.5pt; }
hr { border: none; border-top: 0.6pt solid #ccd1d8; margin: 10pt 0; }
img { max-width: 100%; display: block; margin: 8pt auto 3pt auto; }
img + em, p > em:only-child { font-size: 8.2pt; color: #52585f; }
blockquote { border-left: 3pt solid #c8901c; background: #fdf6e6;
             margin: 12pt 0; padding: 8pt 12pt; font-size: 9.3pt;
             break-inside: avoid; }
blockquote p { margin-bottom: 5pt; }
.fill { background: #ffe6a0; border: 0.5pt solid #d9a520;
        padding: 0.5pt 3pt; font-family: "DejaVu Sans", sans-serif;
        font-size: 8.5pt; color: #6b4a00; }
sub, sup { font-size: 7.5pt; }
"""

HTML(string=html, base_url=str(ROOT / "reports")).write_pdf(
    str(ROOT / "reports" / "report.pdf"), stylesheets=[CSS(string=CSS_TEXT)]
)
print("built")
