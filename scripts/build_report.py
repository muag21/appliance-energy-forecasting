"""Render a markdown file to a print-ready A4 PDF.

Uses the pure-Python `markdown` package (pip install markdown).
PDF output uses WeasyPrint when available; on Windows it usually is not,
so the script writes styled HTML instead and you print it from a browser.
Placeholders of the form ⟦FILL: ...⟧ are rendered as highlighted spans so
that unfinished sections are visible on the page.

    python scripts/build_report.py                 # reports/report.md
    python scripts/build_report.py CHECKLIST.md    # any markdown file
"""
import re, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Optional argument: any markdown file. Defaults to the report.
import sys
_arg = sys.argv[1] if len(sys.argv) > 1 else "reports/report.md"
SRC_PATH = (ROOT / _arg).resolve()
OUT_PATH = SRC_PATH.with_suffix(".pdf")
src = SRC_PATH.read_text(encoding="utf-8")

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

# Task-list checkboxes render as CSS-drawn squares, so they print and tick.
protected = re.sub(r"^(\s*)- \[ \] ", r'\1- <span class="box"></span>',
                   protected, flags=re.M)
protected = re.sub(r"^(\s*)- \[[xX]\] ", r'\1- <span class="box done"></span>',
                   protected, flags=re.M)

for i, original in enumerate(vault):
    protected = protected.replace(f"@@V{i}@@", original)


# --- Markdown -> HTML fragment ---------------------------------------------
def to_html(text):
    """Convert with the pure-Python markdown package; fall back to pandoc."""
    try:
        import markdown
        return markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "attr_list", "md_in_html"],
        )
    except ImportError:
        pass
    try:
        return subprocess.run(
            ["pandoc", "-f", "markdown+pipe_tables+raw_html", "-t", "html5"],
            input=text, capture_output=True, text=True, check=True,
            encoding="utf-8",
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "Could not convert markdown to HTML.\n"
            "Install the markdown package:  pip install markdown"
        ) from exc


body = to_html(protected)

CSS_TEXT = """
@media print { body { padding: 0; max-width: none; } }
@page {
  size: A4; margin: 17mm 17mm 15mm 17mm;
  @bottom-center {
    content: counter(page) " of " counter(pages);
    font-family: "DejaVu Sans", sans-serif; font-size: 8pt; color: #6a7280;
  }
}
body { font-family: "DejaVu Serif", Georgia, "Times New Roman", serif;
       max-width: 900px; margin: 0 auto; padding: 20px; font-size: 9.6pt;
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
pre { white-space: pre-wrap; background: #f1f3f5; padding: 7pt 9pt; border-left: 2.5pt solid #9aa1ac;
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
.box { display: inline-block; width: 8.5pt; height: 8.5pt;
        border: 0.8pt solid #6a7280; border-radius: 1.5pt;
        margin-right: 5pt; vertical-align: -0.5pt; }
.box.done { background: #6a7280; }
ul li { list-style: none; margin-left: -12pt; }
ul li ul li, ol li { list-style: revert; margin-left: 0; }
.fill { background: #ffe6a0; border: 0.5pt solid #d9a520;
        padding: 0.5pt 3pt; font-family: "DejaVu Sans", sans-serif;
        font-size: 8.5pt; color: #6b4a00; }
sub, sup { font-size: 7.5pt; }
"""

html = (
    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
    f"<title>{SRC_PATH.stem}</title><style>{CSS_TEXT}</style></head>"
    f"<body>{body}</body></html>"
)

HTML_PATH = SRC_PATH.with_suffix(".html")


def build_pdf():
    try:
        from weasyprint import HTML as _HTML
    except Exception:
        return False
    try:
        _HTML(string=html, base_url=str(SRC_PATH.parent)).write_pdf(str(OUT_PATH))
        return True
    except Exception as exc:
        print(f"WeasyPrint failed: {exc}")
        return False


if build_pdf():
    print(f"built {OUT_PATH.relative_to(ROOT)}")
else:
    import webbrowser
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"built {HTML_PATH.relative_to(ROOT)}")
    print()
    print("WeasyPrint is unavailable. On Windows it needs GTK, which the")
    print("platform does not ship, so an HTML file was written instead.")
    print()
    print("To produce the PDF:")
    print("  1. The file opens in your browser now (or open it manually).")
    print("  2. Press Ctrl+P")
    print("  3. Destination: 'Save as PDF'")
    print("  4. Paper A4, Margins 'Default', tick 'Background graphics'")
    print(f"  5. Save as {OUT_PATH.name} into the same folder")
    print()
    try:
        webbrowser.open(HTML_PATH.as_uri())
    except Exception:
        print(f"Open manually: {HTML_PATH}")
