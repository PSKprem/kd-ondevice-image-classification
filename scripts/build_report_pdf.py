"""P25 — build report/report.pdf from report/report.md.

Markdown -> HTML (python-markdown, GitHub-style heading anchors, embedded print CSS) -> PDF with headless Microsoft Edge / Chrome.
Images are referenced relative to report/ (../results/figures/...), so run it from anywhere; the HTML is written next to the .md.

Usage (needs the `markdown` package, e.g. the export venv):
    D:\\kd-project\\.venv-export\\Scripts\\python.exe scripts/build_report_pdf.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "report" / "report.md"
HTML = ROOT / "report" / "report.html"
PDF = ROOT / "report" / "report.pdf"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
html { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #1b1b1b; }
body { max-width: 100%; }
h1 { font-size: 22pt; margin: 0 0 6pt 0; color: #12305a; }
h2 { font-size: 15pt; color: #12305a; border-bottom: 1.5px solid #12305a; padding-bottom: 2pt; margin-top: 20pt; page-break-after: avoid; }
h3 { font-size: 12pt; color: #1f4e8c; margin-top: 14pt; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt 0; font-size: 8.6pt; page-break-inside: avoid; }
th { background: #e8eef7; color: #12305a; }
th, td { border: 1px solid #b9c3d3; padding: 3pt 5pt; vertical-align: top; text-align: left; }
tr:nth-child(even) td { background: #fafbfd; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.8pt; background: #f2f4f7; padding: 0 2pt; border-radius: 2px; }
pre { background: #f2f4f7; border: 1px solid #d9dee6; border-radius: 4px; padding: 6pt 8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { display: block; max-width: 70%; max-height: 80mm; margin: 6pt auto; page-break-inside: avoid; }
hr { border: 0; border-top: 1px solid #c8d0dc; margin: 14pt 0; }
a { color: #1f4e8c; text-decoration: none; }
blockquote { border-left: 3px solid #b9c3d3; margin-left: 0; padding-left: 10pt; color: #444; }
"""


def slugify(value: str, sep: str) -> str:
    """GitHub-style anchor, so the table of contents links in report.md keep working."""
    return re.sub(r"[^\w\- ]", "", value.lower()).strip().replace(" ", sep)


def build_html() -> None:
    text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"slugify": slugify}})
    HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>P25 — Knowledge Distillation for On-Device '
                    f'Image Classification</title><style>{CSS}</style></head><body>{body}</body></html>', encoding="utf-8")


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    sys.exit("No Edge/Chrome found; open report/report.html in a browser and print it to PDF.")


def build_pdf() -> None:
    browser = find_browser()
    profile = tempfile.mkdtemp(prefix="p25_pdf_")
    if PDF.exists():
        PDF.unlink()
    cmd = [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer",
           "--virtual-time-budget=20000", f"--print-to-pdf={PDF}", HTML.resolve().as_uri()]
    subprocess.run(cmd, check=False, timeout=180, capture_output=True)
    shutil.rmtree(profile, ignore_errors=True)
    if not PDF.exists():
        sys.exit("PDF was not created.")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", PDF.read_bytes()))
    print(f"wrote {PDF}  ({PDF.stat().st_size / 1024:.0f} KB, ~{pages} pages)")


if __name__ == "__main__":
    build_html()
    build_pdf()
