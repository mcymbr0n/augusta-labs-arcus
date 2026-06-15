"""
Converte writeup/writeup.md -> HTML limpo (CSS profissional) para renderizar em PDF
via Chromium headless. UTF-8, fontes com cobertura Unicode (Segoe UI / Consolas).
"""
import markdown
from pathlib import Path

ROOT = Path(r"C:\Users\migue\Documents\augusta-labs-arcus")
md_text = (ROOT / "writeup" / "writeup.md").read_text(encoding="utf-8")

html_body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    output_format="html5",
)

CSS = """
@page { size: A4; margin: 2cm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a;
}
h1 { font-size: 21pt; font-weight: 700; margin: 0 0 6pt; padding-bottom: 7pt;
     border-bottom: 2px solid #333; }
h2 { font-size: 14.5pt; font-weight: 700; margin: 22pt 0 7pt; padding-bottom: 3pt;
     border-bottom: 1px solid #ccc; page-break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: 700; margin: 14pt 0 4pt; color: #222;
     page-break-after: avoid; }
p { margin: 0 0 8pt; }
a { color: #1a4d8f; text-decoration: none; }
strong { font-weight: 700; }
em { font-style: italic; }
ul, ol { margin: 0 0 9pt; padding-left: 20pt; }
li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0 13pt;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4pt 8pt; text-align: left; vertical-align: top; }
th { background: #eef0f2; font-weight: 700; }
tr:nth-child(even) td { background: #fafbfc; }
code { font-family: "Cascadia Code", "Consolas", monospace; font-size: 9pt;
       background: #f1f2f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #dde1e6; border-radius: 5px;
      padding: 10pt 12pt; white-space: pre-wrap; word-wrap: break-word;
      page-break-inside: avoid; margin: 8pt 0 12pt; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.45; }
blockquote { border-left: 4px solid #c8ccd1; margin: 9pt 0 13pt; padding: 5pt 14pt;
             background: #f8f9fa; color: #2a2a2a; }
blockquote p { margin: 4pt 0; }
hr { border: none; border-top: 1px solid #dde1e6; margin: 16pt 0; }
"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Ode Triunfal — Write-up</title>
<style>{CSS}</style></head>
<body>
{html_body}
</body></html>
"""

out = ROOT / "writeup" / "_writeup_render.html"
out.write_text(html, encoding="utf-8")
print("HTML escrito:", out)
