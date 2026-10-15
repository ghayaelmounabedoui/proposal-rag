"""Assemble les documents de formation et d'entretien en un seul PDF.

Usage (depuis la racine du projet, avec le venv activé) :

    python scripts/build_pdf.py                       # guide complet (parties A à F)
    python scripts/build_pdf.py --only llm --output docs/transformers-llm-lora.pdf
    python scripts/build_pdf.py --only rag

Dépendance : ``markdown-it-py`` (déjà installée avec les dépendances du projet).
Le rendu PDF utilise Microsoft Edge ou Google Chrome en mode sans interface.
"""

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# (fichier, étiquette de partie, sous-titre, description pour la couverture)
PARTS = [
    ("formation-rag.md", "Partie A", "Formation RAG de A à Z",
     "23 modules, dont le coût des appels LLM et le cache"),
    ("formation-deploiement.md", "Partie B", "Formation déploiement",
     "Docker, HTTPS, CI/CD"),
    ("entretien-rag.md", "Partie C", "Questions techniques d'entretien (RAG)",
     "plus de 200 questions avec réponses"),
    ("entretien-projet-rag.md", "Partie D", "Ton projet, conception de systèmes et code",
     "présentation, pièges, exercices de code"),
    ("formation-transformers-lora.md", "Partie E", "Formation Transformers, LLM et LoRA",
     "24 modules : attention, inférence, LoRA, QLoRA, alignement, déploiement"),
    ("entretien-transformers-lora.md", "Partie F", "Questions d'entretien Transformers, LLM et LoRA",
     "plus de 200 questions, conception de systèmes, code"),
]

# Sous-ensembles nommés pour `--only`
GROUPS = {
    "all": [p[0] for p in PARTS],
    "rag": ["formation-rag.md", "formation-deploiement.md", "entretien-rag.md", "entretien-projet-rag.md"],
    "llm": ["formation-transformers-lora.md", "entretien-transformers-lora.md"],
}

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font: 9pt 'Segoe UI', Arial, sans-serif; color: #666; } }
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 'Segoe UI', Arial, sans-serif; color: #1c1c1c; margin: 0; }
h1, h2, h3, h4 { color: #12355b; line-height: 1.25; }
h1 { font-size: 24pt; margin: 0 0 8mm; padding-bottom: 3mm; border-bottom: 3px solid #12355b;
     page-break-before: always; }
h2 { font-size: 16pt; margin: 9mm 0 3mm; padding-bottom: 1.5mm; border-bottom: 1px solid #c8d3e0;
     page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 4mm 0 1.5mm; page-break-after: avoid; }
p { margin: 0 0 2.5mm; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
a { color: #0b5cad; text-decoration: none; }
hr { border: 0; border-top: 1px solid #d5dbe3; margin: 5mm 0; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; background: #eef4fb; border-left: 4px solid #4c86c6; }
blockquote p:last-child { margin-bottom: 0; }
code { font: 9pt Consolas, 'Courier New', monospace; background: #f1f3f6; padding: 0 1.2mm;
       border-radius: 2px; word-wrap: break-word; }
pre { background: #f6f8fa; border: 1px solid #dfe4ea; border-radius: 3px; padding: 3mm;
      margin: 0 0 3.5mm; white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere;
      page-break-inside: auto; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.38; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 4mm; font-size: 9pt; }
th, td { border: 1px solid #c9d1db; padding: 1.4mm 2mm; vertical-align: top; text-align: left;
         word-wrap: break-word; overflow-wrap: anywhere; }
th { background: #e6edf5; color: #12355b; }
tr { page-break-inside: avoid; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { font-size: 34pt; border: 0; page-break-before: avoid; margin-bottom: 4mm; }
.cover .sub { font-size: 14pt; color: #444; margin-bottom: 14mm; }
.cover .meta { color: #666; font-size: 10pt; }
.cover ul { font-size: 12pt; line-height: 1.8; }
.toc h1 { page-break-before: always; }
.toc ul { list-style: none; padding-left: 0; }
.toc li.part { margin-top: 4mm; font-weight: 600; font-size: 11.5pt; }
.toc li.sec { margin-left: 5mm; font-size: 9.6pt; margin-bottom: 0.4mm; }
.toc li.sec.q { font-size: 9.6pt; }
.divider { page-break-before: always; height: 230mm; display: flex; flex-direction: column;
           justify-content: center; align-items: flex-start; }
.divider .label { font-size: 14pt; letter-spacing: 3px; text-transform: uppercase; color: #4c86c6; }
.divider .name { font-size: 30pt; font-weight: 700; color: #12355b; margin-top: 3mm; }
.part-body > h1:first-child { page-break-before: always; }
"""


def slug(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-")
    return text or "section"


def make_parser() -> MarkdownIt:
    # Le préréglage CommonMark ne gère pas les tableaux : on les active.
    return MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")


def strip_local_links(markdown: str) -> str:
    """Remplace les liens vers d'autres fichiers .md par leur texte."""
    return re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]*\.md[^)]*\)", r"\1", markdown)


def break_after_bold_questions(markdown: str) -> str:
    """Saut de ligne après une ligne entièrement en gras (« **12. Question ?** »)."""
    lines = markdown.split("\n")
    in_code = False
    for i, line in enumerate(lines[:-1]):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        if re.match(r"^\*\*.+\*\*( \*\(.*\)\*)?$", line) and lines[i + 1].strip():
            lines[i] = line + "  "
    return "\n".join(lines)


def render_part(md: MarkdownIt, markdown: str, prefix: str, toc: list) -> str:
    tokens = md.parse(break_after_bold_questions(strip_local_links(markdown)))
    used: set[str] = set()

    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        level = int(token.tag[1])
        inline = tokens[i + 1]
        title = inline.content
        base = f"{prefix}-{slug(title)}"
        anchor, n = base, 2
        while anchor in used:
            anchor = f"{base}-{n}"
            n += 1
        used.add(anchor)
        token.attrSet("id", anchor)
        if level == 2:
            toc.append((anchor, title))

    return md.renderer.render(tokens, md.options, {})


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("Aucun navigateur (Edge ou Chrome) trouvé pour produire le PDF.")


def build_html(files: list[str], title: str, subtitle: str) -> str:
    md = make_parser()
    parts_html = []
    toc_by_part = []

    selected = [part for part in PARTS if part[0] in files]
    labels = {}
    for index, (filename, _label, part_subtitle, description) in enumerate(selected):
        labels[filename] = (f"Partie {chr(ord('A') + index)}", part_subtitle, description)

    for filename, _label, _subtitle, _description in selected:
        label, part_subtitle, _description = labels[filename]
        path = DOCS / filename
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        prefix = slug(label)
        toc: list = []
        body = render_part(md, text, prefix, toc)
        divider_id = f"{prefix}-divider"
        parts_html.append(
            f'<div class="divider" id="{divider_id}"><div class="label">{html.escape(label)}</div>'
            f'<div class="name">{html.escape(part_subtitle)}</div></div>'
            f'<div class="part-body">{body}</div>'
        )
        toc_by_part.append((divider_id, label, part_subtitle, toc))

    toc_html = ['<div class="toc"><h1>Sommaire</h1><ul>']
    for divider_id, label, part_subtitle, toc in toc_by_part:
        toc_html.append(
            f'<li class="part"><a href="#{divider_id}">{html.escape(label)} : {html.escape(part_subtitle)}</a></li>'
        )
        for anchor, heading in toc:
            toc_html.append(f'<li class="sec"><a href="#{anchor}">{html.escape(heading)}</a></li>')
    toc_html.append("</ul></div>")

    today = date.today().strftime("%d/%m/%Y")
    items = "".join(
        f"<li><b>{html.escape(labels[name][0])}</b> : {html.escape(labels[name][1])}"
        f" ({html.escape(labels[name][2])})</li>"
        for name in files if name in labels
    )
    cover = f"""
<div class="cover">
  <h1>{html.escape(title)}</h1>
  <div class="sub">{html.escape(subtitle)}</div>
  <ul>{items}</ul>
  <div class="meta">Généré le {today}. Source : dossier <code>docs/</code> du dépôt.</div>
</div>"""

    return (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{CSS}</style></head><body>"
        + cover
        + "".join(toc_html)
        + "".join(parts_html)
        + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS / "guide-complet-rag-llm-entretien.pdf",
        help="Chemin du PDF produit.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(GROUPS),
        default="all",
        help="Sous-ensemble : all (défaut), rag (parties A à D) ou llm (formation et entretien LoRA).",
    )
    parser.add_argument("--title", default=None, help="Titre de la couverture.")
    args = parser.parse_args()

    titles = {
        "all": ("Guide complet RAG, LLM et entretien",
                "Formations, déploiement, questions d'entretien et exercices de code, "
                "avec le projet proposal-rag comme fil rouge."),
        "rag": ("Guide RAG et entretien",
                "Formation RAG, déploiement et questions d'entretien, avec le projet proposal-rag."),
        "llm": ("Transformers, LLM et LoRA",
                "Formation complète et questions d'entretien : de l'attention au fine-tuning "
                "avec LoRA et QLoRA, jusqu'au déploiement."),
    }
    title, subtitle = titles[args.only]
    if args.title:
        title = args.title

    browser = find_browser()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Edge garde quelques fichiers de profil ouverts un instant après sa sortie :
    # un échec de nettoyage du dossier temporaire ne doit pas faire échouer le build.
    output.unlink(missing_ok=True)  # évite de prendre un ancien PDF pour le résultat
    tmp = Path(tempfile.mkdtemp(prefix="guide-pdf-"))
    try:
        html_path = tmp / "guide.html"
        html_path.write_text(build_html(GROUPS[args.only], title, subtitle), encoding="utf-8")

        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--generate-pdf-document-outline",
            f"--user-data-dir={tmp / 'profile'}",
            f"--print-to-pdf={output}",
            html_path.as_uri(),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=300)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not output.exists() or output.stat().st_size == 0:
        sys.exit("Le PDF n'a pas été produit.")

    print(f"PDF écrit : {output} ({output.stat().st_size / 1024:.0f} Ko)")


if __name__ == "__main__":
    main()
