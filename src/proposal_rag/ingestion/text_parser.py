import re
from pathlib import Path

from proposal_rag.models import DocumentBlock


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def parse_text_document(path: Path) -> list[DocumentBlock]:
    """Lire un fichier Markdown ou TXT en conservant ses paragraphes."""

    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Format non pris en charge : {path.suffix}")

    content = path.read_text(encoding="utf-8-sig")

    blocks: list[DocumentBlock] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        """Enregistrer le paragraphe en cours, puis vider le tampon."""

        if paragraph_lines:
            blocks.append(
                DocumentBlock(
                    text="\n".join(paragraph_lines)
                )
            )
            paragraph_lines.clear()

    for line in content.splitlines():
        stripped = line.strip()

        # Une ligne vide termine le paragraphe.
        if not stripped:
            flush_paragraph()
            continue

        heading_match = HEADING_PATTERN.match(stripped)

        if heading_match:
            # Un titre termine également le paragraphe précédent.
            flush_paragraph()

            blocks.append(
                DocumentBlock(
                    text=heading_match.group(2).strip(),
                    heading_level=len(heading_match.group(1)),
                )
            )
        else:
            paragraph_lines.append(stripped)

    # Enregistrer le dernier paragraphe, même sans ligne vide finale.
    flush_paragraph()

    return blocks