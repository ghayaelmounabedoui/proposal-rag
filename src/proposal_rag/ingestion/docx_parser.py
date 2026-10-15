import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from proposal_rag.models import DocumentBlock

HEADING_PATTERN = re.compile(
    r"^(?:Heading|Titre)\s*([1-9])$",
    re.IGNORECASE,
)


def get_heading_level(paragraph: Paragraph) -> int:
    """Trouver le niveau du titre dans son style ou un style parent."""

    style = paragraph.style

    while style is not None:
        match = HEADING_PATTERN.match(style.name or "")

        if match:
            return int(match.group(1))

        style = style.base_style

    return 0


def parse_docx_document(path: Path) -> list[DocumentBlock]:
    """Extraire les titres, paragraphes et tableaux du corps du document."""

    document = Document(str(path))
    blocks: list[DocumentBlock] = []

    for element in document.iter_inner_content():
        if isinstance(element, Paragraph):
            text = element.text.strip()

            if text:
                blocks.append(
                    DocumentBlock(
                        text=text,
                        heading_level=get_heading_level(element),
                    )
                )

        elif isinstance(element, Table):
            rows = []

            for row in element.rows:
                cells = [
                    cell.text.strip().replace("\n", " ; ")
                    for cell in row.cells
                ]

                if any(cells):
                    rows.append(" | ".join(cells))

            if rows:
                blocks.append(
                    DocumentBlock(text="\n".join(rows))
                )

    return blocks
