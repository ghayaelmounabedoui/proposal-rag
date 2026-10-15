from collections import Counter
from pathlib import Path

import pymupdf

from proposal_rag.models import DocumentBlock


def parse_pdf_document(path: Path) -> list[DocumentBlock]:
    """Extraire des titres et des blocs de paragraphes depuis un PDF."""

    pages = []
    font_weights = Counter()

    with pymupdf.open(str(path)) as document:
        for page in document:
            data = page.get_text(
                "dict",
                sort=True,
                flags=pymupdf.TEXTFLAGS_TEXT,
            )
            pages.append(data)

            for block in data["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        text = span["text"].strip()

                        if text:
                            size = round(span["size"], 1)
                            font_weights[size] += len(text)

    if not font_weights:
        raise ValueError(
            "Aucun texte extrait : ce PDF peut nécessiter un OCR."
        )

    # Taille la plus représentée, pondérée par le nombre de caractères.
    body_size = font_weights.most_common(1)[0][0]

    # Les plus grandes tailles correspondent aux titres les plus hauts.
    heading_sizes = sorted(
        [
            size
            for size in font_weights
            if size > body_size * 1.15
        ],
        reverse=True,
    )

    heading_levels = {
        size: min(index + 1, 6)
        for index, size in enumerate(heading_sizes)
    }

    blocks: list[DocumentBlock] = []

    for page_number, data in enumerate(pages, start=1):
        for pdf_block in data["blocks"]:
            paragraph_lines = []

            def flush_paragraph() -> None:
                if paragraph_lines:
                    blocks.append(
                        DocumentBlock(
                            text=" ".join(paragraph_lines),
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
                    paragraph_lines.clear()

            for line in pdf_block.get("lines", []):
                spans = [
                    span
                    for span in line["spans"]
                    if span["text"].strip()
                ]

                if not spans:
                    continue

                text = "".join(
                    span["text"] for span in spans
                ).strip()

                # Prendre la taille du segment contenant le plus de texte.
                main_span = max(
                    spans,
                    key=lambda span: len(span["text"].strip()),
                )
                size = round(main_span["size"], 1)

                heading_level = heading_levels.get(size, 0)

                if heading_level:
                    flush_paragraph()

                    blocks.append(
                        DocumentBlock(
                            text=text,
                            heading_level=heading_level,
                            page_start=page_number,
                            page_end=page_number,
                        )
                    )
                else:
                    paragraph_lines.append(text)

            flush_paragraph()

    return blocks