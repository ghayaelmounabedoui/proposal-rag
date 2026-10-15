from pathlib import Path

from proposal_rag.ingestion.docx_parser import parse_docx_document
from proposal_rag.ingestion.pdf_parser import parse_pdf_document
from proposal_rag.ingestion.text_parser import parse_text_document
from proposal_rag.models import DocumentBlock


def load_document(path: str | Path) -> list[DocumentBlock]:
    """Choisir le parser correspondant au document."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Document introuvable : {path}")

    extension = path.suffix.lower()

    if extension in {".md", ".txt"}:
        return parse_text_document(path)

    if extension == ".docx":
        return parse_docx_document(path)

    if extension == ".pdf":
        return parse_pdf_document(path)

    raise ValueError(
        f"Format non pris en charge : {extension}. "
        "Formats disponibles : .md, .txt, .docx, .pdf"
    )
