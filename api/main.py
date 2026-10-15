import json
import logging
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from groq import APIError, BadRequestError
from pydantic import BaseModel, Field

from proposal_rag.generation.groq_llm import EmptyLLMResponse, GroqLLM
from proposal_rag.indexes.chroma_store import ChromaStore
from proposal_rag.ingestion.service import DocumentIngestionService
from proposal_rag.rag import ProposalRAG
from proposal_rag.rfp.analyzer import RFPAnalyzer
from proposal_rag.rfp.compliance import ComplianceGenerator
from proposal_rag.rfp.docx_export import markdown_to_docx
from proposal_rag.rfp.proposal_generator import ProposalGenerator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("proposal_rag.api")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document"
)

rag: ProposalRAG | None = None


def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> None:
    """Protéger l'API par une clé partagée.

    Si la variable d'environnement API_KEY n'est pas définie,
    l'authentification est désactivée (développement local).
    """

    expected = os.getenv("API_KEY")

    if not expected:
        return

    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide ou manquante.",
        )


def get_rag() -> ProposalRAG:
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG non initialisé.",
        )

    return rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag

    if not os.getenv("API_KEY"):
        logger.warning(
            "API_KEY n'est pas définie : l'API n'est pas protégée."
        )

    # Chargement des modèles une seule fois au démarrage.
    rag = ProposalRAG(llm=GroqLLM())

    yield

    rag = None


app = FastAPI(
    title="Proposal RAG API",
    description="RAG pour répondre aux appels d'offres.",
    version="1.1.0",
    lifespan=lifespan,
)

protected = [Depends(require_api_key)]


def read_upload(file: UploadFile) -> tuple[str, bytes]:
    """Valider un fichier envoyé et renvoyer (nom, contenu)."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Nom de fichier manquant.",
        )

    filename = Path(file.filename.replace("\\", "/")).name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Format non supporté.",
        )

    content = file.file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Fichier trop volumineux (20 MB maximum).",
        )

    return filename, content


# --------------------------------------------------
# Santé
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Questions / réponses
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    id: str
    document: str
    section: str = ""
    page: int | None = None
    score: float
    excerpt: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = []


@app.post(
    "/ask",
    response_model=QuestionResponse,
    dependencies=protected,
)
def ask_question(request: QuestionRequest):
    engine = get_rag()

    try:
        result = engine.ask_with_sources(
            question=request.question,
            top_k=request.top_k,
            candidate_k=max(10, request.top_k),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (APIError, EmptyLLMResponse) as exc:
        logger.exception("Erreur du fournisseur LLM")
        raise HTTPException(
            status_code=502,
            detail="Le service de génération est indisponible.",
        ) from exc

    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
    )


# --------------------------------------------------
# Documents
# --------------------------------------------------

@app.post("/documents/upload", dependencies=protected)
def upload_document(file: UploadFile = File(...)):
    engine = get_rag()
    filename, content = read_upload(file)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / filename

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail="Un document avec ce nom existe déjà.",
        )

    destination.write_bytes(content)

    try:
        dense = engine.retrieval.hybrid_retriever.dense_retriever

        # Réutiliser l'embedder déjà chargé en mémoire.
        service = DocumentIngestionService(
            embedder=dense.embedder,
            store=dense.store,
        )
        result = service.index(
            path=destination,
            source=filename,
        )
    except ValueError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception:
        destination.unlink(missing_ok=True)
        logger.exception("Échec de l'indexation de %s", filename)
        raise

    # Le nouveau document doit aussi être visible par BM25.
    engine.retrieval.refresh()

    logger.info("Document indexé : %s", filename)
    return result


@app.get("/documents", dependencies=protected)
def list_documents():
    documents = ChromaStore().list_documents()

    return {
        "total": len(documents),
        "documents": documents,
    }


@app.delete("/documents/{filename}", dependencies=protected)
def delete_document(filename: str):
    engine = get_rag()

    # Les documents sont stockés directement dans data/raw.
    if (
        filename != Path(filename).name
        or "/" in filename
        or "\\" in filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Nom de fichier invalide.",
        )

    deleted_chunks = ChromaStore().delete_document(filename)

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="Document introuvable dans l'index.",
        )

    (RAW_DIR / filename).unlink(missing_ok=True)
    engine.retrieval.refresh()

    logger.info("Document supprimé : %s", filename)

    return {
        "message": "Document supprimé.",
        "filename": filename,
        "deleted_chunks": deleted_chunks,
    }


# --------------------------------------------------
# Appels d'offres
# --------------------------------------------------

@app.post("/rfp/analyze", dependencies=protected)
def analyze_rfp(file: UploadFile = File(...)):
    filename, content = read_upload(file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        path.write_bytes(content)

        try:
            return RFPAnalyzer().analyze(path)

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        except BadRequestError as exc:
            raise HTTPException(
                status_code=502,
                detail="Groq n'a pas réussi à générer un JSON valide.",
            ) from exc


class ComplianceRequest(BaseModel):
    requirements: list[dict]


@app.post("/rfp/compliance", dependencies=protected)
def generate_compliance_matrix(request: ComplianceRequest):
    engine = get_rag()

    if not request.requirements:
        raise HTTPException(
            status_code=400,
            detail="Aucune exigence fournie.",
        )

    generator = ComplianceGenerator(
        llm=engine.generator.llm,
        retrieval=engine.retrieval,
    )

    try:
        return generator.generate_matrix(request.requirements)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except (APIError, EmptyLLMResponse) as exc:
        logger.exception("Erreur du fournisseur LLM")
        raise HTTPException(
            status_code=502,
            detail="Le service de génération est indisponible.",
        ) from exc


class ProposalRequest(BaseModel):
    results: list[dict]


@app.post("/rfp/proposal", dependencies=protected)
def generate_proposal(request: ProposalRequest):
    engine = get_rag()

    if not request.results:
        raise HTTPException(
            status_code=400,
            detail="La matrice est vide.",
        )

    try:
        proposal = ProposalGenerator(
            llm=engine.generator.llm
        ).generate(request.results)
    except (APIError, EmptyLLMResponse) as exc:
        logger.exception("Erreur du fournisseur LLM")
        raise HTTPException(
            status_code=502,
            detail="Le service de génération est indisponible.",
        ) from exc

    return {
        "proposal": proposal,
        "requirements_count": len(request.results),
    }


class ExportRequest(BaseModel):
    proposal: str = Field(min_length=1)
    title: str | None = None


@app.post("/rfp/proposal/export", dependencies=protected)
def export_proposal(request: ExportRequest):
    try:
        content = markdown_to_docx(
            request.proposal,
            title=request.title,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return Response(
        content=content,
        media_type=DOCX_MEDIA_TYPE,
        headers={
            "Content-Disposition": (
                'attachment; filename="technical_proposal.docx"'
            )
        },
    )
