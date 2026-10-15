import io

import api.main as main


def test_health_is_public(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"question": "Chiffrement ?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Réponse à : Chiffrement ?"
    assert body["sources"][0]["id"] == "[SOURCE 1]"
    assert body["sources"][0]["document"] == "security.md"
    assert body["sources"][0]["section"] == "Sécurité > Chiffrement"


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert (
        client.post("/ask", json={"question": "x", "top_k": 99}).status_code
        == 422
    )


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    payload = {"question": "test"}

    assert client.post("/ask", json=payload).status_code == 401
    assert (
        client.post(
            "/ask", json=payload, headers={"X-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/ask", json=payload, headers={"X-API-Key": "secret"}
        ).status_code
        == 200
    )


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/documents/upload",
        files={"file": ("virus.exe", b"data")},
    )
    assert response.status_code == 400


def test_upload_indexes_and_refreshes_bm25(client, fake_rag, monkeypatch):
    class FakeService:
        def __init__(self, embedder, store):
            pass

        def index(self, path, source):
            return {"source": source, "chunks_indexed": 2, "total_chunks": 2}

    monkeypatch.setattr(main, "DocumentIngestionService", FakeService)

    response = client.post(
        "/documents/upload",
        files={"file": ("notes.md", io.BytesIO(b"# Titre\n\nTexte"))},
    )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 2
    assert fake_rag.retrieval.refresh_count == 1
    assert (main.RAW_DIR / "notes.md").exists()

    # Un second envoi du même nom est refusé.
    duplicate = client.post(
        "/documents/upload",
        files={"file": ("notes.md", io.BytesIO(b"autre"))},
    )
    assert duplicate.status_code == 409


def test_upload_failure_removes_file(client, monkeypatch):
    class BrokenService:
        def __init__(self, embedder, store):
            pass

        def index(self, path, source):
            raise ValueError("Aucun contenu à indexer.")

    monkeypatch.setattr(main, "DocumentIngestionService", BrokenService)

    response = client.post(
        "/documents/upload",
        files={"file": ("empty.md", io.BytesIO(b""))},
    )

    assert response.status_code == 422
    assert not (main.RAW_DIR / "empty.md").exists()


def test_delete_rejects_path_traversal(client):
    response = client.delete("/documents/..%2F..%2F.env")
    assert response.status_code in (400, 404)


def test_delete_unknown_document(client, monkeypatch):
    class FakeStore:
        def delete_document(self, source):
            return 0

    monkeypatch.setattr(main, "ChromaStore", FakeStore)
    assert client.delete("/documents/absent.md").status_code == 404


def test_delete_refreshes_bm25(client, fake_rag, monkeypatch):
    class FakeStore:
        def delete_document(self, source):
            return 3

    monkeypatch.setattr(main, "ChromaStore", FakeStore)

    response = client.delete("/documents/security.md")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 3
    assert fake_rag.retrieval.refresh_count == 1


def test_compliance_requires_requirements(client):
    response = client.post("/rfp/compliance", json={"requirements": []})
    assert response.status_code == 400


def test_proposal_requires_results(client):
    response = client.post("/rfp/proposal", json={"results": []})
    assert response.status_code == 400


def test_proposal_generation(client):
    response = client.post(
        "/rfp/proposal",
        json={
            "results": [
                {
                    "id": "REQ-001",
                    "requirement": "AES-256",
                    "status": "COMPLIANT",
                    "answer": "ok",
                    "evidence": ["[SOURCE 1]"],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["requirements_count"] == 1


def test_export_proposal_returns_docx(client):
    response = client.post(
        "/rfp/proposal/export",
        json={"proposal": "# Titre\n\n- point **important**"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "application/vnd.openxmlformats"
    )
    # Un .docx est une archive ZIP.
    assert response.content[:2] == b"PK"


def test_export_rejects_empty_proposal(client):
    response = client.post("/rfp/proposal/export", json={"proposal": ""})
    assert response.status_code == 422


def test_proposal_returns_502_when_llm_returns_nothing(client, fake_rag):
    from proposal_rag.generation.groq_llm import EmptyLLMResponse

    def broken(system_prompt, user_prompt):
        raise EmptyLLMResponse("vide")

    fake_rag.generator.llm.generate = broken

    response = client.post(
        "/rfp/proposal",
        json={"results": [{"id": "REQ-001", "requirement": "x", "status": "COMPLIANT"}]},
    )

    assert response.status_code == 502
