import json

import pytest

from proposal_rag.rfp.compliance import ComplianceGenerator

from .conftest import FakeLLM, FakeRetrieval, make_result


def generator(reply, results=None):
    results = [make_result()] if results is None else results
    return ComplianceGenerator(
        llm=FakeLLM(reply),
        retrieval=FakeRetrieval(results),
    )


REQ = {"id": "REQ-001", "requirement": "Chiffrement AES-256"}


def payload(**overrides):
    data = {
        "status": "COMPLIANT",
        "answer": "Documenté.",
        "evidence": ["[SOURCE 1]"],
    }
    data.update(overrides)
    return json.dumps(data)


def test_compliant_with_valid_evidence():
    item = generator(payload()).evaluate(REQ)

    assert item["status"] == "COMPLIANT"
    assert item["evidence"] == ["[SOURCE 1]"]
    assert item["id"] == "REQ-001"


def test_invented_source_ids_are_dropped_and_status_downgraded():
    item = generator(payload(evidence=["[SOURCE 9]"])).evaluate(REQ)

    assert item["evidence"] == []
    assert item["status"] == "NOT_DEMONSTRATED"


def test_compliant_without_evidence_is_downgraded():
    item = generator(payload(evidence=[])).evaluate(REQ)

    assert item["status"] == "NOT_DEMONSTRATED"


def test_partial_is_kept_without_evidence():
    item = generator(payload(status="PARTIAL", evidence=[])).evaluate(REQ)

    assert item["status"] == "PARTIAL"


def test_markdown_fenced_json_is_accepted():
    item = generator("```json\n" + payload() + "\n```").evaluate(REQ)

    assert item["status"] == "COMPLIANT"


def test_invalid_status_raises():
    with pytest.raises(ValueError):
        generator(payload(status="MAYBE")).evaluate(REQ)


def test_no_documents_means_not_demonstrated_without_llm_call():
    gen = generator(payload(), results=[])

    item = gen.evaluate(REQ)

    assert item["status"] == "NOT_DEMONSTRATED"
    assert gen.llm.calls == []


def test_empty_requirement_raises():
    with pytest.raises(ValueError):
        generator(payload()).evaluate({"id": "x", "requirement": " "})


def test_matrix_requires_requirements():
    with pytest.raises(ValueError):
        generator(payload()).generate_matrix([])
