from proposal_rag.rfp.analyzer import MAX_BATCH_CHARS, RFPAnalyzer

RFP = """# Appel d'offres

Le système doit chiffrer les données avec AES-256.

Le système doit chiffrer les données avec AES-256.

Une sauvegarde quotidienne est obligatoire.
"""


def make_analyzer(monkeypatch, extracted):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    analyzer = RFPAnalyzer()
    monkeypatch.setattr(
        analyzer, "_analyze_batch", lambda batch: list(extracted)
    )
    return analyzer


def test_long_blocks_are_split_without_losing_text(monkeypatch):
    analyzer = make_analyzer(monkeypatch, [])
    long_block = "a" * (MAX_BATCH_CHARS * 2 + 10)

    batches = analyzer._split_batches([long_block, "fin"])

    assert all(len(b) <= MAX_BATCH_CHARS + 2 for b in batches)
    assert "".join(batches).count("a") == len(long_block)


def test_requirements_without_proof_in_text_are_rejected(
    monkeypatch, tmp_path
):
    path = tmp_path / "rfp.md"
    path.write_text(RFP, encoding="utf-8")

    analyzer = make_analyzer(
        monkeypatch,
        [
            {
                "category": "Security",
                "requirement": "Chiffrement AES-256",
                "mandatory": True,
                "source_excerpt": "chiffrer les données avec AES-256",
            },
            {
                "category": "Security",
                "requirement": "Exigence inventée",
                "mandatory": True,
                "source_excerpt": "texte qui n'existe pas",
            },
            "pas un dictionnaire",
        ],
    )

    result = analyzer.analyze(path)

    assert result["total_requirements"] == 1
    assert result["requirements"][0]["id"] == "REQ-001"
    assert result["requirements"][0]["mandatory"] is True


def test_duplicates_are_removed_and_mandatory_must_be_true(
    monkeypatch, tmp_path
):
    path = tmp_path / "rfp.md"
    path.write_text(RFP, encoding="utf-8")

    item = {
        "category": "Operations",
        "requirement": "Sauvegarde  quotidienne",
        "mandatory": "true",
        "source_excerpt": "Une sauvegarde quotidienne est obligatoire.",
    }
    analyzer = make_analyzer(monkeypatch, [item, dict(item)])

    result = analyzer.analyze(path)

    assert result["total_requirements"] == 1
    # La chaîne "true" n'est pas le booléen True.
    assert result["requirements"][0]["mandatory"] is False
