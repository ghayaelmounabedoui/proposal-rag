import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from proposal_rag.ingestion.chunker import chunk_document
from proposal_rag.ingestion.text_parser import parse_text_document


DOCUMENT = """# Sécurité

## Protection des données

Les données sont chiffrées au repos avec AES-256.
Les communications utilisent TLS 1.3.

Les sauvegardes sont réalisées quotidiennement.
Elles sont conservées pendant trente jours.

## Gestion des accès

Les accès privilégiés nécessitent une authentification multifacteur.

# Continuité de service

Un exercice de reprise est réalisé tous les six mois.
"""


class IngestionTests(unittest.TestCase):
    def test_paragraphs_and_heading_hierarchy(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "security.md"
            path.write_text(DOCUMENT, encoding="utf-8")

            blocks = parse_text_document(path)
            chunks = chunk_document(blocks, source="security.md")

        self.assertEqual(len(chunks), 4)

        self.assertEqual(
            chunks[0].heading_path,
            ("Sécurité", "Protection des données"),
        )

        self.assertEqual(
            chunks[0].content,
            "Les données sont chiffrées au repos avec AES-256.\n"
            "Les communications utilisent TLS 1.3.",
        )

        self.assertEqual(
            chunks[2].heading_path,
            ("Sécurité", "Gestion des accès"),
        )

        self.assertEqual(
            chunks[3].heading_path,
            ("Continuité de service",),
        )

        # Vérifier que chaque passage possède un ID distinct.
        self.assertEqual(len({c.chunk_id for c in chunks}), 4)

        # Vérifier que les IDs restent identiques à contenu égal.
        repeated = chunk_document(blocks, source="security.md")
        self.assertEqual(chunks, repeated)


if __name__ == "__main__":
    unittest.main()