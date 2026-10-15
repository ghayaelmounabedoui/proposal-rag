# Proposal RAG

Assistant pour répondre aux appels d'offres (RFP) à partir des documents de l'entreprise.
Chaque réponse est **appuyée par des sources citées** : le système n'affirme rien qui ne soit pas dans les documents indexés.

## Fonctionnalités

- **Ingestion** de documents PDF, DOCX, Markdown et texte, découpés par titres et paragraphes complets.
- **Recherche hybride** : vectorielle (ChromaDB + `multilingual-e5-small`) et mots-clés (BM25), fusionnées par Reciprocal Rank Fusion, puis reclassées par un cross-encoder.
- **Questions/réponses** avec citations `[SOURCE X]` et liste des passages utilisés (document, section, extrait).
- **Analyse d'un RFP** : extraction des exigences (catégorie, caractère obligatoire, extrait vérifié dans le texte source).
- **Matrice de conformité** : `COMPLIANT` / `PARTIAL` / `NOT_DEMONSTRATED`, avec preuves. Un statut `COMPLIANT` sans preuve valide est automatiquement rétrogradé.
- **Proposition technique** générée depuis la matrice, exportable en Markdown et **Word (.docx)**.
- **Interface Streamlit** et **API FastAPI** (clé API optionnelle).

## Architecture

```
Streamlit (8501) ──HTTP──▶ FastAPI (8000) ──▶ ProposalRAG
                                                │
              ┌─────────────────────────────────┼───────────────────────────┐
              ▼                                 ▼                           ▼
        Ingestion                       Recherche hybride              Génération (Groq)
  parsers pdf/docx/md → chunks     dense (Chroma) + BM25 → RRF      contexte + prompts
  → embeddings → ChromaDB          → reranker cross-encoder         → réponse citée
```

| Dossier | Rôle |
|---|---|
| `src/proposal_rag/ingestion` | Parsers, découpage en chunks, service d'indexation |
| `src/proposal_rag/retrieval` | Dense, BM25, fusion RRF, reranker |
| `src/proposal_rag/generation` | Contexte, prompts, client Groq |
| `src/proposal_rag/rfp` | Analyse du RFP, matrice de conformité, proposition, export Word |
| `api/` | API FastAPI |
| `frontend/` | Interface Streamlit |
| `scripts/` | Outils en ligne de commande (indexation, recherche, évaluation) |
| `data/samples` | Documents **fictifs** d'exemple |
| `data/eval` | Questions de référence pour mesurer la recherche |

## Démarrage rapide (Docker)

Prérequis : Docker Desktop et une clé [Groq](https://console.groq.com).

```bash
cp .env.example .env        # puis renseigner GROQ_API_KEY
docker compose up --build
```

- Interface : <http://127.0.0.1:8501>
- API et documentation interactive : <http://127.0.0.1:8000/docs>

Le premier build est long (dépendances ML) ; les modèles sont intégrés à l'image, le démarrage ne télécharge rien.

Pour essayer avec des données d'exemple : importer les fichiers de `data/samples` depuis la page **Documents**, puis analyser `data/raw/sample_rfp.md` depuis la page **Analyse RFP**.

## Documentation

| Document | Contenu |
|---|---|
| [`docs/formation-rag.md`](docs/formation-rag.md) | Cours RAG complet, de la théorie au code de ce projet |
| [`docs/entretien-rag.md`](docs/entretien-rag.md) | 142 questions techniques d'entretien avec réponses |
| [`docs/entretien-projet-rag.md`](docs/entretien-projet-rag.md) | Questions sur le projet, conception de systèmes, exercices de code |
| [`docs/formation-deploiement.md`](docs/formation-deploiement.md) | Formation au déploiement, de A à Z |

## Déploiement en production

La configuration de production est dans `docker-compose.prod.yml` : seul Caddy (reverse proxy) est exposé,
avec HTTPS automatique et mot de passe ; l'API et l'interface restent sur le réseau interne Docker.

```bash
cp .env.example .env                  # GROQ_API_KEY, API_KEY, DOMAIN
cp Caddyfile.example Caddyfile        # y coller le hash du mot de passe (voir le fichier)
docker compose -f docker-compose.prod.yml up -d --build
```

Sauvegardes : `scripts/backup.sh`. Le guide complet, de zéro (réseau, Docker, HTTPS, CI/CD, entreprise)
est dans [`docs/formation-deploiement.md`](docs/formation-deploiement.md).
Cette configuration a été validée en local (HTTPS, authentification, ports internes fermés) ; elle n'est pas laissée en ligne.

## Développement local

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows (Linux/macOS : source .venv/bin/activate)
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

export GROQ_API_KEY=...                     # PowerShell : $env:GROQ_API_KEY="..."
python -m uvicorn api.main:app --reload     # API
streamlit run frontend/app.py               # interface
```

Indexer un document en ligne de commande (il doit être dans `data/raw`) :

```bash
PYTHONPATH=src python scripts/index_document.py data/raw/mon_document.pdf
```

## Configuration

| Variable | Défaut | Description |
|---|---|---|
| `GROQ_API_KEY` | — (obligatoire) | Clé du fournisseur LLM |
| `API_KEY` | vide | Si définie, l'API exige l'en-tête `X-API-Key` (sauf `/health`) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Modèle de génération |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder de reclassement |
| `LOG_LEVEL` | `INFO` | Niveau de journalisation |
| `PROPOSAL_RAG_API_URL` | `http://127.0.0.1:8000` | URL de l'API, côté interface |

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Santé (publique) |
| POST | `/ask` | Question → réponse + sources |
| POST | `/documents/upload` | Importer et indexer un document |
| GET | `/documents` | Lister les documents indexés |
| DELETE | `/documents/{filename}` | Supprimer un document et son index |
| POST | `/rfp/analyze` | Extraire les exigences d'un RFP |
| POST | `/rfp/compliance` | Générer la matrice de conformité |
| POST | `/rfp/proposal` | Générer la proposition (Markdown) |
| POST | `/rfp/proposal/export` | Convertir la proposition en `.docx` |

## Tests et qualité

```bash
pytest -q                                   # 45 tests, sans réseau ni modèle
ruff check .                                # lint
PYTHONPATH=src python scripts/evaluate.py --k 3 --min-hit-rate 0.8   # qualité de la recherche
```

`scripts/evaluate.py` mesure hit@1, hit@k et MRR sur `data/eval/questions.json`. Il nécessite un index rempli (documents de `data/samples`) et les modèles locaux. **Ajoutez vos propres questions** : le jeu fourni est petit et facile.

La CI GitHub (`.github/workflows/ci.yml`) exécute lint, tests et build des images à chaque push.

## Sécurité

- Ne jamais committer `.env` (ignoré par git) ; utiliser `.env.example` comme modèle.
- Définir `API_KEY` dès que l'API est accessible autrement qu'en local.
- Les conteneurs tournent sans privilèges administrateur et les ports sont liés à `127.0.0.1`.
- Les documents envoyés à `/ask`, `/rfp/*` sont transmis à Groq (service tiers) : vérifier la politique de confidentialité avant d'indexer des documents sensibles.

## Limites connues

- Le reranker par défaut est entraîné sur de l'anglais ; pour des corpus surtout français, configurer `RERANKER_MODEL` avec un modèle multilingue et le valider avec `scripts/evaluate.py`.
- Un seul index partagé : pas de séparation par client ou par utilisateur.
- Les appels au LLM sont synchrones : une matrice de conformité sur un gros RFP peut prendre plusieurs minutes.
- Les dépendances directes sont figées, pas les dépendances transitives.
