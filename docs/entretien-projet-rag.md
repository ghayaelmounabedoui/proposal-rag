# Entretien RAG : ton projet, conception de systèmes, pièges et code

Suite de [`entretien-rag.md`](entretien-rag.md) (questions techniques). Cours complet :
[`formation-rag.md`](formation-rag.md).

## Sommaire
1. Présenter le projet (30 secondes, 2 minutes, démo)
2. Questions sur ton projet
3. Conception de systèmes (7 scénarios)
4. Questions pièges
5. Exercices de code (avec solutions vérifiées)
6. Raconter une histoire (méthode STAR)
7. Les questions à poser au recruteur
8. Les erreurs à éviter

---

## 1. Présenter le projet

### La version de 30 secondes
« J'ai construit un assistant qui aide à répondre aux **appels d'offres** à partir des documents de
l'entreprise. Il retrouve les passages pertinents avec une recherche **hybride** (vectorielle +
mots-clés, reclassée), génère des réponses **avec citations**, et produit une **matrice de
conformité** et une proposition technique. Chaque sortie du modèle est **vérifiée en code**. »

### La version de 2 minutes
1. **Le problème** (15 s) : répondre à un appel d'offres demande de retrouver des preuves dans des
   dizaines de documents ; un LLM seul invente et ne cite pas.
2. **La solution** (45 s) : ingestion de PDF/DOCX/Markdown découpés par titres et paragraphes ;
   index vectoriel (ChromaDB, embeddings multilingues) et lexical (BM25) fusionnés par RRF ; reranking
   par cross-encoder ; génération contrainte au contexte avec citations `[SOURCE n]`.
3. **Ce qui la rend fiable** (30 s) : un `COMPLIANT` sans preuve valide est rétrogradé, les sources
   inventées sont supprimées, un extrait cité doit exister dans le texte, et sans passage le modèle
   n'est même pas appelé.
4. **L'ingénierie** (20 s) : API FastAPI, interface Streamlit, Docker avec HTTPS et authentification,
   53 tests, CI GitHub.
5. **Mon regard critique** (10 s) : « Mon jeu d'évaluation est encore petit ; ma priorité serait de
   l'étoffer pour prouver chaque brique et améliorer le français du reranker. »

### La démo de 5 minutes (prépare les fichiers à l'avance)
1. **Documents** : importer 2-3 fichiers de `data/samples`. Montrer le nombre de chunks.
2. **Question** : « Quel est l'engagement de disponibilité ? » → réponse avec `[SOURCE 1]`, ouvrir le
   bloc **Sources** (document, section, extrait).
3. **Question sans réponse** (ex. une information absente) → le système refuse au lieu d'inventer.
4. **Analyse d'un RFP** → tableau d'exigences.
5. **Matrice de conformité** → montrer `COMPLIANT` / `PARTIAL` / `NOT_DEMONSTRATED` avec les preuves.
6. **Proposition** → export Word.
7. Conclure par une limite honnête et la suite prévue.

Prévois un **plan B** : vidéo enregistrée si le réseau ou le fournisseur LLM tombe.

---

## 2. Questions sur ton projet

**1. Quel problème résout-il ?**
Aider à répondre aux appels d'offres : extraire les exigences, retrouver les preuves dans les documents
de l'entreprise, rédiger une réponse sourcée. Le gain visé est le temps de recherche, avec une
relecture humaine à la fin.

**2. Décris l'architecture.**
Parsing → chunks par paragraphe avec chemin de titres → embeddings E5 multilingue (384 dim) → ChromaDB.
À la question : dense + BM25 → RRF → reranking → contexte numéroté → LLM → réponse avec citations.
API FastAPI, interface Streamlit, le tout en Docker.

**3. Pourquoi une recherche hybride ? L'as-tu prouvé ?**
Pour couvrir le sens (embeddings) et les mots exacts comme `AES-256` (BM25). Honnêtement : sur mon petit
jeu de 10 questions, tout est à 100 %, donc **je n'ai pas encore de chiffre qui le prouve**. L'étape
suivante est une ablation sur un jeu plus difficile (dense, BM25, hybride, hybride + rerank).

**4. Pourquoi RRF plutôt qu'une somme de scores ?**
Les scores n'ont pas la même échelle (distance cosinus contre score BM25). RRF utilise seulement les
rangs, sans normalisation : plus simple et plus robuste.

**5. Pourquoi un chunk par paragraphe ?**
Un paragraphe porte une idée complète, garde un sens autonome, et se cite proprement. C'est adapté à
des documents structurés. Limite connue : pas de plafond de taille, un paragraphe de plus de 512 tokens
fait échouer l'indexation. Je le découperais par phrases avec chevauchement.

**6. Pourquoi `multilingual-e5-small` ?**
Documents et questions en français et en anglais, modèle léger qui tourne sur CPU, coût nul. Je ne l'ai
pas comparé à d'autres sur mes données : c'est une décision de prototype que je réévaluerais avec un
jeu plus grand.

**7. Raconte un bug que tu as trouvé.**
Deux exemples. (a) Les questions étaient encodées avec le préfixe `passage:` au lieu de `query:`
demandé par E5 : invisible dans mon évaluation à 100 %, trouvé en relisant le code. (b) BM25, en
mémoire, n'était pas rafraîchi après un upload ou une suppression : un document supprimé restait
trouvable. J'ai ajouté `refresh()` et des tests. (Voir la méthode STAR, section 6.)

**8. Comment gères-tu les hallucinations ?**
En couches : recherche de qualité, prompt contraignant avec repli explicite, citations obligatoires,
**vérification en code** (sources existantes, extrait présent dans le texte, `COMPLIANT` sans preuve
rétrogradé), refus sans appel au LLM quand aucun passage n'est trouvé, relecture humaine. Je ne dis pas
que le risque est nul.

**9. Comment fonctionne la matrice de conformité ?**
Pour chaque exigence : recherche (top 3 sur 10 candidats), le LLM renvoie un JSON (statut, réponse,
preuves), je valide en code, puis je retire les preuves qui ne figurent pas dans le contexte réellement
fourni. Sans preuve valide, `COMPLIANT` devient `NOT_DEMONSTRATED`.

**10. Que se passe-t-il si le RFP contient « ignore tes instructions » ?**
Les prompts disent de traiter le texte comme des données, et le code valide les sorties : un statut hors
liste est rejeté, les sources inventées sont supprimées. Ce n'est pas une garantie absolue : le système
ne déclenche aucune action et un humain relit. Je l'ai noté comme exercice à tester.

**11. Comment évalues-tu ?**
`scripts/evaluate.py` mesure hit@1, hit@k et MRR sur 10 questions. Limite assumée : trop petit et trop
facile. Plus de tests unitaires sur la logique déterministe (53), mais l'évaluation de la **fidélité** des
réponses n'est pas automatisée.

**12. Combien de documents supporte-t-il ?**
C'est un prototype : ChromaDB local et BM25 reconstruit en mémoire. Je n'ai **pas** testé à grande
échelle. Pour des millions de chunks, je passerais à une base vectorielle serveur et à un moteur BM25
dédié.

**13. Comment gères-tu la confidentialité ?**
Les questions et passages partent chez le fournisseur du LLM (Groq). Pour des données sensibles, il
faut un accord de traitement, un fournisseur européen ou un modèle auto-hébergé. Mon code isole le LLM
derrière une interface pour que ce changement soit simple.

**14. Comment as-tu déployé et sécurisé ?**
Docker avec images non-root, reverse proxy Caddy (HTTPS, mot de passe), clé API, ports internes fermés,
healthchecks, sauvegardes. J'ai validé la configuration de production **en local** ; je ne l'ai pas
laissée en ligne.

**15. Comment testes-tu sans appeler le LLM ?**
Des faux (LLM, recherche) injectés par interface : 53 tests en quelques secondes, sans réseau ni modèle.
La CI exécute lint, tests et build des images.

**16. Pourquoi Groq et ce modèle ?**
Rapide et simple à intégrer pour un prototype. Le modèle est un modèle de **raisonnement**, ce qui m'a
valu des réponses vides quand la limite de tokens était trop basse : j'ai fixé une limite, ajouté des
tentatives et une erreur 502 propre.

**17. Pourquoi pas LangChain ou LlamaIndex ?**
J'ai écrit les briques moi-même pour comprendre et contrôler chaque étape (fusion, validation, erreurs).
Ces frameworks font gagner du temps pour aller vite, mais ajoutent de l'abstraction et des dépendances
qu'il faut comprendre pour déboguer. Je peux utiliser l'un ou l'autre ; l'important est de savoir ce qu'ils
font dessous.

**18. Quelle est la limite la plus importante ?**
L'évaluation : sans jeu difficile, je ne peux pas **prouver** l'apport de chaque brique ni détecter une
régression fine. Viennent ensuite le reranker anglais, l'absence de mémoire de conversation et l'absence
de droits par utilisateur.

**19. Que ferais-tu avec deux semaines de plus ?**
1) Jeu d'évaluation de 50+ questions réelles et ablation chiffrée. 2) Découpage des longs paragraphes.
3) Reranker multilingue validé par la mesure. 4) Condensation de l'historique de conversation. 5) Tâches
asynchrones pour les gros RFP. 6) Authentification par utilisateur et filtrage par droits.

**20. Comment passer à 100 utilisateurs ?**
API sans état en plusieurs instances derrière un répartiteur, base vectorielle serveur, BM25 dans un
moteur dédié, file de tâches pour la conformité, cache, monitoring des coûts, authentification SSO.

**21. Comment ajouter la mémoire de conversation ?**
Avant la recherche, un LLM condense l'historique et la dernière question en une **question autonome** ; le
reste du pipeline est inchangé. Il faut aussi limiter la taille de l'historique.

**22. Comment ajouter des droits par utilisateur ?**
Stocker `tenant_id` / groupes dans les métadonnées de chaque chunk, les filtrer **à la recherche** (dense
et BM25) avec l'identité du jeton, et tester qu'un utilisateur ne voit jamais les documents d'un autre.

**23. Pourquoi ne pas appeler le LLM quand il n'y a aucun passage ?**
Il inventerait. Un message fixe est plus sûr, moins cher et déterministe.

**24. Qu'as-tu appris de plus important ?**
Qu'un résultat à 100 % sur un petit jeu ne prouve rien, que la qualité se joue dans le parsing, la
recherche et la validation autant que dans le modèle, et qu'il faut mesurer avant de « complexifier ».

**25. Comment as-tu utilisé l'IA pour ce projet ?**
Réponds **honnêtement** : tu as utilisé des assistants (par exemple Claude) pour accélérer l'écriture, les
tests et la documentation. Ce qui compte, c'est que tu **comprends chaque brique** et que tu peux la
modifier et la défendre. Si tu ne peux pas expliquer une ligne, relis-la avant l'entretien.


**26. Combien coûte une requête dans ton projet, et comment le réduirais-tu ?**
Une question : un appel LLM avec environ 3 passages en contexte (plafond 12 000 caractères) et une
réponse courte. Une conformité de RFP : **un appel par exigence**, donc un coût linéaire (80 exigences =
80 appels). Je n'ai **pas** de compteur de tokens, donc pas de chiffre mesuré : ma première amélioration
serait de journaliser `usage` à chaque appel. Ensuite : cache exact, `top_k` validé par l'évaluation,
petit modèle pour l'extraction, regroupement des exigences.

**27. As-tu mis un cache en place ?**
Non, et je le dis. L'interface `LLM` permet d'ajouter un `CachedLLM` sans toucher au reste. Je mesurerais
d'abord la part de questions répétées pour choisir entre un cache exact (sans risque) et un cache
sémantique (plus de hits, risque de faux hit, à éviter pour la conformité).

**28. Où mettrais-tu le cache dans ton pipeline ?**
Deux endroits possibles. **En amont de la recherche** (clé = question normalisée + version de l'index +
modèle + prompt) : économise aussi l'embedding, BM25 et le reranking. **Autour du LLM** (clé = prompt
complet) : l'invalidation est automatique car le contexte change quand les documents changent. Je
choisirais le second pour sa simplicité et sa sûreté.

**29. Que mettrais-tu dans la clé de cache pour ton projet ?**
Question normalisée, nom du modèle (`GROQ_MODEL`), version du prompt, version de la collection
(`company-e5-small-v1`), `top_k`, `candidate_k`, et plus tard le tenant. Si je change l'un d'eux, la
clé change.

**30. Quels appels de ton projet sont déjà évités ou limités ?**
Aucun passage trouvé : pas d'appel (message fixe). `MAX_ATTEMPTS = 3` : tentatives plafonnées. Extraction
de RFP : lots de 10 000 caractères plutôt qu'un appel par phrase. Plafond de contexte à 12 000 / 8 000
caractères. Question limitée à 2 000 caractères.

**31. Quel est le point faible de ton projet côté coût ?**
Trois choses : aucun comptage de tokens, un appel séquentiel par exigence sans regroupement ni cache, et
un seul modèle de raisonnement pour toutes les tâches, avec `max_tokens` à 8 000 jamais réglé par la
mesure.

**32. Quelle tâche passerais-tu sur un modèle moins cher ?**
L'**extraction des exigences** (JSON, température 0, entrée structurée) : tâche d'extraction, validée en
code. La rédaction de la proposition resterait sur le modèle fort. Je le déciderais avec une
évaluation comparant les deux modèles sur mes RFP.

**33. Le prompt caching serait-il utile ici ?**
Peu pour l'instant : le prompt système de la conformité est identique à chaque exigence, mais court
(quelques centaines de tokens), sous la taille minimale de préfixe de plusieurs fournisseurs. Il le
deviendrait avec un long prompt ou des exemples (few-shot) réutilisés. Je vérifierais le seuil et la
grille tarifaire avant de promettre un gain.

**34. Comment accélérer et alléger la matrice de conformité ?**
Paralléliser les exigences avec un pool borné (latence), mettre en cache les exigences déjà évaluées
(clé = texte de l'exigence + version de l'index), regrouper les exigences voisines sur les mêmes
passages, et utiliser un petit modèle quand le retrieval est évident. Attention aux limites de débit.

**35. Comment t'assures-tu qu'une optimisation de coût ne dégrade pas la qualité ?**
Je rejoue `evaluate.py` avant/après, je compare hit@k, MRR et un échantillon de réponses relues. Honnêtement,
mon jeu actuel (10 questions faciles) ne détecterait pas une dégradation fine : l'étoffer est un
préalable à toute optimisation sérieuse.

---

## 3. Conception de systèmes

Méthode pour chaque scénario : **clarifier** (volumes, langues, latence, confidentialité, qui utilise) →
**schéma** → **choix justifiés** → **risques** → **comment mesurer**. Pose des questions avant de dessiner.

### Scénario 1 : RAG sur 10 millions de documents d'entreprise
- **Clarifier** : formats, mises à jour par jour, droits d'accès, langues, latence cible.
- **Ingestion** : pipeline asynchrone (files), parsing par type, OCR si besoin, chunking structurel,
  embeddings par lots (GPU), IDs déterministes, métadonnées de droits.
- **Stockage** : base vectorielle serveur avec HNSW (+ quantization), moteur lexical dédié
  (OpenSearch/Elasticsearch), partitionnement.
- **Requête** : hybride + RRF, reranking de 50 candidats, filtrage par droits **dans** la recherche.
- **Risques** : coût des embeddings, fraîcheur, doublons, droits. **Mesure** : rappel@k sur un jeu
  annoté, latence p95, coût par requête.

### Scénario 2 : chatbot de support client, multilingue, avec historique
- Condensation de l'historique en question autonome, embeddings multilingues, détection de langue,
  réponse dans la langue de l'utilisateur, **escalade vers un humain** quand la confiance est basse.
- Streaming pour la latence perçue, cache des questions fréquentes, retour pouce haut/bas.
- **Mesure** : taux de résolution, taux d'escalade, fidélité sur un échantillon relu.

### Scénario 3 : RAG juridique / conformité (zéro invention tolérée)
- Citations **obligatoires et vérifiées** (passage exact), refus par défaut, seuil de confiance,
  relecture humaine systématique, journalisation complète (audit).
- Chunking qui respecte articles et clauses, métadonnées de version et de date (la loi change).
- Évaluation de fidélité stricte, jeu de cas piégeurs, revue par un juriste.

### Scénario 4 : plateforme SaaS multi-clients
- Isolation : collection ou `tenant_id` filtré systématiquement, tests d'isolation automatisés.
- Clés de chiffrement par client si exigé, quotas et limitation de débit, facturation au token.
- Suppression complète d'un client (droit à l'effacement) : documents, vecteurs, index lexical, logs.

### Scénario 5 : questions mixtes (texte + chiffres dans des tables SQL)
- Routeur : question documentaire → RAG ; question chiffrée → génération de SQL (lecture seule,
  schéma fourni, validation) ; question mixte → les deux puis synthèse.
- Garde-fous : requêtes en lecture seule, limites de lignes, validation du SQL avant exécution.

### Scénario 6 : réponse en moins de deux secondes
- Budget : le LLM domine. Leviers : modèle plus petit/rapide, moins de contexte, **streaming**,
  cache exact et sémantique, embedding de la question en ligne seulement (documents pré-vectorisés),
  reranking limité ou désactivé si la mesure montre qu'il apporte peu.
- Mesurer la latence **par étape** avant d'optimiser.


### Scénario 7 : réduire de 60 % la facture LLM d'un assistant très utilisé
- **Clarifier** : quel poste coûte le plus (chat, génération longue, traitement par lots) ? Combien de
  requêtes sont répétées ? Quelle qualité minimale est exigée ? Données sensibles (limite le cache
  partagé) ?
- **Mesurer d'abord** : tokens entrée/sortie par fonctionnalité, par modèle, par client ; part des
  requêtes identiques ou quasi identiques ; taux de réponses vides ou en échec (tentatives gaspillées).
- **Gains sûrs** : cache exact (clé versionnée), `top_k` et plafond de contexte réglés par la mesure,
  tentatives bornées, sortie limitée, traitement hors ligne passé en API batch.
- **Gains à valider** : routage vers un petit modèle pour les tâches simples (cascade avec validation),
  prompt caching sur les gros préfixes stables, compression du contexte.
- **Gains risqués** : cache sémantique (seuil prudent, jamais sur les domaines à enjeu), distillation
  (seulement à fort volume stable).
- **Garde-fous** : budget par client, alertes, limitation de débit, tableau de bord du coût.
- **Risques** : péremption du cache, fuite entre clients, qualité dégradée sans qu'on le voie.
  **Mesure** : coût par requête, taux de hit, qualité sur le jeu d'évaluation avant/après.

---

## 4. Questions pièges

**Ton RAG est-il fiable à 100 % ?**
Non, et je me méfie de qui l'affirme. Je réduis le risque (recherche, prompt, vérifications en code) et
je le détecte (citations vérifiées, relecture humaine), sans le supprimer.

**Pourquoi ne pas simplement utiliser ChatGPT avec les documents ?**
Données privées envoyées à un service généraliste, pas de contrôle d'accès ni d'audit, citations non
vérifiables, pas de pipeline reproductible ni d'évaluation. Un RAG maîtrisé apporte contrôle et
traçabilité.

**Plus de contexte, est-ce toujours mieux ?**
Non : bruit, coût, latence, et information « perdue au milieu ». Mieux vaut moins de passages mieux
choisis.

**Un plus gros modèle d'embedding est-il toujours meilleur ?**
Pas forcément : dépend de la langue, du domaine et des données. On le prouve avec une évaluation, en
tenant compte du coût et de la latence.

**Le reranker est-il inutile avec de bons embeddings ?**
Il apporte surtout de la précision sur le **haut** du classement. À prouver par ablation : si le gain est
nul sur ton jeu, retire-le.

**Quelle différence entre RAG et recherche classique ?**
La recherche classique renvoie des documents ; le RAG ajoute une **génération** qui synthétise une
réponse à partir des passages, avec les risques propres à la génération.

**Comment prouves-tu que l'hybride est meilleur que le dense seul ?**
Par une ablation sur un jeu varié (hit@k, MRR) : dense, BM25, hybride, hybride + rerank. Je ne prétends
pas l'avoir chiffré tant que le jeu est trop facile.

**Un utilisateur dit que la réponse est fausse : que fais-tu ?**
Je remonte la chaîne : le passage existe-t-il ? est-il récupéré ? est-il dans le contexte envoyé ? le
modèle l'a-t-il utilisé ? Puis j'ajoute le cas au jeu d'évaluation.

**Combien coûte une requête ?**
Tokens d'entrée (contexte + question) plus tokens de sortie, au tarif du modèle. Avec 3 courts passages,
c'est modeste ; le contexte maximal autorisé est de 12 000 caractères. Je mesure plutôt que d'estimer.

**Peux-tu garantir qu'aucune donnée ne fuit ?**
Non. Je peux limiter : filtrage par droits, minimisation, masquage, fournisseur contractuel ou
auto-hébergé, journalisation sans contenu sensible.

**Pourquoi une température de 0,1 et non 0 ?**
Peu d'écart en pratique : 0 pour l'extraction structurée où la reproductibilité compte, 0,1 pour garder
une rédaction naturelle tout en restant stable. À régler par la mesure.

**Qu'est-ce qui te rend le moins sûr dans ton projet ?**
L'évaluation. Dire cela spontanément est un signe de maturité, pas de faiblesse.


**Le cache ne risque-t-il pas de servir une vieille réponse ?**
Si, c'est son principal danger. Parades : version de l'index et du prompt dans la clé, TTL, purge à
l'ingestion et à la suppression, et ne cacher que des réponses validées.

**Un cache sémantique à 0,85 de seuil, c'est bien ?**
Probablement trop bas : il servira des réponses à des questions différentes (« résilier » contre
« renouveler »). Le seuil se règle sur des paires réelles étiquetées, avec une marge de prudence, et on
le désactive pour les domaines à enjeu.

**Le prompt caching est-il un cache de réponses ?**
Non. Le modèle génère une nouvelle réponse à chaque fois ; seul le traitement du préfixe identique est
facturé moins cher. Il ne réduit donc pas le nombre d'appels, seulement le coût des tokens d'entrée
répétés.

**Pourquoi ne pas simplement prendre le modèle le moins cher ?**
Parce que le coût total inclut les erreurs : réponses fausses, reformulations, relecture humaine,
perte de confiance. On prend le moins cher **qui atteint la qualité requise**, prouvée par l'évaluation.

**`max_tokens` à 100 pour économiser, bonne idée ?**
Pas sans réfléchir : une réponse coupée est inutile, et avec un modèle de raisonnement la réflexion peut
consommer tout le budget et donner une réponse **vide** (c'est arrivé dans mon projet). On règle par la
mesure.

---

## 5. Exercices de code

Tous ces extraits ont été exécutés et vérifiés. Apprends à les écrire **sans** les relire.

### Similarité cosinus
```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
```

### Reciprocal Rank Fusion
```python
def rrf(rankings, k=60):
    """rankings : liste de classements (listes d'ids du meilleur au moins bon)."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

# rrf([["A", "B", "C"], ["X", "Y", "A"]])[0] == "A"
# A : 1/61 + 1/63 = 0.03227  ;  B : 1/62 = 0.01613
```

### BM25 minimal
```python
import math
from collections import Counter

def bm25_scores(query_tokens, docs_tokens, k1=1.5, b=0.75):
    n = len(docs_tokens)
    avgdl = sum(map(len, docs_tokens)) / n
    df = Counter(t for d in docs_tokens for t in set(d))
    scores = []
    for d in docs_tokens:
        tf = Counter(d)
        s = 0.0
        for q in query_tokens:
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores
```
Remarque : ce `idf` (variante « 1 + ») est toujours positif. `rank_bm25` utilise
`log((N − n + 0.5)/(n + 0.5))`, qui tombe à zéro pour un mot présent dans la moitié des documents.

### Métriques de recherche
```python
import math

def mrr(results, expected):
    """results : listes ordonnées d'ids ; expected : l'id attendu pour chaque question."""
    total = 0.0
    for ranked, exp in zip(results, expected):
        total += next((1 / i for i, d in enumerate(ranked, 1) if d == exp), 0.0)
    return total / len(expected)

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(rels, k):
    """rels : pertinence de chaque résultat, dans l'ordre du classement."""
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1))
    ideal = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal[:k], start=1))
    return dcg / idcg if idcg else 0.0
```
Précision : ici le classement idéal est calculé sur les résultats renvoyés ; la définition complète le
calcule sur **tous** les documents pertinents connus.

### Chunking avec chevauchement
```python
def chunk_with_overlap(words, size, overlap):
    if overlap >= size:
        raise ValueError("overlap doit être inférieur à size")
    step = size - overlap
    return [words[i:i + size] for i in range(0, max(len(words) - overlap, 1), step)]

# chunk_with_overlap(list(range(10)), 4, 1) -> [[0,1,2,3], [3,4,5,6], [6,7,8,9]]
```

### MMR (diversité)
```python
def mmr(query_vec, doc_vecs, k, lambda_=0.7):
    selected, candidates = [], list(range(len(doc_vecs)))
    while candidates and len(selected) < k:
        def score(i):
            rel = cosine(query_vec, doc_vecs[i])
            red = max((cosine(doc_vecs[i], doc_vecs[j]) for j in selected), default=0.0)
            return lambda_ * rel - (1 - lambda_) * red
        best = max(candidates, key=score)
        selected.append(best)
        candidates.remove(best)
    return selected
```
`lambda_` proche de 1 : pertinence prioritaire (un quasi-doublon peut passer). Proche de 0 : diversité
prioritaire. Avec `lambda_=0.3`, le document différent est préféré au quasi-doublon.

### Cache exact avec TTL et LRU
```python
import time
from collections import OrderedDict

class TTLCache:
    def __init__(self, max_size=1000, ttl=3600, clock=time.monotonic):
        self.max_size, self.ttl, self.clock = max_size, ttl, clock
        self._data = OrderedDict()
        self.hits = self.misses = 0

    def get(self, key):
        item = self._data.get(key)
        if item is None or item[0] < self.clock():
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item[1]

    def set(self, key, value):
        self._data[key] = (self.clock() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)
```
Test attendu avec une horloge injectée : `max_size=2`, on ajoute `a`, `b`, on lit `a`, on ajoute `c` :
`b` est évincé ; après avoir avancé l'horloge au-delà du TTL, tout a expiré.

### Clé de cache complète
```python
import hashlib, json, re

def cache_key(question, *, model, prompt_version, index_version, tenant, top_k):
    norm = re.sub(r"\s+", " ", question.strip().lower())
    payload = json.dumps([norm, model, prompt_version, index_version, tenant, top_k],
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```
Vérifié : changer le tenant, la version du prompt ou le modèle change la clé ; espaces et casse non.

### Décorateur de LLM avec cache
```python
class CachedLLM:
    def __init__(self, llm, cache, version="v1"):
        self.llm, self.cache, self.version = llm, cache, version

    def generate(self, system_prompt, user_prompt):
        key = hashlib.sha256(
            json.dumps([self.version, system_prompt, user_prompt]).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        answer = self.llm.generate(system_prompt, user_prompt)
        self.cache.set(key, answer)
        return answer
```
Test avec un faux LLM qui compte ses appels : deux appels identiques → 1 appel réel ; un prompt
différent → 2.

### Estimation de coût
```python
def cost_usd(in_tok, out_tok, price_in, price_out):
    """Prix en $ par million de tokens."""
    return (in_tok * price_in + out_tok * price_out) / 1_000_000

# cost_usd(2000, 300, 1.0, 5.0) == 0.0035
```

### Cascade petit modèle → gros modèle
```python
def cascade(question, small, large, confident):
    answer = small(question)
    return answer if confident(answer) else large(question)
```

### Backoff exponentiel avec plafond
```python
import time

def retry(fn, attempts=3, base=0.5, sleep=time.sleep):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            sleep(base * 2 ** i)       # attentes : 0,5 s puis 1 s
```

Le *single-flight* et le cache sémantique sont dans le module 16 de la formation. Tous ces extraits ont
été exécutés avec des tests (clé, LRU, TTL, expiration, cache sémantique avec isolation par tenant,
single-flight sur 5 threads, cascade, backoff).

### Questions de code fréquentes
- Écrire un `ContextBuilder` qui respecte une limite de caractères et renvoie les sources utilisées
  (c'est `generation/context.py`).
- Valider la sortie JSON d'un LLM (statut autorisé, types, sources existantes) : voir `rfp/compliance.py`.
- Rendre l'indexation **idempotente** (IDs déterministes par hash).
- Écrire un **cache** LRU avec TTL, puis un `CachedLLM` testé avec un faux LLM (voir plus haut).
- Écrire un test avec un **faux** LLM : voir `tests/conftest.py`.

---

## 6. Raconter une histoire (méthode STAR)

Structure : **S**ituation, **T**âche, **A**ction, **R**ésultat. Court, concret, honnête.

### Exemple : le préfixe E5
- **Situation** : mon RAG affichait 100 % de réussite sur mon jeu d'évaluation.
- **Tâche** : préparer le projet pour une revue approfondie.
- **Action** : j'ai relu chaque module en le comparant à la documentation du modèle. J'ai vu que
  `embed_query` utilisait le préfixe `passage:` alors qu'E5 demande `query:` pour les questions. J'ai
  corrigé, ajouté trois tests qui vérifient le préfixe utilisé, et vérifié que l'index existant restait
  valable.
- **Résultat** : défaut corrigé sans ré-indexation. Le point important : mon évaluation ne l'avait pas
  détecté, car le jeu était trop facile ; j'ai donc fait de son extension ma priorité.

### Exemple : BM25 pas à jour
- **Situation** : après un upload, les nouvelles données n'apparaissaient pas par mots-clés ; l'interface
  disait même de redémarrer l'API.
- **Action** : j'ai compris que BM25 est construit en mémoire au démarrage, ajouté `refresh()` appelé
  après chaque upload et suppression, avec des tests, puis vérifié sur la stack réelle (document trouvé
  puis plus trouvé après suppression).
- **Résultat** : plus de redémarrage nécessaire.

---

## 7. Les questions à poser au recruteur

- Quel est le volume de documents et la nature des données (confidentialité, langues) ?
- Comment évaluez-vous aujourd'hui la qualité de vos réponses ? Avez-vous un jeu de référence ?
- Quels fournisseurs de modèles utilisez-vous, et quelles contraintes de localisation des données ?
- Comment sont déployés et surveillés les systèmes en production ?
- Quelle part du travail est de la recherche et de l'évaluation contre de l'ingénierie ?
- Comment l'équipe gère-t-elle les retours des utilisateurs pour améliorer le système ?

---

## 8. Les erreurs à éviter

- **Promettre 100 % de fiabilité.** Parle de réduction et de détection du risque.
- **Citer une technique sans la justifier.** Mieux : « je n'ai pas fait X faute de mesure qui en montre le besoin ».
- **Prétendre avoir mis en production** ce qui a seulement été validé en local.
- **Réciter des chiffres que tu n'as pas mesurés.** Distingue « mesuré » de « repère de la littérature ».
- **Ne pas connaître tes limites.** Les citer toi-même montre de la maturité.
- **Ne pas savoir expliquer une ligne de ton code.** Relis chaque module avant l'entretien.
- **Parler uniquement du modèle.** Un bon RAG se joue dans le parsing, la recherche, la validation et
  l'évaluation.
