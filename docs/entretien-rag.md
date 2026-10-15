# Entretien RAG : questions techniques et réponses

Plus de 200 questions classées par thème. Chaque réponse est volontairement **courte**, comme on la
donnerait à l'oral. Le cours détaillé est dans [`formation-rag.md`](formation-rag.md) ; les questions
sur ton projet, la conception de systèmes et les exercices de code sont dans
[`entretien-projet-rag.md`](entretien-projet-rag.md).

**Comment t'entraîner** : cache la réponse, réponds à voix haute en 30 à 60 secondes, puis compare.
Si tu bloques sur une question, retourne au module correspondant du cours.

## Sommaire
A. Fondamentaux · B. Ingestion et chunking · C. Embeddings et bases vectorielles ·
D. Recherche hybride et reranking · E. Génération et prompts · F. Évaluation ·
G. Sécurité et conformité · H. Production et architecture ·
I. Coût des appels LLM et cache

---

## A. Fondamentaux

**1. Qu'est-ce que le RAG ?**
Une architecture où, avant de générer, on **récupère** dans une base de documents les passages
pertinents et on les donne au LLM comme contexte. Le modèle répond à partir de ces passages, ce qui
apporte des connaissances privées ou récentes et permet de **citer des sources**.

**2. RAG ou fine-tuning ?**
Le RAG apporte des **connaissances** (changeantes, privées, citables, avec droits d'accès). Le
fine-tuning change un **comportement** (style, format, vocabulaire). Pour des faits qui évoluent, le
fine-tuning est mauvais : il faudrait ré-entraîner, et il ne cite rien. Les deux se combinent.

**3. RAG ou long contexte (tout mettre dans le prompt) ?**
Le long contexte convient aux petits corpus et analyses ponctuelles. Le RAG reste meilleur quand le
corpus est grand, pour le **coût** (on n'envoie que le pertinent), la **latence**, les **citations**
précises et le **contrôle d'accès**. De plus, la qualité d'utilisation de l'information peut baisser
sur de très longs contextes.

**4. Pourquoi un LLM seul ne suffit-il pas pour des documents internes ?**
Il ne les connaît pas, il peut inventer, il ne cite pas, et on ne peut pas lui imposer de droits
d'accès. Les données internes ne doivent pas non plus servir à l'entraîner sans réflexion.

**5. Quels sont les composants d'un pipeline RAG ?**
Ingestion (parsing, chunking, embeddings, stockage), recherche (dense, lexicale, fusion, reranking),
construction du contexte, génération (prompt, LLM), et autour : évaluation, sécurité, observabilité.

**6. Quand le RAG est-il un mauvais choix ?**
Corpus minuscule, questions qui demandent une synthèse globale de tout le corpus, données
tabulaires (préférer SQL ou des outils), ou exigence de précision absolue sans relecture humaine.

**7. Qu'est-ce qu'un embedding ?**
Un vecteur de nombres qui représente le **sens** d'un texte : des textes proches en sens ont des
vecteurs proches. La recherche devient un calcul de distance.

**8. Cosinus, produit scalaire, distance euclidienne : quelle différence ?**
Le cosinus mesure l'**angle** (indépendant de la longueur du vecteur). Le produit scalaire dépend
aussi de la norme. La distance euclidienne mesure l'écart de position. Pour des vecteurs
**normalisés**, cosinus et produit scalaire donnent le même classement.

**9. Bi-encoder ou cross-encoder ?**
Le bi-encoder encode question et document **séparément** : les documents sont vectorisés à l'avance,
donc très rapide. Le cross-encoder lit la **paire** ensemble : plus précis mais un calcul par paire à
chaque requête. D'où l'entonnoir : bi-encoder pour récupérer, cross-encoder pour reclasser.

**10. Pourquoi normaliser les embeddings ?**
Pour que le cosinus se réduise à un produit scalaire (plus rapide) et que la longueur du vecteur
n'influence pas la comparaison.

**11. Qu'est-ce qu'un token et pourquoi est-ce important en RAG ?**
Un morceau de mot que le modèle manipule. Les **limites** (entrée de l'embedding, contexte du LLM,
sortie) et les **coûts** se comptent en tokens, pas en mots.

**12. Qu'est-ce que la fenêtre de contexte ?**
Le nombre maximal de tokens que le modèle peut lire (entrée + sortie). Elle borne la quantité de
passages qu'on peut fournir.

**13. Que fait la température ?**
Elle règle le caractère aléatoire de la génération. Basse (0 à 0,2) pour des réponses factuelles et
reproductibles, plus haute pour la créativité.

**14. Recherche sémantique et lexicale : la différence ?**
La sémantique (embeddings) trouve le **sens**, y compris avec des mots différents. La lexicale (BM25)
trouve les **mots exacts**. Chacune rate ce que l'autre trouve.

**15. Qu'est-ce qu'une hallucination ?**
Une réponse fluide mais **non appuyée** par les sources (invention, mauvaise attribution,
extrapolation).

**16. Le RAG supprime-t-il les hallucinations ?**
Non, il les **réduit**. Si la recherche rate le bon passage, ou si le modèle extrapole, l'erreur
reste possible. On ajoute citations, vérifications en code et relecture humaine.

**17. Qu'est-ce qu'une réponse « fondée » (grounded) ?**
Chaque affirmation est soutenue par un passage fourni. C'est ce que mesure la métrique de
**fidélité** (faithfulness).

**18. Erreur de recherche ou erreur de génération : comment distinguer ?**
Regarde les passages récupérés. Si le bon passage n'y est pas, c'est la **recherche**. S'il y est
mais que la réponse est fausse, c'est la **génération** (prompt, modèle, trop de bruit).

**19. Que signifient top_k et candidate_k ?**
`candidate_k` : nombre de candidats récupérés par la recherche rapide. `top_k` : nombre de passages
gardés (après reranking) et envoyés au LLM. Dans mon projet : 10 candidats, 3 passages par défaut.

**20. Pourquoi citer les sources ?**
Vérifiabilité, confiance, audit, et possibilité pour l'utilisateur de contrôler. Dans un contexte
d'appel d'offres, c'est indispensable.

---

## B. Ingestion et chunking

**21. Comment choisir la taille des chunks ?**
Il n'y a pas de valeur universelle : compromis précision (petits) contre contexte (grands). On part
d'un repère (200-500 tokens) puis on **mesure** hit@k et MRR pour plusieurs tailles sur ses
propres questions.

**22. Pourquoi un chevauchement entre chunks ?**
Pour ne pas couper une idée à la frontière de deux chunks. Typiquement 10 à 20 %. Coût : plus de
chunks, quelques doublons.

**23. Quelles stratégies de chunking connais-tu ?**
Taille fixe, avec chevauchement, récursive (paragraphe puis phrase puis mot), structurelle (suit les
titres), sémantique (coupe quand le sens change), parent-enfant.

**24. Chunking sémantique : avantages et limites ?**
Il respecte les changements de sujet, mais coûte des calculs d'embeddings à l'ingestion et le gain
est variable. À justifier par une mesure.

**25. Qu'est-ce que le parent-document retrieval ?**
On cherche sur de **petits** chunks (précis) mais on renvoie au LLM le **passage plus large** qui
les contient (contexte). On gagne les deux.

**26. Comment traiter des PDF scannés ?**
Il faut de l'**OCR** (Tesseract, services cloud). Sans cela le parser ne trouve pas de texte. Mon
parser le détecte et renvoie une erreur explicite.

**27. Comment traiter les tableaux ?**
Les extraire en texte structuré (lignes `a | b | c` ou Markdown), parfois avec un résumé, plutôt que
de laisser le parser les mélanger. Les tableaux complexes de PDF sont un point faible courant.

**28. Quelles métadonnées stocker ?**
Document source, section (chemin de titres), page, date, auteur, version, client/droits. Elles
servent à citer, filtrer, supprimer et déboguer.

**29. Comment gérer la mise à jour d'un document ?**
Soit supprimer ses anciens chunks puis ré-indexer, soit **upsert** avec des identifiants
déterministes (même contenu, même ID, donc pas de doublon). Penser aussi à l'index lexical.

**30. Comment gérer une suppression ?**
Supprimer les chunks de la base vectorielle **et** rafraîchir l'index lexical, sinon le document
supprimé continue d'être trouvé. J'ai eu exactement ce bug avant de le corriger.

**31. Pourquoi des identifiants de chunk déterministes ?**
Pour l'**idempotence** : ré-indexer deux fois donne le même résultat. Ici, un SHA-256 du document, de
la position, du chemin de titres et du texte.

**32. Comment dédoublonner ?**
Hash du contenu normalisé à l'ingestion, ou au moment de la recherche avec MMR / suppression des
passages quasi identiques dans le top-k.

**33. Que se passe-t-il si un passage dépasse la limite du modèle d'embedding ?**
Il est tronqué (perte silencieuse) ou refusé. Mon embedder le **refuse** avec une erreur claire
plutôt que de tronquer en silence ; la correction propre est de découper les longs passages avant.

**34. Comment traiter des documents multilingues ?**
Modèle d'embedding **multilingue** pour la recherche croisée (question en anglais, document en
français). Attention : BM25 est monolingue, un reranker anglais est moins fiable sur du français.

**35. Pourquoi préfixer chaque chunk par son chemin de titres ?**
Un paragraphe isolé perd son sujet. Le titre (« Sécurité > Protection des données ») l'ancre et
améliore les embeddings comme BM25.

---

## C. Embeddings et bases vectorielles

**36. Comment choisir un modèle d'embedding ?**
Langues, longueur d'entrée maximale, dimension, vitesse/coût, licence, scores de benchmark (MTEB) et
surtout performance sur **mes** données avec mon jeu d'évaluation.

**37. À quoi servent les préfixes `query:` et `passage:` de E5 ?**
Le modèle est entraîné avec : `query: ` devant les questions, `passage: ` devant les documents. Les
omettre ou les inverser dégrade la qualité. Mon projet encodait les questions avec `passage:` ; je
l'ai corrigé.

**38. Que se passe-t-il si on change de modèle d'embedding ?**
Les espaces vectoriels sont incompatibles : il faut **tout ré-indexer**. On versionne la collection
(`...-v1`, `...-v2`) et on bascule quand la nouvelle est prête.

**39. Dimension 384 ou 1536 ?**
Plus de dimensions = plus de finesse mais plus de stockage, de mémoire et de temps de recherche. 384
suffit souvent pour un prototype ou du CPU ; à valider par la mesure.

**40. Explique HNSW.**
Un graphe à plusieurs couches. On entre par la couche supérieure (peu de nœuds, grands sauts) puis on
descend en affinant vers les voisins les plus proches. Recherche approximative très rapide.

**41. Recherche exacte ou approximative (ANN) ?**
L'exacte compare tout (correcte mais lente à grande échelle). L'ANN sacrifie un peu de rappel pour
gagner énormément en vitesse. Jusqu'à quelques dizaines de milliers de vecteurs, l'exacte suffit.

**42. Quels paramètres règlent HNSW ?**
`M` (liens par nœud), `efConstruction` (qualité de construction), `efSearch` (qualité de recherche).
Plus ils sont élevés, meilleur est le rappel, plus c'est lent et gourmand en mémoire.

**43. Chroma, pgvector, Qdrant, Pinecone, FAISS ?**
Chroma : simple, local, prototype. pgvector : vecteurs dans PostgreSQL, SQL et filtres. Qdrant/Weaviate/
Milvus : bases spécialisées, filtrage riche, distribuées. Pinecone : managé. FAISS : bibliothèque
rapide, pas une base.

**44. Pré-filtrage ou post-filtrage par métadonnées ?**
Le **pré**-filtrage restreint l'espace avant la recherche (correct, nécessite le support de la base).
Le **post**-filtrage filtre après : risque de se retrouver avec trop peu de résultats.

**45. Que renvoie Chroma avec l'espace cosinus ?**
Une **distance** = 1 − cosinus : plus c'est petit, plus c'est proche.

**46. Comment stocker 10 millions de chunks ?**
Base vectorielle serveur, index HNSW ou IVF, quantization (PQ) pour la mémoire, partitionnement
(sharding), filtrage par métadonnées, et surveiller le rappel.

**47. Qu'est-ce que la quantization ?**
Réduire la précision des vecteurs (par exemple en entiers) pour gagner de la mémoire et de la vitesse,
au prix d'une légère perte de précision.

**48. Faut-il fine-tuner les embeddings ?**
Seulement si le vocabulaire est très spécialisé, qu'on a assez de paires (question, passage) et que la
mesure montre un gain. Sinon, un meilleur modèle ou un reranker rapporte plus pour moins d'effort.

**49. Qu'est-ce que la dérive des embeddings (drift) ?**
Quand le corpus évolue (nouveaux sujets, vocabulaire) et que le modèle ou les chunks ne le couvrent
plus bien, ou qu'on change de version de modèle. Se détecte par l'évaluation continue.

**50. Quelles sont les faiblesses des embeddings ?**
Identifiants exacts, nombres, négations (« n'est pas chiffré » proche de « est chiffré »), noms
propres rares, sigles. D'où le lexical et le reranking.

---

## D. Recherche hybride et reranking

**51. Pourquoi une recherche hybride ?**
Le dense comprend le sens mais rate les mots exacts (codes, sigles) ; BM25 trouve les mots exacts mais
pas les synonymes. Ensemble, ils se complètent.

**52. Explique BM25.**
Score = somme, sur les mots de la question, de `IDF × fréquence saturée normalisée par la longueur`.
`IDF` : rareté du mot. `k1` : saturation (répéter ne rapporte pas linéairement). `b` : normalisation
de la longueur du document.

**53. Qu'est-ce que l'IDF ?**
L'inverse de la fréquence documentaire : un mot présent dans presque tous les documents compte peu, un
mot rare compte beaucoup.

**54. Limites de BM25 ?**
Pas de synonymes, pas de langues croisées, sensible à la tokenisation (pas de racinisation sans
réglage), dégénéré sur de très petits corpus (j'ai vu des scores nuls avec 2 documents).

**55. Explique RRF avec un exemple.**
`score(d) = Σ 1/(k + rang)`. Un document rang 1 en dense et rang 3 en BM25 : 1/61 + 1/63 ≈ 0,0323. Un
document rang 1 seulement en dense : 1/61 ≈ 0,0164. Le premier gagne : être trouvé par les deux
méthodes est récompensé.

**56. Pourquoi k = 60 ?**
Valeur classique de la littérature : elle atténue l'écart entre rangs consécutifs et empêche un seul
rang 1 de dominer. Réglable, mais 60 marche bien en pratique.

**57. RRF ou somme pondérée de scores ?**
RRF n'a besoin d'aucune normalisation (les échelles des scores diffèrent) et est robuste. La somme
pondérée est réglable mais fragile.

**58. Rôle du reranker ?**
Reclasser les candidats avec un cross-encoder qui lit chaque paire (question, passage) : plus précis
que la similarité vectorielle, utilisé sur peu de candidats.

**59. Coût d'un cross-encoder ?**
Un calcul par paire à chaque requête : le coût croît avec `candidate_k`. D'où un nombre limité de
candidats (10 à 50) et un petit modèle.

**60. Comment choisir candidate_k ?**
Assez grand pour que le bon passage soit dans les candidats (rappel), assez petit pour garder le
reranking rapide. On trace le rappel en fonction de `candidate_k` sur son jeu d'évaluation.

**61. Qu'est-ce que ColBERT ?**
Un modèle à **interaction tardive** : un vecteur par token, scores calculés entre tokens de la
question et du document. Plus précis qu'un seul vecteur, moins coûteux qu'un cross-encoder complet.

**62. Un LLM peut-il servir de reranker ?**
Oui : on lui demande de noter ou ordonner les passages. Souvent précis mais plus lent et plus cher.

**63. Qu'est-ce que MMR ?**
*Maximal Marginal Relevance* : sélectionne des passages à la fois **pertinents** et **différents** les
uns des autres, pour éviter les doublons dans le contexte.

**64. Quand réécrire la requête ?**
Quand les questions sont vagues, courtes, ou dépendent de la conversation. Un LLM la transforme en
requête de recherche autonome.

**65. Qu'est-ce que HyDE ?**
On demande au LLM d'écrire une **réponse hypothétique**, et on cherche avec son embedding plutôt
qu'avec celui de la question. Utile quand question et document n'ont pas la même forme.

**66. Qu'est-ce que le multi-requêtes ?**
Générer plusieurs reformulations, chercher avec chacune, fusionner (RRF). Améliore le rappel au prix
de la latence.

**67. Comment traiter une question en plusieurs parties ?**
La **décomposer** en sous-questions, chercher pour chacune, puis synthétiser.

**68. Comment gérer une conversation (« et pour les sauvegardes ? ») ?**
Condenser l'historique et la dernière question en **une question autonome** avant la recherche.

**69. Plusieurs sources ou plusieurs index : comment router ?**
Un classifieur ou un LLM choisit la source selon la question, ou on cherche partout et on fusionne.
Les approches agentiques laissent le modèle décider.

**70. Le rappel est faible : par où commencer ?**
1) Regarder les échecs un par un. 2) Vérifier le parsing et le chunking. 3) Augmenter `candidate_k`.
4) Essayer hybride / autre modèle d'embedding. 5) Réécriture de requête. À chaque étape, **mesurer**.

---

## E. Génération et prompts

**71. Que contient un bon prompt RAG ?**
Rôle, règle « réponds uniquement avec le contexte », comportement de repli (« dis-le si l'information
manque »), format de citation, séparation nette contexte/question.

**72. Comment obtenir des citations fiables ?**
Numéroter les sources dans le contexte, demander de citer ces identifiants, puis **vérifier en code**
que les citations existent. Dans la conformité, les sources inventées sont supprimées.

**73. Qu'est-ce que « lost in the middle » ?**
Les modèles exploitent moins bien l'information placée au milieu d'un long contexte. Donc peu de
passages bien choisis, et les meilleurs aux extrémités.

**74. Combien de passages envoyer ?**
Assez pour couvrir la réponse, assez peu pour limiter bruit, coût et dilution. Souvent 3 à 8 ; à régler
avec l'évaluation.

**75. Comment gérer « je ne sais pas » ?**
Dans le prompt (repli explicite), et **en code** : si aucun passage n'est trouvé, ne pas appeler le LLM
et renvoyer un message fixe. Option : seuil de score sur le reranker.

**76. Pourquoi une température basse ?**
Pour des réponses factuelles et reproductibles. Pour l'extraction JSON, j'utilise 0.

**77. Les sorties JSON d'un LLM sont-elles fiables ?**
Non, pas à 100 %. Mode JSON du fournisseur, **validation stricte en code** (types, valeurs autorisées),
nouvelle tentative ou découpage du lot en cas d'échec.

**78. Que se passe-t-il avec un modèle de raisonnement et une limite de tokens basse ?**
Il peut consommer tout son budget à raisonner et renvoyer une réponse **vide**. J'ai fixé une limite
explicite, ajouté des tentatives et une erreur 502 propre.

**79. À quoi sert le streaming ?**
Afficher la réponse au fur et à mesure : la latence **perçue** baisse même si le temps total est
identique.

**80. Comment vérifier qu'une citation est correcte ?**
Technique : l'identifiant existe dans le contexte. Sémantique : un second contrôle (LLM-juge ou
vérification de fidélité) que la phrase est vraiment soutenue par ce passage.

**81. Que faire si les sources se contredisent ?**
Le prompt demande de signaler la contradiction et de citer les deux ; idéalement, des métadonnées de
date/version permettent de privilégier la plus récente.

**82. Comment répondre à une question qui demande plusieurs documents ?**
S'assurer que la recherche renvoie des passages de plusieurs documents (MMR, décomposition), et que le
contexte reste assez large pour les contenir.

**83. Comment réduire le contexte sans perdre l'information ?**
Meilleur reranking (moins de passages), compression extractive (garder les phrases utiles), résumé
des passages, suppression des doublons.

**84. Comment choisir le LLM ?**
Qualité sur ma tâche (mesurée), coût par token, latence, confidentialité, contexte maximum, support des
sorties structurées. Une interface abstraite permet de changer sans refonte.

**85. Function calling ou RAG ?**
Le RAG récupère du **texte** non structuré. Le function calling interroge des **systèmes** (base SQL,
API). Souvent combinés.

**86. Question en français, documents en anglais ?**
Embeddings multilingues pour la recherche, prompt qui demande de répondre dans la langue de la question
en citant le document d'origine.

**87. Comment contrôler le style de la réponse ?**
Instructions de format dans le prompt, exemples (few-shot), sections imposées (ma proposition suit 8
sections fixes).

**88. Que faire si un passage contient des instructions malveillantes ?**
Traiter le contexte comme des **données**, le dire dans le prompt, limiter les actions possibles,
valider les sorties en code, relecture humaine (module sécurité).

**89. Comment tester un changement de prompt ?**
Rejouer le jeu d'évaluation avant/après, comparer fidélité et pertinence, relire un échantillon. Jamais
« ça me semble mieux ».

**90. Faut-il versionner les prompts ?**
Oui, comme du code (git), avec le modèle et la température utilisés, pour pouvoir reproduire un
résultat et revenir en arrière.

---

## F. Évaluation

**91. Comment évalues-tu ton RAG ?**
Deux niveaux séparés : la **recherche** (hit@k, MRR sur des questions avec le bon document attendu) et
la **génération** (fidélité, pertinence). Un script rejoué à chaque modification.

**92. hit@k et recall@k : la différence ?**
hit@k : au moins un bon résultat dans les k premiers (oui/non). recall@k : la **part** des résultats
pertinents retrouvés dans les k premiers.

**93. Explique le MRR.**
Moyenne de `1 / rang` du premier bon résultat. Rang 1 → 1, rang 2 → 0,5, rang 4 → 0,25. Il récompense
de mettre le bon résultat tôt.

**94. Explique nDCG.**
`DCG = Σ pertinence_i / log2(i+1)` : les résultats pertinents bien classés comptent plus. On divise par
le DCG idéal pour obtenir une valeur entre 0 et 1. Utile quand il y a plusieurs degrés de pertinence.

**95. Precision@k ?**
Parmi les k premiers résultats, la proportion de pertinents. Importante quand le contexte est cher en
tokens.

**96. Comment évaluer la génération ?**
Fidélité (soutenue par le contexte), pertinence de la réponse, précision et rappel du contexte.
Combinaison de LLM-juge validé sur échantillon et relecture humaine.

**97. Qu'est-ce que RAGAS / TruLens ?**
Des cadres d'évaluation de RAG avec des métriques comme la fidélité, la pertinence de la réponse et du
contexte (le « RAG triad »), souvent basées sur un LLM-juge.

**98. Quels sont les biais d'un LLM-juge ?**
Préférence pour les réponses longues, pour son propre style, effet de position (première réponse).
Parades : critères précis, échelle simple, juge différent du générateur, **calibrage sur notes
humaines**.

**99. Quelle taille pour un jeu d'évaluation ?**
Quelques dizaines pour détecter de gros écarts, des centaines pour comparer finement. La **diversité**
des cas compte autant que le nombre.

**100. Comment construire un golden set ?**
À partir de vraies questions d'utilisateurs, annotées avec le bon document (ou passage) et une réponse
de référence, relues par un expert, versionnées.

**101. Pourquoi inclure des questions sans réponse ?**
Pour vérifier que le système **refuse** au lieu d'inventer. Sans elles, on récompense l'invention.

**102. Qu'est-ce qu'une ablation ?**
Retirer une brique (BM25, reranker...) et mesurer l'effet pour prouver qu'elle sert. Si elle n'apporte
rien, on la supprime.

**103. Comment éviter les régressions ?**
Évaluation rejouée en CI avec un seuil minimal (`--min-hit-rate`), tests unitaires de la logique
déterministe, jeux de questions versionnés.

**104. L'évaluation humaine reste-t-elle nécessaire ?**
Oui, au moins par échantillon, pour calibrer les métriques automatiques et juger ce qu'elles ne voient
pas (ton, utilité, nuance).

**105. Comment faire un A/B test en production ?**
Répartir les utilisateurs entre deux variantes, mesurer retours, taux de reformulation, taux de « je ne
sais pas », latence et coût, avec assez de volume.

**106. Comment utiliser le retour utilisateur ?**
Pouce haut/bas par réponse, journalisé avec la question et les sources ; les réponses mal notées
nourrissent le jeu d'évaluation.

**107. Une réponse est fausse : comment déboguer ?**
1) Le bon passage est-il dans le corpus ? 2) Est-il récupéré (rang) ? 3) Est-il dans le contexte envoyé
(coupé par `max_chars` ?) ? 4) Le prompt et le modèle l'ont-ils utilisé ? On remonte la chaîne.

**108. Pourquoi 100 % sur un jeu ne veut rien dire ?**
Si le jeu est petit et facile, tout variante réussit : il ne distingue rien. Mon jeu de 10 questions
donne 100 % partout ; je dois l'étoffer avant de tirer des conclusions.

**109. Quelles métriques opérationnelles suivre ?**
Latence (par étape), taux d'erreur, tokens et coût, taux de refus, taille du contexte.

**110. Comment détecter une dérive de qualité ?**
Rejouer l'évaluation régulièrement (le corpus et les modèles changent), suivre les retours utilisateurs
et le taux de refus, relire des échantillons.

---

## G. Sécurité et conformité

**111. Injection de prompt directe et indirecte ?**
Directe : l'utilisateur donne l'ordre. **Indirecte** : un document du corpus contient des instructions
cachées ; c'est le risque principal d'un RAG. Défenses en couches (aucune n'est suffisante seule).

**112. Comment implémenter le contrôle d'accès dans un RAG ?**
Filtre par métadonnées (droits, client) **à la recherche**, avant que le LLM voie quoi que ce soit.
Jamais « le LLM ne révélera pas ».

**113. Quels enjeux RGPD ?**
Les questions et passages partent chez le fournisseur du LLM : base légale, accord de traitement,
localisation des données, durée de conservation, droit à l'effacement (supprimer aussi dans l'index).

**114. Les logs peuvent-ils fuiter des données ?**
Oui : ne pas journaliser le contenu des documents ni des questions sensibles, restreindre l'accès aux
logs.

**115. Qu'est-ce que l'empoisonnement du corpus ?**
Un document erroné ou malveillant indexé qui influence les réponses. On contrôle qui peut ajouter des
documents et on garde la traçabilité.

**116. Comment faire du multi-tenant ?**
Une collection par client, ou un champ `tenant_id` filtré systématiquement, avec des tests qui
vérifient qu'un client ne voit jamais les données d'un autre.

**117. Et si les données sont très sensibles ?**
Modèle auto-hébergé, ou fournisseur avec garanties contractuelles, masquage des données personnelles,
minimisation de ce qui est indexé.

**118. Qu'est-ce qu'un jailbreak ?**
Une tentative de contourner les consignes du modèle. On ne s'y fie pas au prompt seul : limites de
permissions, validation en code, supervision.

**119. Comment sécuriser l'envoi de fichiers ?**
Liste d'extensions autorisées, taille maximale, nom de fichier nettoyé (anti *path traversal*), stockage
hors du dossier public, analyse antivirus si besoin.

**120. Comment sécuriser l'API ?**
Authentification (clé, JWT/SSO), HTTPS, limitation de débit, validation des entrées, erreurs sans détails
internes. Comparaison de clé en temps constant.

**121. Comment masquer les données personnelles ?**
Détection (règles ou modèle) puis remplacement avant indexation ou avant envoi au LLM, en gardant une
table de correspondance si on doit réidentifier.

**122. Comment garantir l'auditabilité ?**
Tracer qui a demandé quoi, quels passages ont été fournis, quelle version du modèle et du prompt. Les
citations aident à justifier une réponse.

---

## H. Production et architecture

**123. Décris l'architecture de déploiement.**
Interface et API dans des conteneurs derrière un reverse proxy HTTPS avec authentification ; base
vectorielle sur volume sauvegardé ; secrets par variables d'environnement ; CI qui teste et construit.

**124. Où passe le temps d'une requête ?**
Embedding de la question, recherche vectorielle et BM25 : millisecondes. Reranking : dizaines à
centaines de ms. **Le LLM : secondes**, c'est le poste dominant.

**125. Comment réduire le coût ?** *(version courte ; détail en section I, questions 143 à 200)*
Moins de passages, meilleur reranking, compression du contexte, cache, modèle moins cher pour les
tâches simples, limiter la sortie.

**126. Quel cache mettre ?** *(détail en section I, questions 154 à 176)*
Cache exact (même question), cache **sémantique** (question proche), cache des embeddings. Attention à
l'invalidation quand les documents changent.

**127. Comment passer à l'échelle ?**
Plusieurs instances d'API sans état, base vectorielle serveur, BM25 dans un moteur dédié, ingestion
asynchrone par lots, files de tâches.

**128. Pourquoi des tâches asynchrones ?**
Une matrice de conformité envoie un appel LLM par exigence : trop long pour une requête HTTP. Un job avec
suivi de progression évite les expirations.

**129. Comment ré-indexer sans interruption ?**
Construire la nouvelle collection en parallèle (`v2`), valider avec l'évaluation, puis basculer
l'application ; garder l'ancienne pour revenir en arrière.

**130. Que monitorer ?**
Latence par étape, erreurs, tokens/coût, taux de refus, retours utilisateurs, fraîcheur de l'index,
espace disque/mémoire, sauvegardes.

**131. Comment gérer les limites de débit du fournisseur ?**
Réessais avec attente croissante, file d'attente, limitation côté client, repli vers un autre modèle ou
fournisseur.

**132. Que faire si le fournisseur LLM est indisponible ?**
Erreur 502 explicite, réessais, modèle de secours derrière la même interface, et dégradation douce
(afficher au moins les passages trouvés).

**133. Comment ingérer de grands volumes ?**
Traitement par lots, parallélisme, reprise sur erreur, file d'attente, suivi de l'avancement, et upserts
idempotents.

**134. Comment garder cohérents l'index vectoriel et l'index lexical ?**
Une seule source de vérité et une procédure de rafraîchissement déclenchée à chaque écriture. Mon bug :
BM25 n'était pas rafraîchi après upload ou suppression.

**135. Que sauvegarder ?**
Les **documents d'origine** (irremplaçables) et l'index (reconstructible mais long), hors du serveur,
avec une restauration testée.

**136. À quoi ressemble la CI d'un RAG ?**
Lint, tests unitaires sans réseau (faux modèles), build des images, et idéalement évaluation de la
recherche avec un seuil.

**137. Que versionner ?**
Code, prompts, modèle d'embedding (dans le nom de collection), modèle de reranking, modèle de
génération, jeu d'évaluation, configuration.

**138. Qu'est-ce que la traçabilité (tracing) d'une requête RAG ?**
Enregistrer chaque étape (question, candidats, rangs, contexte, réponse, latences) pour reproduire et
déboguer une réponse précise. Des outils existent (Langfuse, OpenTelemetry...).

**139. Comment viser la haute disponibilité ?**
Plusieurs instances derrière un répartiteur de charge, base vectorielle répliquée, sauvegardes,
healthchecks et redémarrage automatique.

**140. Comment tester sans appeler le LLM ni charger de modèle ?**
Des **faux** (fakes) pour le LLM et la recherche, injectés par interface. Mes 53 tests tournent en
quelques secondes sans réseau.

**141. Pourquoi Docker pour un RAG ?**
Environnement identique partout, dépendances lourdes (PyTorch) figées, modèles intégrés à l'image pour
un démarrage rapide, déploiement reproductible.

**142. Comment gouverner les documents indexés ?**
Propriétaire, date, version, durée de vie, processus d'ajout/retrait, qualité minimale. Un RAG est aussi
bon que son corpus.

---

## I. Coût des appels LLM et cache

Le cours détaillé est le **module 16** de [`formation-rag.md`](formation-rag.md). Les questions 125 et
126 (section H) en donnent la version courte ; celles-ci vont plus loin.

### I.1 La question centrale

**143. Comment minimises-tu le coût des appels au LLM ?** *(la question d'entretien)*
Je **mesure** d'abord (tokens entrée/sortie par fonctionnalité) pour trouver le poste dominant, puis
j'agis sur quatre leviers : (1) **moins d'appels** (cache exact et sémantique, ne pas appeler quand
aucun passage n'est trouvé, regrouper), (2) **moins de tokens par appel** (retrieval précis,
reranking, plafond de contexte, sortie courte, prompt caching), (3) **un modèle adapté** (petit pour
les tâches simples, gros pour le raisonnement, cascade), (4) **des garde-fous** (budgets, quotas,
tentatives bornées). Je vérifie chaque fois la qualité avec le jeu d'évaluation.

**144. Comment est calculé le coût d'un appel ?**
`tokens_entrée × prix_entrée + tokens_sortie × prix_sortie`. La sortie coûte en général plusieurs fois
plus cher par token que l'entrée. Avec un modèle de raisonnement, les tokens de réflexion comptent
comme de la sortie.

**145. Dans un RAG, où part le coût ?**
Surtout dans l'**entrée**, car le contexte (les passages) est bien plus long que la question. D'où
l'importance du `top_k`, du reranking et du plafond de contexte. La sortie pèse aussi (réponses
longues, raisonnement).

**146. Quel est le premier geste avant d'optimiser ?**
Mesurer. Journaliser les tokens et le coût par appel, par fonctionnalité et par client. Sans cela on
optimise le mauvais poste.

**147. Comment estimer le coût d'un projet avant de le lancer ?**
Tokens moyens d'entrée et de sortie par requête × volume prévu × tarif, avec une marge, puis un
**pilote** pour remplacer les estimations par des mesures. Exemple à tarif fictif : environ 1 200
tokens en entrée et 250 en sortie pour une question, selon la taille des passages.

**148. Pourquoi la qualité doit-elle rester dans l'équation ?**
Une économie qui dégrade les réponses coûte plus cher ensuite (reformulations, erreurs, perte de
confiance). Chaque optimisation est validée sur le jeu d'évaluation avant d'être gardée.

### I.2 Réduire le nombre d'appels

**149. Comment éviter d'appeler le LLM ?**
Cache, court-circuit quand aucun passage n'est trouvé (message fixe, c'est le cas dans mon projet),
seuil de pertinence, règles pour les questions triviales, réponse extractive pour les faits simples.

**150. Pourquoi ne pas appeler le LLM quand la recherche ne trouve rien ?**
Il inventerait, ce qui est faux **et** payant. Un message fixe est gratuit, déterministe et plus sûr.

**151. Quand regrouper plusieurs tâches dans un seul appel ?**
Pour des tâches identiques et indépendantes (ma découpe de RFP en lots de 10 000 caractères). Limite :
trop regrouper fait oublier des éléments au modèle et rend l'échec plus coûteux à rejouer.

**152. Qu'est-ce qu'une API batch ?**
Une interface où l'on soumet beaucoup de requêtes à traiter en différé, souvent à environ moitié prix.
Adaptée à l'évaluation, à l'enrichissement hors ligne, à la ré-indexation, pas au chat en temps réel.

**153. Les tentatives automatiques coûtent-elles ?**
Oui : chaque tentative est facturée. Mon `GroqLLM` en fait au plus 3 sur réponse vide. Il faut plafonner
et traiter la cause (budget de tokens trop bas pour un modèle de raisonnement) plutôt que réessayer.

### I.3 Le cache

**154. Quels types de cache connais-tu pour un RAG ?**
Exact (même requête), sémantique (requête de sens proche), cache d'embeddings, et prompt caching
(côté fournisseur, sur le préfixe du prompt). Gains et risques différents.

**155. Comment fonctionne un cache exact ?**
On calcule une clé (hash d'une requête normalisée et de ce qui influence la réponse). Si elle existe,
on renvoie la réponse stockée ; sinon on appelle le LLM et on stocke.

**156. Que mets-tu dans la clé de cache ?**
La question normalisée, le **modèle**, la **version du prompt**, la **version de l'index** (ou les IDs
des chunks), le **tenant** et les paramètres (`top_k`, température). Oublier un élément = servir une
mauvaise réponse.

**157. Pourquoi normaliser la question ?**
Pour que « Quel est le SLA ? » et « quel est le sla ? » partagent la même entrée. On met en minuscules
et on compresse les espaces ; on évite de normaliser trop (ne pas fusionner deux questions
différentes).

**158. Qu'est-ce qu'un cache LRU et un TTL ?**
LRU : quand le cache est plein on retire l'entrée la moins récemment utilisée. TTL : durée de vie
au-delà de laquelle l'entrée expire. Le premier borne la mémoire, le second la péremption.

**159. Comment invalides-tu le cache quand les documents changent ?**
Mettre une **version de l'index** dans la clé (elle change à chaque ingestion), ou calculer la clé sur
le prompt complet (le contexte change donc la clé change), plus un TTL de sécurité. Purger à la
suppression d'un document (et pour le droit à l'effacement).

**160. Qu'est-ce qu'un cache sémantique ?**
On vectorise la question ; si une question déjà vue a une similarité supérieure à un seuil, on renvoie
sa réponse. Il couvre les reformulations que le cache exact rate.

**161. Comment choisis-tu le seuil du cache sémantique ?**
Sur des paires de questions réelles étiquetées « même réponse / réponse différente ». Je choisis un
seuil **prudent** : un hit raté coûte un appel, un faux hit sert une fausse réponse.

**162. Quel est le risque principal du cache sémantique ?**
Le **faux hit** : « résilier » et « renouveler » un contrat sont proches en embedding mais ont des
réponses opposées. À proscrire ou à régler très haut dans les domaines à enjeu (juridique,
conformité).

**163. Cache exact ou sémantique : lequel choisir ?**
D'abord l'exact (simple, sans risque de faux hit). Je mesure ensuite la part de questions
reformulées ; si elle est importante et le domaine tolérant, j'ajoute le sémantique.

**164. Qu'est-ce que le prompt caching ?**
Le fournisseur mémorise le traitement d'un préfixe de prompt identique entre appels et le facture
bien moins cher à la relecture. Ce n'est pas un cache de réponses : le modèle génère toujours une
nouvelle réponse.

**165. Comment structurer un prompt pour profiter du prompt caching ?**
Contenu **stable en premier** (instructions, exemples, schéma, documents fixes), contenu **variable à
la fin** (la question). Un caractère changé dans le préfixe invalide le cache.

**166. Quelles sont les limites du prompt caching ?**
Taille minimale de préfixe (de l'ordre du millier de tokens), durée de vie courte (minutes),
écriture légèrement plus chère. Rentable seulement si le préfixe est réutilisé souvent. Je vérifie la
grille tarifaire actuelle avant de chiffrer.

**167. Quand le prompt caching est-il intéressant dans un RAG ?**
Pour un **long** prompt système ou un document de référence réutilisé à chaque appel. Moins pour
mon cas où le prompt système est court et le contexte change à chaque requête.

**168. Qu'est-ce que le cache stampede et comment l'éviter ?**
Quand une entrée populaire expire, de nombreuses requêtes identiques arrivent en même temps et
déclenchent autant d'appels. Remède : *single-flight* (un seul appel, les autres attendent), expiration
étalée (jitter), renouvellement anticipé.

**169. Peut-on mettre en cache une réponse générée avec température > 0 ?**
Oui, mais on fige une réponse parmi plusieurs possibles. Acceptable pour du factuel ; je garde une
température basse pour les cas à cacher.

**170. Quelle réponse mettre en cache ?**
Seulement celles qui ont **passé la validation** (citations existantes, format correct). Cacher une
mauvaise réponse la ressert longtemps.

**171. Comment éviter les fuites de données par le cache ?**
Inclure le **tenant ou les droits** dans la clé (ou le périmètre du cache sémantique), tester qu'un
utilisateur ne reçoit jamais la réponse d'un autre, chiffrer le stockage, inclure le cache dans la
procédure d'effacement RGPD.

**172. Où stocker le cache ?**
Mémoire (LRU) pour un prototype ; SQLite si on veut survivre aux redémarrages ; **Redis** dès qu'il y a
plusieurs instances (partagé, TTL natif) ; base vectorielle pour le sémantique.

**173. Comment mesure-t-on l'efficacité d'un cache ?**
Le **taux de hit** (hits / total), l'économie réelle (appels évités × coût moyen − coût du cache), et
la qualité des réponses servies (échantillon relu). Un taux de hit très bas = cache à retirer.

**174. Le cache sémantique est-il toujours rentable ?**
Non. Chaque requête paie un embedding et une recherche. Si le taux de hit est faible, il coûte plus
qu'il ne rapporte et ajoute du risque.

**175. Le cache change-t-il le comportement du système ?**
Oui, il peut masquer une dégradation (on resservirait l'ancienne réponse) et complique les tests.
Donc : versionner les clés, journaliser hit/miss, pouvoir le vider, le désactiver en évaluation.

**176. Comment tester un cache ?**
Avec un **faux LLM** qui compte ses appels : deux requêtes identiques → 1 appel ; changement de
tenant, de prompt ou d'index → 2 appels ; expiration avec une horloge injectée ; N threads simultanés
→ 1 appel (*single-flight*).

### I.4 Réduire les tokens

**177. Comment réduire les tokens d'entrée sans perdre en qualité ?**
Meilleur retrieval et reranking (moins de passages, mieux choisis), suppression des doublons, chunks
propres, plafond de contexte, compression extractive, prompt concis.

**178. Comment réduire les tokens de sortie ?**
Demander un format court, limiter `max_tokens`, JSON sans champs inutiles, consigne de concision.
Utile car la sortie est la plus chère.

**179. `max_tokens` est-il un coût ?**
Non, c'est un **plafond** : on paie ce qui est produit. Mais avec un modèle de raisonnement, un plafond
très élevé autorise une longue réflexion coûteuse, et un plafond trop bas donne une réponse vide
(mon bug). On le règle par la mesure.

**180. Plus de contexte améliore-t-il toujours la réponse ?**
Non : plus de coût, de latence et de bruit, et l'information « perdue au milieu ». Réduire le contexte
améliore souvent coût et qualité ensemble si le reranking est bon.

**181. Qu'est-ce que la compression de contexte ?**
Réduire les passages avant l'envoi : extraire les phrases pertinentes, ou résumer. Gain de tokens, au
prix d'un calcul supplémentaire et d'un risque de perdre un détail.

**182. Le reranking réduit-il le coût ?**
Indirectement : il permet d'envoyer 3 passages au lieu de 10 sans perdre le bon, donc moins de tokens
d'entrée. Il a lui-même un coût (CPU), bien inférieur à celui du LLM.

**183. Comment la taille des chunks influence-t-elle le coût ?**
Des chunks plus gros = plus de tokens par passage envoyé. Des chunks trop petits obligent à en envoyer
plus. On cherche l'optimum par la mesure (hit@k contre tokens).

### I.5 Choisir le modèle

**184. Qu'est-ce que le routage de modèles ?**
Envoyer chaque requête vers le modèle le moins cher qui suffit : un petit modèle pour classer,
extraire, reformuler ; un gros pour le raisonnement complexe. Par règles ou par un classifieur.

**185. Qu'est-ce qu'une cascade ?**
On essaie le petit modèle d'abord ; si la validation échoue ou la confiance est basse, on escalade vers
le gros. Piège : si le petit échoue souvent, on paie les deux appels.

**186. Comment décider qu'un petit modèle suffit ?**
En l'évaluant sur **mon** jeu de tâches : si la qualité est équivalente sur cette tâche, je l'adopte.
Je ne me fie pas aux benchmarks génériques.

**187. Distillation ou fine-tuning pour baisser le coût ?**
Pour un cas d'usage stable et à fort volume : un petit modèle entraîné à imiter un grand sur la tâche
coûte moins cher à chaque appel. À justifier par le volume, car l'entraînement et la maintenance ont
un coût.

**188. Modèle local ou API ?**
Local : pas de coût par token, confidentialité, mais coût d'infrastructure (GPU), d'exploitation et
qualité souvent moindre. API : zéro infrastructure, paiement à l'usage. On compare au volume.

**189. Quelle tâche de mon projet irait à un petit modèle ?**
L'extraction des exigences d'un RFP (JSON, température 0) est un bon candidat ; la rédaction de la
proposition resterait sur un modèle plus fort. À valider par l'évaluation.

### I.6 Garde-fous et production

**190. Comment éviter une facture surprise ?**
Budgets et alertes, plafonds de tokens et de tentatives, limitation de débit par utilisateur, taille
maximale des entrées, nombre maximal d'étapes pour un agent, et tableau de bord du coût.

**191. Qu'est-ce que le backoff exponentiel ?**
Attendre de plus en plus longtemps entre deux tentatives (0,5 s, 1 s, 2 s...), avec un nombre maximal.
Ménage le fournisseur et évite de multiplier les appels pendant une panne.

**192. Comment suivre le coût par client ?**
Journaliser tokens et modèle avec un identifiant de client à chaque appel, agréger par jour. Utile pour
la facturation, les quotas et repérer les abus.

**193. Que faire quand le fournisseur augmente ses tarifs ?**
L'interface `LLM` abstraite permet de changer de fournisseur ou de modèle sans refonte ; le jeu
d'évaluation dit si le remplaçant tient la qualité.

**194. Le parallélisme réduit-il le coût ?**
Non, il réduit la **latence**. Il peut même coûter plus s'il déclenche des erreurs de limite de débit
et des tentatives. Pour un RFP de 80 exigences, je paralléliserais avec un nombre de threads borné.

**195. Comment arbitrer coût, latence et qualité ?**
Ce sont trois curseurs liés. Je fixe d'abord le seuil de qualité acceptable (par l'évaluation), puis
je cherche la moins chère des options qui le respecte et tient la latence cible. Je documente le choix.

**196. Comment présenter à un client l'économie réalisée ?**
Avec des chiffres mesurés : coût par requête avant/après, taux de hit du cache, qualité inchangée sur le
jeu d'évaluation. Jamais avec une estimation non vérifiée.

**197. Quelles optimisations de coût sont dangereuses ?**
Cache sémantique à seuil bas, contexte tronqué trop agressivement, petit modèle sans évaluation,
suppression des citations pour gagner des tokens. Elles économisent en dégradant la fiabilité.

**198. Et si on te demande de réduire le coût de 50 % demain ?**
1) Je mesure et j'identifie le poste dominant. 2) Gains sûrs d'abord : cache exact, plafonds, tentatives
bornées, `top_k` minimal validé par l'évaluation. 3) Routage vers un petit modèle pour les tâches
simples, validé par l'évaluation. 4) Seulement ensuite des techniques plus risquées (sémantique,
compression). J'annonce un objectif conditionné à la qualité maintenue.

**199. Que répondre si on te dit « le cache règle tout » ?**
Il règle les requêtes **répétées**. Si les questions sont toutes différentes, le taux de hit est
faible. Il faut mesurer la répétition réelle, et le cache ajoute des risques de péremption et de fuite.

**200. Quelle est la différence entre réduire le coût et réduire la latence ?**
Souvent alignés (moins de tokens = moins cher et plus rapide), parfois opposés (paralléliser réduit la
latence sans réduire le coût ; un cache aide les deux). Je précise toujours l'objectif visé.
