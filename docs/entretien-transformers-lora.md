# Entretien Transformers, LLM et LoRA : questions et réponses

Plus de 200 questions classées par thème, avec des réponses **courtes**, comme on les donnerait à l'oral.
Le cours complet est dans [`formation-transformers-lora.md`](formation-transformers-lora.md), le code dans
[`labs/mini_transformer_lora.py`](labs/mini_transformer_lora.py). Pour le RAG et le coût des appels, voir
[`entretien-rag.md`](entretien-rag.md).

**Comment t'entraîner** : cache la réponse, réponds à voix haute en 30 à 60 secondes, compare. Pour les
formules et le code, **écris de mémoire** avant de vérifier.

## Sommaire
A. LLM et tokenisation · B. Attention et Transformer · C. Pré-entraînement · D. Inférence et KV cache ·
E. Mémoire et entraînement distribué · F. Choisir et fine-tuning · G. **LoRA** · H. QLoRA et variantes ·
I. Données, alignement et évaluation · J. Quantification et déploiement ·
K. Conception de systèmes · L. Questions pièges · M. Exercices de code · N. Les erreurs à éviter

---

## A. LLM et tokenisation

**1. Qu'est-ce qu'un LLM ?**
Un réseau de neurones (généralement un Transformer decoder-only) qui prédit la **distribution de
probabilité du token suivant** à partir des précédents. Répondre, résumer ou coder découle de cette
prédiction répétée.

**2. Qu'est-ce que la génération autorégressive ?**
On tire un token, on l'ajoute à l'entrée, on recommence. La probabilité d'une séquence est le produit des
probabilités conditionnelles `P(tᵢ | t<ᵢ)`.

**3. Pourquoi le Transformer a-t-il remplacé les RNN et LSTM ?**
Les RNN lisent séquentiellement (impossible à paralléliser) et oublient le début des longues séquences. Le
Transformer relie **directement** tous les tokens par l'attention et calcule tout en **parallèle**, ce qui
permet de passer à l'échelle.

**4. Qu'est-ce qu'un token ?**
Un morceau de texte du vocabulaire (mot fréquent, sous-mot, caractère ou octet). Le modèle manipule des
identifiants entiers, et limites comme coûts se comptent en tokens.

**5. Pourquoi des sous-mots plutôt que des mots ou des caractères ?**
Les mots entiers donnent un vocabulaire énorme et des mots inconnus ; les caractères donnent des séquences
très longues. Les sous-mots sont le compromis : mots fréquents en un token, mots rares découpés, aucun mot
inconnu.

**6. Explique BPE.**
On part de petites unités (octets) et on **fusionne itérativement la paire adjacente la plus fréquente** en
un nouveau token, jusqu'à la taille de vocabulaire voulue. À l'encodage, on rejoue les fusions dans l'ordre
appris.

**7. Quel est l'intérêt du BPE au niveau des octets ?**
L'alphabet de départ est les 256 octets : **tout texte** (accents, emoji, autres alphabets) est encodable,
sans jamais de token « inconnu ».

**8. BPE, WordPiece, Unigram : la différence ?**
BPE fusionne la paire la plus **fréquente**. WordPiece choisit les fusions qui maximisent la
**vraisemblance** (BERT). Unigram part d'un gros vocabulaire et **retire** les tokens les moins utiles
(SentencePiece).

**9. Pourquoi le français ou l'arabe coûtent-ils plus de tokens que l'anglais ?**
Le vocabulaire est appris surtout sur de l'anglais : les autres langues sont découpées en morceaux plus
petits. Même texte = plus de tokens = plus cher et moins de contexte utile.

**10. Pourquoi les LLM comptent-ils mal les lettres d'un mot ?**
Ils voient des tokens, pas des lettres. « fraise » peut être un seul token : les lettres ne sont pas
directement accessibles.

**11. Pourquoi l'arithmétique est-elle difficile ?**
Les nombres sont découpés de façon irrégulière (« 1234 » en « 12 » et « 34 »), et le modèle apprend des
motifs statistiques, pas un algorithme exact.

**12. Qu'est-ce que les tokens spéciaux ?**
Début et fin de séquence, séparateurs de rôles d'un chat, etc. Les oublier ou en utiliser de mauvais
dégrade le fine-tuning sans erreur visible (par exemple un modèle qui ne s'arrête jamais faute de token de
fin).

**13. Peut-on changer le tokenizer d'un modèle ?**
Pas sans ré-entraîner l'embedding et la sortie : le tokenizer est lié au modèle. Étendre le vocabulaire
demande un entraînement supplémentaire.

**14. Qu'est-ce qu'un embedding de tokens ?**
Une ligne d'une matrice `V × d` apprise : chaque identifiant est transformé en un vecteur de dimension `d`.

**15. Qu'est-ce que le *weight tying* ?**
Réutiliser la matrice d'embedding comme matrice de projection de sortie : économise `V × d` paramètres et
couple lecture et écriture des tokens.

**16. Que sont les logits ?**
Les scores bruts, un par token du vocabulaire, produits en sortie avant le softmax.

**17. Que fait le softmax ?**
Il convertit des scores en probabilités positives qui somment à 1 : `softmax(zᵢ) = exp(zᵢ) / Σ exp(zⱼ)`.

**18. Qu'est-ce que la perplexité ?**
`exp(perte d'entropie croisée)` : le nombre moyen de tokens équiprobables entre lesquels le modèle hésite.
Plus bas = mieux.

**19. Quelle perte initiale attendre d'un modèle non entraîné ?**
`ln(V)` (distribution uniforme). Dans mon mini-modèle (`V = 30`) : 3,41 pour `ln 30 = 3,40`. Un écart
important signale un bug d'initialisation.

**20. Quelle est la différence entre un modèle de base et un modèle « instruct » ?**
Le modèle de base **complète** du texte (pré-entraîné). Le modèle instruct a subi un **SFT** puis un
alignement (RLHF/DPO) pour suivre des instructions et dialoguer.

---

## B. Attention et Transformer

**21. Écris la formule de l'attention.**
`Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V`.

**22. Quel est le rôle de Q, K et V ?**
`Q` (query) : ce que je cherche. `K` (key) : ce que je contiens pour être trouvé. `V` (value) :
l'information transmise. Les scores `Q·K` pondèrent la moyenne des `V`.

**23. Pourquoi diviser par `√d_k` ?**
Si `q` et `k` ont des composantes de variance 1, leur produit scalaire a une variance de `d_k`. Sans
normalisation, les scores sont grands, le softmax sature et les gradients s'annulent.

**24. Qu'est-ce que le masque causal et pourquoi ?**
Une matrice triangulaire qui met `−∞` sur les positions futures avant le softmax, pour que chaque token ne
voie que le passé. Il empêche la fuite du futur et permet d'entraîner **toutes les positions en un seul
passage**.

**25. Pourquoi plusieurs têtes d'attention ?**
Chaque tête a ses propres projections et peut se spécialiser (syntaxe, coréférence, position). On découpe
`d_model` en `h` têtes de dimension `d_model/h`, puis on concatène et on projette.

**26. Quelle est la complexité de l'attention ?**
`O(T² · d)` en temps et `O(T²)` en mémoire pour la matrice de scores (version naïve) : quadratique en la
longueur `T`.

**27. Comment FlashAttention améliore-t-il cela ?**
Il calcule l'attention **exacte** par blocs en gardant les calculs dans la mémoire rapide du GPU, sans
écrire la matrice `T × T` en mémoire principale. Mêmes résultats, bien moins de mémoire et plus rapide
(le goulot est la bande passante mémoire).

**28. FlashAttention est-il une approximation ?**
Non. C'est le même calcul réorganisé pour réduire les lectures/écritures mémoire.

**29. Qu'est-ce que la self-attention et la cross-attention ?**
Self : `Q`, `K`, `V` viennent de la **même** séquence. Cross : `Q` vient d'une séquence, `K` et `V` d'une
autre (encoder-decoder, multimodal).

**30. MHA, MQA, GQA : la différence ?**
MHA : chaque tête a ses `K`/`V`. MQA : **une seule** paire `K`/`V` partagée par toutes les têtes. GQA : les
têtes sont réparties en **groupes** qui partagent une paire. Le but est de réduire le KV cache.

**31. Pourquoi GQA plutôt que MQA ?**
GQA est un compromis : presque la qualité de MHA avec presque l'économie mémoire de MQA.

**32. Pourquoi l'attention a-t-elle besoin d'une information de position ?**
Elle est invariante à l'ordre : sans position, « le chien mord l'homme » et « l'homme mord le chien »
donnent les mêmes calculs.

**33. Quelles méthodes de position connais-tu ?**
Sinusoïdale (ajoutée), apprise (table), **RoPE** (rotation de `q` et `k`), ALiBi (malus selon la
distance).

**34. Explique RoPE.**
On découpe `q` et `k` en paires de dimensions et on fait tourner chaque paire d'un angle
`position × fréquence`. Le produit scalaire résultant ne dépend que de l'**écart** de position : la
position relative est capturée naturellement.

**35. Comment étendre le contexte d'un modèle RoPE ?**
Interpolation de positions, NTK-aware, YaRN : on compresse les angles pour rester dans la plage vue à
l'entraînement, avec un court fine-tuning sur longues séquences.

**36. Que fait une connexion résiduelle ?**
`x + f(x)` : chaque sous-couche ajoute une correction au flux principal. Elle permet aux gradients de
traverser de nombreuses couches.

**37. LayerNorm et RMSNorm ?**
LayerNorm soustrait la moyenne et divise par l'écart type. RMSNorm divise seulement par la racine de la
moyenne des carrés : plus simple, aussi efficace (Llama).

**38. Pre-norm ou post-norm ?**
Pre-norm (norme avant la sous-couche) est plus **stable** pour les modèles profonds. Post-norm est le
Transformer original.

**39. Que fait le feed-forward (FFN) ?**
Appliqué à **chaque token indépendamment** : élargit la dimension, applique une non-linéarité, la réduit.
Il porte la majorité des paramètres et une grande part des connaissances factuelles.

**40. Qu'est-ce que SwiGLU ?**
`W_down · (SiLU(W_gate x) ⊙ W_up x)` : une FFN à trois matrices avec un mécanisme de porte, `d_ff ≈ 8/3 d`
pour garder le même nombre de paramètres.

**41. Rôle respectif de l'attention et du FFN ?**
L'attention fait **communiquer** les tokens ; le FFN **calcule** sur chaque token. Les deux alternent.

**42. Encoder-only, decoder-only, encoder-decoder ?**
Encoder (BERT) : attention bidirectionnelle, classification/embeddings. Decoder (GPT, Llama) : causal,
génération. Encoder-decoder (T5) : traduction, résumé.

**43. Pourquoi le decoder-only domine-t-il ?**
Un seul objectif simple (le token suivant), s'entraîne sur tout texte brut, passe très bien à l'échelle et
gère nativement le prompting et le few-shot.

**44. BERT peut-il générer du texte ?**
Mal : il est bidirectionnel et entraîné à remplir des mots masqués, pas à prédire la suite.

**45. Compte les paramètres d'un Transformer.**
Par bloc ≈ `12 d²` (attention `4d²` + FFN `8d²`), total ≈ `12 N d² + V d`. GPT-2 small : `12·12·768² ≈ 85 M` +
embeddings `38,6 M` ≈ 124 M.

**46. Et pour Llama-2 7B ?**
Par couche : attention `4 d² = 67 M`, FFN `3 · d · d_ff = 3 · 4096 · 11008 = 135 M`, soit 202 M ; ×32 = 6,48 G ;
plus 262 M d'embeddings = **6,74 G**.

**47. Où sont la plupart des paramètres d'un bloc ?**
Dans le **FFN** (environ les deux tiers). D'où l'importance de l'adapter avec LoRA.

**48. Comment l'attention scale-t-elle avec la longueur du contexte ?**
Quadratiquement en calcul (sans astuce) ; le KV cache croît linéairement. D'où fenêtre glissante, GQA,
FlashAttention et modèles à coût linéaire.

**49. Que signifie « lost in the middle » ?**
Les modèles exploitent moins bien l'information placée au **milieu** d'un long contexte que celle du début
ou de la fin.

**50. Pourquoi le Transformer est-il parallélisable à l'entraînement mais pas à la génération ?**
À l'entraînement, tous les tokens sont connus : toutes les positions se calculent en parallèle (masque
causal). À la génération, chaque token dépend du précédent : séquentiel.

---

## C. Pré-entraînement

**51. Quel est l'objectif de pré-entraînement d'un LLM ?**
Minimiser l'entropie croisée de la prédiction du token suivant sur un énorme corpus de texte.

**52. Que sont les lois d'échelle ?**
La perte décroît de façon régulière (loi de puissance) avec le nombre de paramètres, de données et de
calcul (Kaplan, 2020).

**53. Qu'a montré Chinchilla ?**
Pour un budget de calcul donné, paramètres et tokens doivent croître ensemble : environ **20 tokens par
paramètre**. Beaucoup de modèles étaient sous-entraînés.

**54. Pourquoi les modèles récents sont-ils sur-entraînés par rapport à Chinchilla ?**
Le coût d'entraînement est payé une fois, celui d'**inférence** à chaque requête : un modèle plus petit,
entraîné sur beaucoup plus de tokens, est moins cher à servir à qualité comparable.

**55. Estime le calcul d'entraînement.**
`≈ 6 × N × D` FLOPs (`N` paramètres, `D` tokens). Un 7B sur 2 000 G de tokens : `6 × 7·10⁹ × 2·10¹² ≈ 8,4·10²²`.

**56. Quels sont les FLOPs par token ?**
Passe avant `≈ 2N`, entraînement (avant + arrière) `≈ 6N`.

**57. Quel optimiseur et quel planning de taux d'apprentissage ?**
AdamW (`β₂ ≈ 0,95`, weight decay ≈ 0,1), échauffement linéaire puis décroissance en cosinus, écrêtage de
gradient ≈ 1,0.

**58. Pourquoi un échauffement (*warmup*) ?**
Au début, les statistiques d'Adam sont imprécises et les gradients grands : un taux trop haut fait diverger.

**59. Pourquoi écrêter les gradients ?**
Pour limiter les pics de perte (*loss spikes*) causés par quelques mini-batchs aberrants.

**60. Pourquoi la déduplication des données est-elle importante ?**
Les doublons font **mémoriser** le modèle (fuite de données, moins de généralisation) et gaspillent du
calcul.

**61. Qu'est-ce que la contamination ?**
La présence dans les données d'entraînement de questions ou réponses d'un benchmark : les scores sont
gonflés.

**62. Qu'est-ce que le pré-entraînement continu ?**
Poursuivre le pré-entraînement sur un corpus de domaine ou de langue, pour de grandes adaptations de
vocabulaire ou de connaissances, avant un SFT.

**63. Qu'est-ce que le *curriculum* de données ?**
Ordonner ou mélanger les données (par qualité, difficulté, domaine) au cours de l'entraînement.

**64. Que signifie « émergence » ?**
Des capacités qui semblent apparaître brusquement avec la taille. Débat : une partie de l'effet dépend de
la métrique (discontinue) plutôt que du modèle.

**65. Pourquoi bf16 plutôt que fp16 ?**
Même étendue d'exposants que fp32 : pas de dépassement, pas besoin de *loss scaling*. fp16 a une plage
étroite et exige un facteur d'échelle.

**66. Comment vérifier qu'un entraînement est sain ?**
Perte initiale ≈ `ln V`, perte qui baisse, **sur-apprentissage d'un petit lot** (20 exemples), normes de
gradient stables, pas de NaN.

---

## D. Inférence et KV cache

**67. Quelles stratégies de décodage connais-tu ?**
Glouton, température, top-k, top-p (noyau), beam search, pénalité de répétition.

**68. Que fait la température ?**
Divise les logits avant le softmax : `T < 1` rend la distribution plus tranchée, `T > 1` plus plate,
`T → 0` équivaut au glouton.

**69. Top-k ou top-p ?**
Top-k garde un nombre **fixe** de candidats. Top-p garde le plus petit ensemble dont la probabilité cumulée
dépasse `p` : il **s'adapte** (peu de choix quand le modèle est sûr, plus quand il hésite).

**70. Quand utiliser une température basse ?**
Pour du factuel, de l'extraction, du JSON : 0 à 0,2. Pour la créativité : 0,7 à 1.

**71. Le décodage glouton est-il optimal ?**
Non : choisir le meilleur token à chaque pas ne donne pas la séquence la plus probable globalement. Le beam
search explore plusieurs candidats.

**72. Qu'est-ce que le prefill et le decode ?**
Prefill : lecture du prompt **en parallèle** (limité par le calcul), détermine le TTFT. Decode : un token à
la fois (limité par la **mémoire**), détermine le débit.

**73. Qu'est-ce que le KV cache ?**
On stocke les `K` et `V` des tokens déjà traités pour ne calculer que ceux du **nouveau** token à chaque
pas. Résultats identiques, bien moins de calcul.

**74. Pourquoi le cache est-il valide ?**
Grâce au masque causal : les `K` et `V` d'un token ne dépendent pas des tokens suivants, donc ne changent
jamais.

**75. Comment calcules-tu la taille du KV cache ?**
`2 × couches × têtes_KV × dim_tête × octets` par token. Llama-2 7B (32 couches, 32 têtes KV, dim 128, fp16) :
`2×32×32×128×2 = 512 Kio` par token, soit **2 Gio pour 4 096 tokens**.

**76. Comment GQA modifie-t-il ce calcul ?**
Il réduit le nombre de têtes KV. Avec 8 têtes au lieu de 32 : cache **divisé par 4** (128 Kio par token).

**77. Pourquoi le KV cache limite-t-il le nombre d'utilisateurs ?**
Il est **par requête** : 50 requêtes de 4 096 tokens sur un 7B sans GQA = 100 Gio de cache.

**78. Pourquoi l'inférence est-elle limitée par la mémoire ?**
À chaque token, on relit tous les poids pour très peu de calcul. Le débit dépend donc de la bande passante
mémoire.

**79. Pourquoi la quantification accélère-t-elle la génération ?**
Moins d'octets à lire par token. Comme le decode est limité par la mémoire, c'est un vrai gain de vitesse.

**80. Qu'est-ce que le batching continu ?**
Ajouter et retirer des requêtes du lot **à chaque pas** au lieu d'attendre la plus lente : meilleure
utilisation du GPU.

**81. Qu'est-ce que PagedAttention ?**
Le KV cache est découpé en pages, comme la mémoire virtuelle d'un OS : moins de fragmentation et de
gaspillage (vLLM).

**82. Qu'est-ce que le décodage spéculatif ?**
Un petit modèle propose plusieurs tokens, le grand les **vérifie en une passe**. Résultat identique, plus
rapide quand les propositions sont souvent acceptées.

**83. Qu'est-ce que le TTFT ?**
*Time to first token* : délai avant le premier token. Il dépend de la longueur du prompt (prefill).

**84. Comment réduire la latence perçue ?**
Streaming des tokens, modèle plus petit, prompt plus court, cache de préfixe, quantification.

**85. Comment contraindre une sortie JSON ?**
Décodage contraint par une grammaire (on masque les tokens invalides), mode JSON du fournisseur, puis
validation en code.

**86. Qu'est-ce que le cache de préfixe ?**
Réutiliser le KV cache de prompts qui partagent un début (instructions système) : économise le prefill.

**87. Pourquoi la même requête peut-elle donner des réponses différentes avec `T = 0` ?**
Non-déterminisme des calculs en virgule flottante sur GPU (ordre d'accumulation, batching) : de petits
écarts changent parfois un `argmax`.

**88. Qu'est-ce que la pénalité de répétition ?**
Réduire le score des tokens déjà générés pour éviter les boucles ; à doser car elle peut dégrader les
réponses qui doivent répéter des termes.

---

## E. Mémoire et entraînement distribué

**89. De quoi dépend la mémoire en inférence ?**
Poids (`paramètres × octets`) + KV cache + activations (petites) + marge.

**90. Combien de mémoire pour un 7B ?**
Poids : fp32 27 Go, bf16 **13,5 Go**, int8 6,7 Go, int4 **3,4 Go**, sans compter le KV cache.

**91. Pourquoi un fine-tuning complet de 7B demande-t-il environ 108 Go ?**
`16 octets par paramètre` : poids bf16 (2) + gradient (2) + copie maître fp32 (4) + moment `m` (4) + moment
`v` (4). Plus les activations.

**92. Quelle mémoire pour un 70B en fine-tuning complet ?**
`70 G × 16 ≈ 1,1 To` : plusieurs GPU et un partage des états (FSDP/ZeRO).

**93. Qu'est-ce que le gradient checkpointing ?**
On ne garde pas toutes les activations : on les **recalcule** pendant la passe arrière. Économise de la
mémoire au prix d'environ 30 % de calcul en plus.

**94. Qu'est-ce que l'accumulation de gradients ?**
Cumuler les gradients de plusieurs micro-batchs avant la mise à jour : gros batch effectif avec peu de
mémoire.

**95. DDP, FSDP/ZeRO, parallélisme tenseur et pipeline ?**
DDP : chaque GPU a une **copie** du modèle, on partage les données. ZeRO/FSDP : on **partage** états
d'optimiseur, gradients et poids entre GPU. Tenseur : on découpe les matrices. Pipeline : on découpe les
couches.

**96. Que partitionne ZeRO ?**
Stade 1 : états d'optimiseur. Stade 2 : + gradients. Stade 3 : + poids. Plus on partage, plus on économise
de mémoire, plus on communique.

**97. Qu'est-ce qui est limitant : calcul ou communication ?**
Cela dépend de l'échelle : sur de gros clusters, la **communication** entre GPU devient souvent le
goulot ; d'où le choix du type de parallélisme selon la topologie.

**98. Pourquoi les optimiseurs 8 bits ?**
Les états d'Adam occupent 8 octets par paramètre : les quantifier réduit fortement la mémoire.

---

## F. Choisir et fine-tuning

**99. Prompting, RAG ou fine-tuning : comment choisir ?**
Du moins au plus coûteux : prompt, RAG, fine-tuning. On monte seulement si l'échelon inférieur ne suffit
pas, **prouvé par une évaluation**. RAG pour les **connaissances**, fine-tuning pour le **comportement**.

**100. Pourquoi le fine-tuning n'est-il pas adapté pour ajouter des faits ?**
Il apprend mieux des motifs de style que des faits isolés, peut apprendre à répondre avec assurance sans
savoir (hallucination), ne cite pas, et il faut ré-entraîner à chaque changement.

**101. Quand fine-tuner ?**
Format ou style strict, vocabulaire ou raisonnement de domaine, consignes complexes fiables, **réduire le
coût** en faisant égaler un gros modèle par un petit sur une tâche étroite.

**102. Peut-on combiner RAG et fine-tuning ?**
Oui : fine-tuner pour mieux **utiliser** un contexte récupéré (citer, refuser, respecter le format) tout en
gardant le RAG pour les connaissances.

**103. Que vérifies-tu avant de fine-tuner ?**
Un prompt de référence qui échoue, une évaluation chiffrée, assez de données de qualité, le gain attendu
face au coût, les licences.

**104. Qu'est-ce que le SFT ?**
Entraînement supervisé sur des paires (instruction, réponse idéale), avec perte sur la réponse : il apprend
le format et le comportement.

**105. Qu'est-ce que l'oubli catastrophique ?**
La perte de capacités générales après un fine-tuning. Parades : taux d'apprentissage et rang modestes,
peu d'époques, données mélangées, évaluation de contrôle.

**106. Fine-tuning complet : avantages et inconvénients ?**
Capacité maximale, mais mémoire énorme, une copie complète par tâche, risque d'oubli et de
sur-apprentissage.

**107. Qu'est-ce que PEFT ?**
Entraîner un **petit nombre** de paramètres en gardant le modèle gelé : mémoire réduite, adaptateurs
minuscules et échangeables.

**108. Quelles familles de PEFT connais-tu ?**
Adapters (modules insérés), prompts doux et prefix tuning (vecteurs virtuels), BitFit (biais), IA³
(vecteurs multiplicatifs), **LoRA** (mise à jour de rang faible).

**109. Pourquoi LoRA a-t-il gagné ?**
Qualité proche du fine-tuning complet, **aucune latence ajoutée** après fusion, adaptateurs minuscules,
implémentation simple, compatible avec la quantification (QLoRA).

**110. Inconvénient des adapters classiques ?**
Les couches insérées **ajoutent de la latence** à l'inférence, contrairement à LoRA fusionné.

**111. Inconvénient du prompt tuning ?**
Il consomme du contexte, est moins stable et souvent moins performant, surtout sur les petits modèles.

**112. Combien d'exemples faut-il ?**
Style/format : 100 à 1 000. Tâche étroite : 500 à 5 000. Instructions variées : 1 000 à 50 000.
**Qualité avant quantité** : on commence petit et on ajoute là où le modèle échoue.

---

## G. LoRA

**113. Explique LoRA en 30 secondes.**
On **gèle** le modèle de base et on apprend, pour chaque matrice ciblée, une mise à jour de **rang faible**
`ΔW = B·A` avec `h = W₀x + (α/r)·B·A·x`. Très peu de paramètres à entraîner, adaptateur de quelques dizaines
de Mo, et on peut **fusionner** `W ← W + (α/r)BA` pour n'avoir aucune latence ajoutée.

**114. Quelle est l'hypothèse derrière LoRA ?**
La mise à jour des poids lors d'un fine-tuning a un **rang intrinsèque faible** : on peut l'approcher par
un produit de deux petites matrices (dimension intrinsèque faible des tâches d'adaptation).

**115. Donne les dimensions de A et B.**
`W₀ ∈ ℝ^(d_out × d_in)`, `A ∈ ℝ^(r × d_in)`, `B ∈ ℝ^(d_out × r)`, avec `r ≪ min(d_in, d_out)`.

**116. Combien de paramètres LoRA ajoute-t-il par matrice ?**
`r × (d_in + d_out)`, au lieu de `d_in × d_out`. Exemple : `r = 8`, matrice 4096×4096 : 65 536 au lieu de
16,8 M (0,4 %).

**117. Combien de paramètres pour un 7B ?**
`r = 8` sur `q` et `v` : **4,19 M** (0,06 %). `r = 16` sur toutes les couches linéaires : **40 M** (0,59 %).
`r = 64` : 160 M (2,4 %).

**118. Quelle taille a le fichier d'adaptateur ?**
`r = 16`, toutes couches, 7B, en bf16 : environ **80 Mo**, contre 13,5 Go pour le modèle.

**119. Pourquoi B est-il initialisé à zéro ?**
Pour que `ΔW = 0` au départ : le modèle démarre **exactement** comme le modèle de base.

**120. Pourquoi A n'est-il pas aussi initialisé à zéro ?**
Si les deux étaient nuls, les gradients de `A` et `B` seraient nuls (chacun dépend de l'autre) et rien
n'apprendrait. `A` aléatoire permet à `B` de recevoir des gradients dès le premier pas.

**121. Et si B et A étaient tous deux aléatoires ?**
Le modèle serait perturbé dès le départ (`ΔW ≠ 0`) : on perd la garantie de partir du modèle d'origine.

**122. Montre que B apprend avant A.**
Avec `s = α/r` et `g = ∂L/∂h` : `∂L/∂B = s·g·(Ax)ᵀ` et `∂L/∂A = s·Bᵀ·g·xᵀ`. À `B = 0`, le gradient de `A` est
nul mais celui de `B` ne l'est pas. Après un pas, `B ≠ 0` et `A` reçoit des gradients.

**123. À quoi sert α ?**
Facteur d'échelle de la mise à jour (`α/r`). Il permet de **changer `r` sans re-régler le taux
d'apprentissage**. Pratique courante : `α = r` ou `2r`.

**124. α augmente-t-il la capacité ?**
Non. Il règle l'**amplitude** de l'adaptation, pas le nombre de degrés de liberté (c'est `r`).

**125. Qu'est-ce que rsLoRA ?**
Utiliser l'échelle `α/√r` au lieu de `α/r`, plus stable pour de **grands rangs**.

**126. Comment choisir le rang ?**
Commencer à 16 ; monter (32 à 128) si le modèle sous-apprend sur une tâche difficile ; descendre à 4-8 pour
du style simple. Un rang plus grand n'est pas toujours mieux : le gain stagne selon la tâche.

**127. Sur quelles matrices appliquer LoRA ?**
Le papier original : `q` et `v`. En pratique : **toutes les couches linéaires** (`q, k, v, o, gate, up,
down`) ; QLoRA a montré que c'est important pour égaler le fine-tuning complet. PEFT :
`target_modules="all-linear"`.

**128. Pourquoi inclure les couches du FFN ?**
Le FFN porte la majorité des paramètres du bloc ; l'omettre limite la capacité d'adaptation.

**129. Quel taux d'apprentissage pour LoRA ?**
Environ `1e-4` à `2e-4`, bien plus haut que pour un fine-tuning complet (≈ `1e-5`), car on part de `B = 0`
et on entraîne peu de paramètres.

**130. Quels autres hyperparamètres ?**
`lora_dropout` (0 à 0,1), 1 à 3 époques, batch effectif 16 à 64, échauffement ≈ 3 % puis cosinus.

**131. LoRA réduit-il le temps de calcul ?**
Peu. Il économise surtout la **mémoire** (gradients et états d'Adam des poids gelés). La passe arrière doit
toujours traverser les poids gelés pour propager les gradients aux activations.

**132. Quelle mémoire LoRA économise-t-il exactement ?**
Gradients et états d'optimiseur des poids gelés. Pour un 7B, `r = 16`, toutes couches : adaptateurs ≈ 80 Mo,
gradients ≈ 80 Mo, états Adam ≈ 320 Mo : moins de 0,5 Go au lieu des ~108 Go d'un entraînement complet.

**133. Que reste-t-il comme coût mémoire ?**
Les poids du modèle (13,5 Go en bf16) et les **activations**, réduites par le gradient checkpointing.

**134. Comment fusionner un adaptateur ?**
`W ← W + (α/r)·B·A`. Plus de chemin LoRA à l'inférence : aucune latence ajoutée. La fusion est exacte, et
inversible tant qu'on conserve `A` et `B`.

**135. Fusionner ou garder séparé ?**
Fusionné : un seul usage, aucune surcharge. Séparé : **plusieurs adaptateurs** sur un même modèle de base,
changés par requête (service multi-adaptateurs).

**136. Comment servir 100 clients avec 100 fine-tunings ?**
Un seul modèle de base en mémoire et 100 adaptateurs LoRA (quelques dizaines de Mo chacun), sélectionnés
par requête avec un serveur multi-adaptateurs. Bien moins cher que 100 modèles complets.

**137. Peut-on combiner plusieurs adaptateurs ?**
Oui : somme pondérée, ou méthodes de fusion (TIES, DARE) ; le résultat est empirique, à évaluer.

**138. Quelles sont les limites de LoRA ?**
« Apprend moins et oublie moins » : sur des tâches qui demandent beaucoup de nouvelles capacités (code,
maths à grande échelle), il peut rester derrière le fine-tuning complet. Sensible aux hyperparamètres ;
n'ajoute pas bien de connaissances factuelles.

**139. LoRA oublie-t-il moins que le fine-tuning complet ?**
Généralement oui : il modifie le modèle dans un sous-espace restreint, ce qui préserve davantage les
capacités d'origine.

**140. Le rang de `BA` peut-il dépasser `r` ?**
Non : `rang(BA) ≤ r`. Les mises à jour sont contraintes à un sous-espace de rang au plus `r`.

**141. LoRA modifie-t-il la latence d'inférence ?**
Fusionné : aucune. Non fusionné : légère surcharge (deux petits produits matriciels en plus).

**142. Que prouve ton lab LoRA ?**
(1) Sortie identique à la base au départ. (2) Paramètres entraînés = `r(d_in + d_out)` par matrice. (3) Un
modèle pré-entraîné sur `n, n+1...` apprend `n, n+2...` avec 3 072 paramètres sur 120 512, perte
**5,97 → 0,02**. (4) Seuls `A` et `B` bougent. (5) La fusion est exacte.

**143. Que se passe-t-il si `lora_dropout` est trop haut ?**
Le chemin LoRA est trop bruité, l'apprentissage ralentit. Il aide surtout avec de petits jeux de données.

**144. LoRA est-il utilisable pour autre chose que le texte ?**
Oui : modèles de diffusion (styles d'image), modèles de vision, multimodaux, encoders (embeddings,
reranker).

**145. Peut-on fine-tuner les embeddings d'entrée avec LoRA ?**
On peut l'appliquer à la couche d'embedding, mais c'est moins courant ; si on ajoute des tokens spéciaux, il
faut entraîner (ou « sauvegarder ») les couches d'embedding et de sortie concernées.

**146. Comment LoRA se compare-t-il aux adapters de Houlsby ?**
Même esprit (petits modules entraînés), mais LoRA est **additif sur les poids** et fusionnable : zéro
latence supplémentaire.

**147. Comment choisir entre LoRA et fine-tuning complet ?**
Je commence par LoRA (peu de mémoire, rapide, réversible). Je passe au complet si l'évaluation montre une
limite de capacité (grand volume de données, nouvelle compétence difficile) et si j'ai le matériel.

**148. Pourquoi le rang suffit-il souvent à `8` ou `16` ?**
Pour des tâches de style, de format ou de consigne, la mise à jour utile est de rang faible. Les tâches plus
riches demandent plus de capacité.

**149. Comment déboguer un LoRA qui n'apprend pas ?**
Vérifier que les paramètres LoRA sont bien `requires_grad`, que `B` bouge, le taux d'apprentissage, les
modules ciblés, le masquage des labels, puis tenter de **sur-apprendre 20 exemples**.

**150. Que sauvegarde-t-on après l'entraînement ?**
Seulement l'adaptateur (`A`, `B` et la configuration : `r`, `α`, modules ciblés) ; on le recharge sur le
**même** modèle de base.

**151. Pourquoi faut-il le même modèle de base pour recharger un adaptateur ?**
`ΔW` est défini **relativement** à `W₀` : appliqué à d'autres poids, il n'a pas de sens.

---

## H. QLoRA et variantes

**152. Qu'est-ce que QLoRA ?**
LoRA appliqué à un modèle de base **quantifié en 4 bits** (NF4) et gelé, avec des adaptateurs en 16 bits.
Les gradients traversent les poids 4 bits. Il permet de fine-tuner un 65B sur un GPU de 48 Go.

**153. Quelles sont les trois innovations de QLoRA ?**
**NF4** (format 4 bits adapté aux poids normalement distribués), **double quantification** (on quantifie les
constantes de quantification), **optimiseurs paginés** (états d'Adam débordant vers la RAM CPU pour éviter
l'*out of memory*).

**154. Qu'est-ce que NF4 ?**
*4-bit NormalFloat* : 16 niveaux placés selon les quantiles d'une loi normale, donc plus précis qu'un int4
uniforme pour des poids de distribution normale.

**155. Pourquoi la double quantification ?**
Chaque bloc de poids a une constante de quantification stockée en 32 bits ; la quantifier à son tour
économise environ 0,4 bit par paramètre.

**156. Comment les calculs se font-ils avec des poids 4 bits ?**
Chaque bloc est **déquantifié à la volée en bf16** pour le calcul, puis jeté. Les adaptateurs et leurs
gradients restent en 16 bits.

**157. QLoRA est-il plus lent que LoRA ?**
Oui : la déquantification à la volée ajoute du temps. On échange **vitesse contre mémoire**.

**158. Combien de mémoire pour un 7B en QLoRA ?**
Poids NF4 ≈ 3,4 Go, adaptateurs + gradients + états < 0,5 Go, plus les activations (dépendent du batch et
de la longueur). Un GPU de 12 à 16 Go suffit souvent.

**159. QLoRA dégrade-t-il la qualité ?**
Le papier rapporte une qualité équivalente au fine-tuning 16 bits sur les benchmarks étudiés. Sur ta tâche :
**mesure**.

**160. Peut-on fusionner l'adaptateur dans le modèle 4 bits ?**
Mal : fusionner une mise à jour fine dans des poids déjà arrondis perd de la précision. On recharge le modèle
en 16 bits, on fusionne, puis on **re-quantifie** si besoin.

**161. Qu'est-ce que DoRA ?**
Il décompose le poids en **magnitude** et **direction** : LoRA adapte la direction, la magnitude est apprise
séparément. Souvent meilleur à faible rang, un peu plus coûteux.

**162. Qu'est-ce que LoRA+ ?**
Deux taux d'apprentissage : plus grand pour `B` que pour `A`. Apprentissage plus rapide et meilleur.

**163. Qu'est-ce qu'AdaLoRA ?**
Il **alloue le rang** dynamiquement entre les matrices selon leur importance, à budget total fixé.

**164. Qu'est-ce que VeRA ?**
`A` et `B` sont aléatoires, **partagés et gelés** ; on n'apprend que de petits vecteurs d'échelle : encore
moins de paramètres que LoRA.

**165. Qu'est-ce que LoftQ ?**
Une initialisation de LoRA qui **compense l'erreur de quantification** du modèle 4 bits : meilleur départ
pour QLoRA.

**166. Qu'est-ce que PiSSA ?**
Initialise l'adaptateur avec les composantes **principales** (SVD) du poids et gèle le résidu : convergence
plus rapide.

**167. Comment choisir une variante ?**
Je commence par LoRA/QLoRA standard sur toutes les couches linéaires. Je teste une variante seulement si
l'évaluation montre une limite, **sur la même évaluation**.

---

## I. Données, alignement et évaluation

**168. Qu'a montré LIMA ?**
Qu'un bon modèle de base affiné sur **1 000 exemples soignés** peut répondre très bien : le
pré-entraînement apporte les connaissances, le fine-tuning apprend surtout le **format et le style**.

**169. Qualité ou quantité des données ?**
**Qualité** d'abord : 500 exemples excellents valent mieux que 50 000 bruités. On étend ensuite là où le
modèle échoue.

**170. Que contient une bonne checklist de données ?**
Cohérence, exactitude, diversité, représentativité, déduplication, équilibre, anonymisation, absence de
contamination, jeu de validation séparé, versionnement.

**171. D'où viennent les données ?**
Données réelles nettoyées, annotation humaine, **données synthétiques** (modèle plus fort), et correction
des échecs du modèle actuel (souvent le meilleur rendement).

**172. Risques des données synthétiques ?**
Erreurs propagées, perte de diversité, et **conditions d'utilisation** du modèle générateur (certaines
interdisent d'entraîner un concurrent). Filtrer et relire un échantillon.

**173. Que sont le *chat template* et pourquoi est-il critique ?**
Le gabarit qui met les messages en forme avec les tokens spéciaux du modèle. Fine-tuner avec un gabarit et
inférer avec un autre dégrade silencieusement le comportement.

**174. Pourquoi masquer le prompt dans la perte ?**
Pour apprendre à **produire la réponse**, pas à recopier la question : labels du prompt à `-100`, perte
uniquement sur les tokens de l'assistant.

**175. Qu'est-ce que le *packing* ?**
Concaténer plusieurs exemples courts dans une même séquence pour remplir le contexte et accélérer
l'entraînement ; attention aux fuites d'attention entre exemples.

**176. Pourquoi inclure des exemples de refus ?**
Sinon le modèle apprend à **toujours répondre**, même sans information, ce qui augmente les
hallucinations.

**177. Que se passe-t-il si le token de fin manque dans les données ?**
Le modèle ne sait pas s'arrêter et génère jusqu'à la limite.

**178. Explique RLHF.**
SFT, puis entraînement d'un **modèle de récompense** sur des préférences humaines (perte de Bradley-Terry),
puis optimisation de la politique par renforcement (souvent PPO) avec une **pénalité KL** vers le modèle de
référence.

**179. Quels sont les problèmes de RLHF ?**
Complexe (politique, référence, récompense, valeur), instable, et **détournement de la récompense** (le
modèle exploite les failles du juge, par exemple des réponses longues et flatteuses).

**180. Explique DPO.**
On entraîne directement sur des paires (préférée, rejetée), sans modèle de récompense ni renforcement :
`L = −log σ(β[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])`. On augmente la probabilité de la
préférée **relativement à la référence**.

**181. Que vaut la perte DPO quand la politique égale la référence ?**
`ln 2 ≈ 0,693` : aucune préférence apprise (vérifié dans le lab).

**182. Que fait β dans DPO ?**
Il règle la force du maintien près de la référence : petit β = on s'autorise à s'éloigner davantage.
Valeur courante : 0,1.

**183. Comment DPO se combine-t-il avec LoRA ?**
La référence s'obtient **sans copie du modèle** en désactivant les adaptateurs. DPO + LoRA tient sur un seul
GPU.

**184. Qu'est-ce que ORPO, KTO, SimPO ?**
ORPO : SFT et préférence en une étape, sans référence. KTO : besoin d'un simple « bon/mauvais » par
réponse, pas de paires. SimPO : sans référence, avec la log-probabilité moyenne comme récompense.

**185. Qu'est-ce que RLAIF / Constitutional AI ?**
Les préférences viennent d'un **modèle** guidé par des principes écrits, au lieu d'humains.

**186. Qu'est-ce que GRPO ?**
Pour chaque prompt, on génère un **groupe** de réponses, on calcule leur récompense, et l'avantage de
chacune est `(r − moyenne du groupe) / écart type du groupe`. **Pas de modèle de valeur** : plus léger que
PPO. Utilisé pour les modèles de raisonnement.

**187. Qu'est-ce que le RLVR ?**
RL avec **récompenses vérifiables** (réponse mathématique correcte, tests de code qui passent), sans
modèle de récompense appris.

**188. Comment évalues-tu un modèle fine-tuné ?**
Contre **deux références** : le modèle de base avec un bon prompt, et ce même modèle avec RAG si pertinent.
Métriques de tâche sur un jeu de test séparé, benchmarks de contrôle (oubli), sécurité, format, coût et
latence.

**189. Pourquoi comparer à un bon prompt et pas à un prompt simple ?**
Un fine-tuning qui ne bat pas un prompt bien fait ne vaut pas son coût de maintenance.

**190. Pourquoi la perte de validation ne suffit-elle pas ?**
Elle mesure la vraisemblance des tokens, pas l'**utilité** des réponses ni leur exactitude.

**191. Quels sont les biais d'un LLM-as-judge ?**
Préfère les réponses longues, son propre style, la première position. Parades : critères précis, échelle
simple, juge différent du générateur, **calibrage sur notes humaines**.

**192. Comment détecter l'oubli catastrophique ?**
Rejouer quelques benchmarks généraux et des tests de capacités avant/après fine-tuning.

**193. Le fine-tuning peut-il affaiblir la sécurité ?**
Oui, même sur des données inoffensives : retester les refus et mélanger des exemples de sécurité.

**194. Pourquoi 10 exemples de test ne suffisent-ils pas ?**
Trop de variance pour distinguer deux modèles. Il faut des dizaines à des centaines d'exemples variés, et
vérifier avec plusieurs graines.

**195. Pourquoi ne pas choisir le meilleur checkpoint sur le jeu de test ?**
Le jeu de test devient alors un jeu d'entraînement et les scores sont optimistes. On choisit sur la
validation et on teste **une fois**.

**196. Qu'est-ce que la contamination d'un benchmark ?**
Des exemples du benchmark présents dans les données d'entraînement : les scores ne reflètent plus la
généralisation.

---

## J. Quantification et déploiement

**197. Qu'est-ce que la quantification ?**
Représenter les poids avec moins de bits (int8, int4) pour économiser mémoire et accélérer la génération.

**198. PTQ ou QAT ?**
PTQ : on quantifie un modèle déjà entraîné avec un petit jeu de calibration (simple). QAT : on simule la
quantification **pendant** l'entraînement (meilleure qualité, plus coûteux).

**199. GPTQ, AWQ, GGUF, NF4 ?**
GPTQ : PTQ 4 bits (information de second ordre), GPU. AWQ : PTQ 4 bits qui protège les poids importants
selon les **activations**. GGUF : format de llama.cpp pour CPU et local (`Q4_K_M`...). NF4 : 4 bits pour
**fine-tuning** (QLoRA).

**200. Quelle perte attendre ?**
8 bits : quasi nulle. 4 bits : modérée, acceptable pour beaucoup d'usages. En dessous : nette. **Mesurer sur
la tâche.**

**201. Comment déployer un modèle fine-tuné avec LoRA ?**
Option A : fusionner en bf16, éventuellement re-quantifier, servir un modèle unique. Option B : garder
l'adaptateur séparé sur un serveur multi-adaptateurs (un modèle de base, N adaptateurs).

**202. Quels serveurs d'inférence connais-tu ?**
vLLM (batching continu, PagedAttention, multi-LoRA), TGI, llama.cpp/Ollama (local), TensorRT-LLM (très
performant sur GPU NVIDIA).

**203. Comment dimensionnes-tu la mémoire GPU d'un service ?**
`poids + KV cache × requêtes simultanées + marge`. 7B bf16 (13,5 Go) + 32 requêtes de 4 096 tokens
(32 × 2 Gio) ≈ 80 Go ; avec GQA (cache ÷4) ≈ 30 Go.

**204. API ou modèle auto-hébergé : comment décider ?**
Coût auto-hébergé = GPU loué ÷ débit. Un petit modèle fine-tuné et quantifié peut coûter bien moins par
requête **si le volume amortit le GPU**. À faible volume, l'API est souvent moins chère et plus simple.

**205. Qu'est-ce que la fusion de modèles ?**
Combiner plusieurs modèles ou adaptateurs sans entraînement (moyenne de poids, SLERP, TIES, DARE). Empirique :
à évaluer.

**206. Que surveilles-tu en production ?**
Latence (TTFT, débit), erreurs, utilisation GPU et mémoire, coût par requête, qualité (échantillons relus,
retours utilisateurs), dérive des entrées.

**207. Comment mettre à jour un modèle sans interruption ?**
Déploiement progressif (canary / blue-green), évaluation avant bascule, retour arrière possible vers
l'ancienne version (adaptateur ou modèle fusionné versionné).

**208. Comment versionner un fine-tuning ?**
Modèle de base (identifiant exact), version du jeu de données, configuration LoRA, hyperparamètres,
graine, scores d'évaluation, adaptateur résultant.

**209. Qu'est-ce qu'un MoE ?**
Plusieurs FFN « experts » et un **routeur** qui en active quelques-uns par token. Paramètres totaux élevés,
**actifs** faibles (Mixtral 8x7B : ≈ 47 G au total, ≈ 13 G actifs). La mémoire reste celle de tous les
experts.

**210. Qu'est-ce qu'un modèle de raisonnement et ses conséquences d'ingénierie ?**
Entraîné (souvent par RL) à produire une longue chaîne de raisonnement avant la réponse. Les tokens de
raisonnement sont **facturés** et allongent la latence ; avec un plafond de tokens trop bas, la réponse peut
être vide.

---

## K. Conception de systèmes

Méthode : **clarifier** (volumes, données, contraintes) → **options** → **choix justifié par la mesure** →
**risques** → **comment mesurer**.

### Scénario 1 : « Notre assistant interne doit connaître nos documents. Fine-tuner un LLM ? »
- Clarifier : les documents changent-ils ? faut-il citer ? droits d'accès ?
- Recommandation : **RAG** (connaissances changeantes, citations, droits). Fine-tuning éventuel
  **ultérieur** pour le format et le comportement.
- Mesure : jeu d'évaluation de questions réelles ; comparer prompt, RAG, RAG + modèle fine-tuné.

### Scénario 2 : « Réduire le coût d'une tâche d'extraction JSON à fort volume »
- Baseline : gros modèle + prompt long (coût et latence mesurés).
- Option : **QLoRA sur un petit modèle** avec 500 à 5 000 exemples validés ; sortie contrainte et validation
  en code.
- Mesure : exactitude, JSON valide, coût par requête, latence. Conclure seulement si le gain de coût
  amortit données, entraînement et maintenance.

### Scénario 3 : « Un modèle par client (50 clients) »
- Un modèle de base partagé et **un adaptateur LoRA par client**, servi en multi-adaptateurs (vLLM).
- Isolation : l'adaptateur est choisi **côté serveur** selon l'identité authentifiée, jamais par le client.
- Risques : fuite entre clients (tests d'isolation), version de base commune à mettre à jour avec tous les
  adaptateurs.

### Scénario 4 : « Fine-tuner un 70B alors qu'on n'a qu'un GPU de 80 Go »
- Full fine-tuning impossible (≈ 1,1 To). **QLoRA** : poids NF4 ≈ 35 Go + adaptateurs et activations :
  possible, avec séquences et batch modestes et gradient checkpointing.
- Alternative : un modèle plus petit (8-14 G) fine-tuné sur le même jeu : souvent suffisant et bien moins
  cher à servir.

### Scénario 5 : « Le modèle fine-tuné est bon sur notre tâche mais a perdu en sécurité et en général »
- Diagnostic : évaluer refus et benchmarks généraux avant/après.
- Remèdes : rang et taux d'apprentissage plus faibles, moins d'époques, **mélanger** des données générales
  et de sécurité, ou fusionner partiellement l'adaptateur (échelle < 1).

### Scénario 6 : « Servir un LLM avec 1 000 requêtes simultanées »
- Goulot : KV cache. Leviers : GQA, quantification du modèle et du cache, PagedAttention, batching continu,
  plusieurs répliques, cache de préfixe.
- Mesure : TTFT p95, débit, coût par million de tokens.

---

## L. Questions pièges

**LoRA entraîne-t-il le modèle moins bien parce qu'il a moins de paramètres ?**
Pas nécessairement : pour du style, du format ou des consignes, la mise à jour utile est de rang faible. Il
peut rester derrière sur des tâches très exigeantes en nouvelles capacités. Se prouve par l'évaluation.

**Un rang 256 est-il toujours meilleur qu'un rang 8 ?**
Non : le gain stagne, le risque de sur-apprentissage augmente et la mémoire aussi. On monte seulement si
l'évaluation le justifie.

**Augmenter α revient-il à augmenter le rang ?**
Non. α règle l'**amplitude** de la mise à jour ; le rang règle la **capacité**.

**Pourquoi le fine-tuning de mon modèle hallucine-t-il plus ?**
Il a peut-être appris à répondre avec assurance sur des faits qu'il ne maîtrise pas, faute d'exemples de
refus. Ajouter des cas « je ne sais pas » et utiliser le RAG pour les faits.

**Pourquoi la perte d'entraînement est-elle presque nulle mais le modèle mauvais en production ?**
Sur-apprentissage : il a mémorisé. Données trop peu nombreuses ou trop d'époques ; il faut une validation
séparée et plus de diversité.

**Le KV cache change-t-il la sortie ?**
Non, seulement la vitesse (vérifié dans le lab : mêmes tokens avec et sans cache).

**Pourquoi un modèle 4 bits ne peut-il pas être entraîné normalement ?**
Les poids quantifiés ne sont pas différentiables comme des flottants. Avec QLoRA, on gèle les poids 4 bits
et on entraîne des adaptateurs 16 bits à côté.

**Un modèle plus gros est-il toujours meilleur ?**
Non : coût, latence, et un petit modèle bien adapté peut égaler un gros sur une tâche étroite. On décide
par la mesure.

**Le contexte long remplace-t-il le RAG ?**
Pas toujours : coût quadratique, information « perdue au milieu », pas de citation ni de droits. Voir la
formation RAG.

**LoRA ajoute-t-il de la latence ?**
Non si on fusionne ; légèrement sinon.

**Peut-on fine-tuner sur des données confidentielles sans risque ?**
Non : le modèle peut **mémoriser et restituer** des passages. Anonymiser, dédupliquer, limiter l'accès au
modèle et à l'adaptateur.

---

## M. Exercices de code

Les extraits ont tous été exécutés dans `labs/mini_transformer_lora.py`. **Apprends à les écrire sans les
relire.**

### Attention causale (avec KV cache)
Voir `CausalSelfAttention` dans le lab. Points qu'on te demandera d'expliquer : la forme des tenseurs
`(B, h, T, d_head)`, le masque `tril(diagonal=S - T)` quand il y a un cache, le `softmax` par ligne, la
concaténation des têtes puis `W_O`.

### Top-p (noyau)
```python
def top_p_filter(logits, p):
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = sorted_logits.softmax(-1)
    remove = probs.cumsum(-1) - probs > p      # on garde le token qui fait franchir p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
```

### LoRA from scratch
```python
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self):
        self.base.weight += self.scale * (self.B @ self.A)
```
Tests à écrire : sortie identique au départ ; seuls `A` et `B` ont `requires_grad` ; nombre de paramètres
`= r (d_in + d_out)` ; sortie identique avant et après `merge()`.

### Taille du KV cache
```python
def kv_cache_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_value=2):
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value

# Llama-2 7B : 524_288 octets par token ; 2 Gio pour 4096 tokens ; ÷4 avec 8 têtes KV (vérifié)
```

### Paramètres LoRA
```python
def lora_params(d_in, d_out, r):
    return r * (d_in + d_out)

# 7B, r=8 sur q et v : 2 * 32 * lora_params(4096, 4096, 8) == 4_194_304 (vérifié)
```

### Perte DPO
```python
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return -F.logsigmoid(beta * margin).mean()     # = ln 2 quand politique == référence
```

### BPE
Voir `train_bpe`, `bpe_encode`, `bpe_decode` dans le lab (module 2 de la formation).

### RoPE
Voir `rope` dans le lab ; vérifie que `rope(q, m) · rope(k, n)` ne dépend que de `m − n`.

### Questions de code fréquentes
- Écrire l'attention multi-têtes **sans regarder**.
- Implémenter LoRA et le **tester** (sortie initiale identique, fusion exacte).
- Calculer la mémoire d'inférence et d'entraînement d'un modèle donné.
- Écrire un `Dataset` qui **masque le prompt** dans les labels (`-100`).
- Lire une courbe de perte et diagnostiquer (sur-apprentissage, divergence, plateau).

---

## N. Les erreurs à éviter

- **Dire que LoRA « ajoute des connaissances ».** Il adapte un comportement ; les faits, c'est le RAG.
- **Promettre qu'un fine-tuning bat toujours le prompting.** Montre-le par une évaluation.
- **Confondre α et r.** α règle l'amplitude, r la capacité.
- **Oublier B = 0** quand on te demande l'initialisation, ou ne pas savoir pourquoi A n'est pas nul.
- **Citer des chiffres non vérifiés** comme des lois. Distingue « calculé » (formule) de « repère » de la
  pratique.
- **Ignorer la mémoire des activations** en parlant de LoRA : il n'économise que gradients et états
  d'optimiseur.
- **Fusionner dans un modèle 4 bits** sans mentionner la perte de précision.
- **Parler d'une expérience que tu n'as pas faite.** Si tu n'as pas fine-tuné sur GPU, dis que tu as
  implémenté et testé LoRA à la main et que tu connais le pipeline Hugging Face.
- **Négliger l'évaluation, les données et la sécurité** pour ne parler que d'architecture.
