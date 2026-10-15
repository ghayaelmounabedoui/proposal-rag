# Formation : du code au service en ligne

Ce document t'apprend le déploiement **de A à Z**, en prenant ce projet comme fil rouge.
Chaque module contient : l'idée, ce qu'on a fait dans le projet (avec les fichiers), et un
exercice gratuit à faire sur ton PC. Les résultats annoncés dans les exercices marqués
« vérifié » ont été observés sur ce projet ; les autres sont des explications ou des exemples
non testés ici.

> Conseil : ne lis pas tout d'un coup. Un module par session, avec son exercice.

## Sommaire

0. La vue d'ensemble
1. Les bases réseau (IP, DNS, ports, HTTPS)
2. Linux et SSH, le minimum vital
3. Docker : images et conteneurs
4. Docker Compose : plusieurs services ensemble
5. Le reverse proxy et le HTTPS (Caddy)
6. Configuration et secrets
7. Sécurité
8. Les données : volumes et sauvegardes
9. Observer : logs, santé, métriques
10. CI/CD : automatiser
11. Le déploiement sur un vrai serveur, pas à pas
12. Et en entreprise ? (Kubernetes, cloud, Terraform)
13. Diagnostiquer une panne
14. Les exercices (labs)
15. Préparer l'entretien
16. Glossaire

---

## 0. La vue d'ensemble

### Dev et production
- **Développement** : ça tourne sur ton PC, tu es le seul utilisateur, si ça plante tu relances.
- **Production** : ça tourne pour d'autres personnes, tout le temps. Il faut que ce soit
  **accessible, sécurisé, fiable, observable, sauvegardé et mettable à jour sans casse**.

### Les 8 questions du déploiement
Un déploiement, c'est répondre à ces 8 questions :

| # | Question | Dans ce projet |
|---|---|---|
| 1 | **Où** ça tourne ? | Un serveur Linux (VPS), ou un cloud |
| 2 | **Comment** on l'empaquette ? | Images Docker (`Dockerfile.backend`, `Dockerfile.frontend`) |
| 3 | **Comment** les services s'orchestrent ? | Docker Compose (`docker-compose.prod.yml`) |
| 4 | **Comment** l'utilisateur y accède ? | Domaine + DNS + Caddy (HTTPS) |
| 5 | **Comment** on le protège ? | Mot de passe Caddy, clé API, pare-feu, utilisateur non-root |
| 6 | **Comment** on le configure ? | Variables d'environnement (`.env`) |
| 7 | **Où** vivent les données ? | Volumes (`var/chroma`, `data/raw`) + sauvegardes |
| 8 | **Comment** on le surveille et on le met à jour ? | Healthchecks, logs, CI (`.github/workflows/ci.yml`) |

### Le schéma final
```
 Utilisateur
     │  https://rfp.mondomaine.com
     ▼
 [ DNS ] "rfp.mondomaine.com = 203.0.113.10"
     ▼
 ┌──────────────────────── SERVEUR (pare-feu : seuls 22, 80, 443 ouverts) ────────────────┐
 │                                                                                         │
 │   Caddy ──(réseau interne Docker, invisible d'Internet)──▶ Streamlit ──▶ API FastAPI    │
 │   HTTPS + mot de passe                                                    │             │
 │                                                                  ┌────────┴───────┐     │
 │                                                                  │ volumes disque │     │
 │                                                                  │ documents+index│     │
 │                                                                  └────────────────┘     │
 └───────────────────────────────────────────────────────────────────────┬─────────────────┘
                                                                          ▼
                                                              Groq (service LLM externe)
```

---

## 1. Les bases réseau

### 1.1 Adresse IP et DNS
- Une **adresse IP** identifie une machine sur le réseau (`203.0.113.10`).
- Un **nom de domaine** est un nom lisible. Le **DNS** est l'annuaire qui traduit nom en IP.
- Un enregistrement **A** dit « ce nom pointe vers cette IP ».
- Après une modification DNS, la propagation peut prendre de quelques minutes à quelques heures.

### 1.2 Ports
Une machine a 65 535 « portes » numérotées. Un programme écoute sur un port.
Les conventions : 22 = SSH, 80 = HTTP, 443 = HTTPS, 8000 et 8501 = nos services.

### 1.3 Les trois adresses d'écoute (très important)
| Adresse | Signification |
|---|---|
| `127.0.0.1` (localhost) | Cette machine uniquement |
| `0.0.0.0` | Toutes les interfaces : accessible de partout où le réseau le permet |
| une IP précise | Seulement cette interface |

Dans notre projet, deux lignes à comprendre :
- `docker-compose.yml` : `"127.0.0.1:8000:8000"` se lit `adresse_hôte:port_hôte:port_conteneur`.
  Le port 8000 de **ta machine** n'est accessible que **depuis ta machine**.
- `Dockerfile.backend` : `--host 0.0.0.0`. **À l'intérieur du conteneur**, le serveur doit écouter
  sur toutes les interfaces, sinon Docker ne peut pas lui transmettre le trafic. Ce n'est pas
  dangereux, parce que c'est Docker qui décide ce qui est publié à l'extérieur.

### 1.4 HTTP, HTTPS, TLS, certificats
- **HTTP** : le texte circule en clair (n'importe qui sur le chemin peut le lire).
- **HTTPS** = HTTP dans un tunnel chiffré **TLS**.
- Un **certificat** prouve que le serveur est bien celui du domaine. Il est signé par une
  **autorité de certification** (Let's Encrypt, gratuite).
- **Comment Let's Encrypt vérifie que le domaine est à toi** (défi ACME, simplifié) :
  il demande à ton serveur de publier un fichier secret à une adresse précise, en HTTP sur le
  **port 80**. S'il le trouve, il te délivre le certificat. **C'est pourquoi le port 80 doit
  être ouvert et que le DNS doit déjà pointer vers ton serveur.**
- Les certificats durent 90 jours. Caddy les renouvelle seul.

### Exercice 1 (gratuit)
```powershell
nslookup google.com                         # le DNS : nom -> IP

# Le certificat d'un site : qui l'a délivré, jusqu'à quand ? (vérifié)
$tcp = New-Object Net.Sockets.TcpClient("example.com",443)
$ssl = New-Object Net.Security.SslStream($tcp.GetStream())
$ssl.AuthenticateAsClient("example.com")
New-Object Security.Cryptography.X509Certificates.X509Certificate2($ssl.RemoteCertificate) |
    Select-Object Subject, Issuer, NotAfter | Format-List
$tcp.Close()

Get-NetTCPConnection -State Listen | Where-Object LocalPort -in 8000,8501 |
    Select LocalAddress, LocalPort          # qui écoute où ? (stack lancée)
```
Question : pourquoi `LocalAddress` vaut-il `127.0.0.1` pour 8000 et 8501 ?
(Réponse : à cause de `127.0.0.1:` dans le compose.)

---

## 2. Linux et SSH, le minimum vital

Presque tous les serveurs sont sous Linux. Tu n'as pas besoin d'être expert, mais il faut ces bases.

### 2.1 SSH
Connexion chiffrée à un serveur distant. On s'authentifie avec une **paire de clés** :
- la **clé privée** reste sur ton PC et ne se partage jamais ;
- la **clé publique** (`.pub`) se copie sur le serveur.

```powershell
ssh-keygen -t ed25519
ssh root@IP_DU_SERVEUR
```

### 2.2 Commandes à connaître
| Commande | Rôle |
|---|---|
| `ls -la`, `cd`, `pwd` | Lister, se déplacer, où suis-je |
| `cat`, `nano fichier` | Lire, éditer un fichier |
| `sudo` | Exécuter en administrateur |
| `chmod 600 fichier` | Seul le propriétaire lit et écrit (pour les secrets) |
| `chown -R 1000:1000 dossier` | Changer le propriétaire |
| `df -h`, `free -h` | Espace disque, mémoire |
| `ps aux`, `top` | Processus en cours |

### 2.3 Les droits de fichiers : le piège classique
Chaque processus tourne sous un **utilisateur** (un numéro, `uid`). Notre conteneur tourne
sous `appuser` (uid 1000). Si le dossier `var/chroma` du serveur appartient à `root`, le
conteneur **ne peut pas y écrire** : le backend plante. D'où `chown -R 1000:1000 var data`.
(Sur Docker Desktop Windows ce problème n'apparaît pas ; sur un serveur Linux, si.)

### Exercice 2 (gratuit)
```powershell
docker run -it --rm ubuntu bash
# dans le conteneur : ls /, whoami, id, df -h, exit
```
Tu viens d'entrer dans un mini-Linux jetable. Tape `exit` pour le détruire.

---

## 3. Docker : images et conteneurs

### 3.1 Les notions
- **Image** : un modèle figé (système + Python + ton code + dépendances). Comme un moule.
- **Conteneur** : une instance en marche d'une image. Comme un gâteau sorti du moule.
- **Registre** : un entrepôt d'images (Docker Hub, GitHub Container Registry).
- **Volume** : un dossier du disque hôte (ou géré par Docker) qui **survit** aux conteneurs.
- Un conteneur n'est pas une machine virtuelle : il partage le noyau de la machine, donc il est
  léger et démarre en secondes.

### 3.2 Lire `Dockerfile.backend` ligne par ligne
| Ligne | Pourquoi |
|---|---|
| `FROM python:3.11-slim` | Base minimale avec Python 3.11 |
| `ENV PYTHONUNBUFFERED=1` | Les logs sortent tout de suite (utile pour `docker logs`) |
| `RUN useradd ... appuser` | Créer un utilisateur normal |
| `RUN pip install torch ... /whl/cpu` | PyTorch **CPU** : évite ~4 Go de bibliothèques CUDA |
| `COPY requirements.txt .` puis `RUN pip install` | Les dépendances **avant** le code (voir cache) |
| `RUN python -c "...SentenceTransformer(...)"` | Télécharger les modèles **au build** : démarrage rapide, hors ligne possible |
| `COPY src/ ./src/` | Le code en dernier |
| `USER appuser` | Le processus ne tourne **pas** en administrateur |
| `HEALTHCHECK` | Docker teste régulièrement `/health` |
| `CMD [...]` | La commande lancée au démarrage |

### 3.3 Le cache de couches (une idée clé)
Chaque instruction crée une **couche**. Docker réutilise une couche si elle et toutes celles
au-dessus n'ont pas changé. Si on copiait `src/` *avant* d'installer les dépendances, chaque
modification d'une ligne de code réinstallerait tout (10 minutes). En copiant `requirements.txt`
d'abord, un changement de code ne reconstruit que les dernières couches (secondes).

### 3.4 Un défaut réel de notre image (à toi de le corriger)
```powershell
docker history proposal-rag-backend
```
Tu verras deux couches de ~585 Mo : la première télécharge les modèles, la seconde (`mkdir ... && chown -R
appuser /app`) les **recopie** pour changer leur propriétaire. Une couche ne peut qu'ajouter, jamais
retirer. Piste de correction : télécharger les modèles **après** `USER appuser`, dans un dossier déjà
possédé par `appuser`, pour qu'aucun `chown -R` ne soit nécessaire. Voir le lab 8.

### Exercice 3 (gratuit, vérifié)
```powershell
docker images                                   # tes images et leurs tailles
docker ps                                       # conteneurs en marche
docker logs --tail 20 proposal-rag-backend      # les logs
docker exec -it proposal-rag-backend bash       # entrer dans le conteneur (exit pour sortir)
docker exec proposal-rag-backend id             # uid=1000(appuser) : non-root
docker stats --no-stream                        # CPU et mémoire de chaque conteneur
```

---

## 4. Docker Compose : plusieurs services ensemble

Compose décrit plusieurs conteneurs dans un seul fichier YAML et les démarre ensemble.

### 4.1 Les notions
- **service** : un conteneur (backend, frontend, caddy).
- **réseau** : Compose crée un réseau privé. Les services s'y trouvent **par leur nom**.
  C'est pour cela que `PROPOSAL_RAG_API_URL=http://backend:8000` fonctionne.
  *Vérifié* : depuis le frontend, `backend` se résout en `172.20.0.2`.
- **ports** : publier un port du conteneur **vers l'hôte**. Sans `ports`, le service n'est
  joignable que par les autres services du réseau. En production, seul Caddy publie des ports.
- **volumes** : deux sortes. *Bind mount* (`./var/chroma:/app/var/chroma`, un dossier de ton
  disque) et *volume nommé* (`caddy_data`, géré par Docker).
- **depends_on + `condition: service_healthy`** : attendre qu'un service soit vraiment prêt,
  pas seulement démarré.
- **restart** : que faire si le conteneur s'arrête.

### 4.2 Les politiques de redémarrage (*vérifié* sur ce projet)
| Politique | Comportement |
|---|---|
| `no` | Jamais |
| `on-failure` | Seulement si le processus sort en erreur |
| `always` | Toujours |
| `unless-stopped` | Toujours, **sauf si tu l'as arrêté toi-même** |

Observations réelles :
- un processus qui plante (`exit 1`) est relancé en boucle (4 redémarrages en 12 s) ;
- `docker stop` : reste arrêté (comme voulu) ;
- **`docker kill` : reste aussi arrêté**, ce qui surprend : Docker le considère comme une
  action manuelle. Donc `docker kill` ne simule pas un crash.

### 4.3 Interpolation des variables
Dans le compose :
- `${GROQ_API_KEY:?message}` : **échoue** si la variable manque (on préfère planter tout de suite
  plutôt que de tourner mal configuré) ;
- `${LOG_LEVEL:-INFO}` : valeur par défaut `INFO` si absente.

Priorité : variable de l'environnement du shell > fichier `.env`.

### 4.4 Dev contre production
| | `docker-compose.yml` (dev) | `docker-compose.prod.yml` |
|---|---|---|
| Ports publiés | 8000 et 8501, sur `127.0.0.1` | **aucun** pour backend/frontend ; 80 et 443 pour Caddy |
| `API_KEY` | facultative | **obligatoire** (`:?`) |
| Reverse proxy | non | Caddy |
| `container_name` | fixé | absent (permet de lancer deux stacks) |

### Exercice 4 (gratuit, vérifié)
**Les données survivent, le conteneur non :**
```powershell
docker exec proposal-rag-backend sh -c "echo a > /tmp/ephemere.txt; echo a > /app/data/raw/persistant.txt"
docker compose up -d --force-recreate backend      # détruit et recrée le conteneur
docker exec proposal-rag-backend cat /tmp/ephemere.txt            # n'existe plus
docker exec proposal-rag-backend cat /app/data/raw/persistant.txt # existe encore (volume)
docker exec proposal-rag-backend rm /app/data/raw/persistant.txt  # nettoyage
```
Leçon : **tout ce qui est écrit hors volume disparaît** à la recréation du conteneur.

---

## 5. Le reverse proxy et le HTTPS (Caddy)

### 5.1 Pourquoi ne pas exposer directement Streamlit ?
- Streamlit ne gère pas le HTTPS proprement, ni l'authentification.
- Il faudrait ouvrir plusieurs ports, donc plusieurs failles potentielles.
- Un **reverse proxy** concentre tout à un seul endroit : un point d'entrée, un certificat,
  un mot de passe, des journaux d'accès.

### 5.2 Lire le `Caddyfile.example`
```
{$DOMAIN} {                      # le site, lu depuis la variable DOMAIN
    basic_auth {                 # demande identifiant + mot de passe
        admin <hash>             # le mot de passe est stocké haché (bcrypt), jamais en clair
    }
    reverse_proxy frontend:8501  # transmet tout à l'interface, sur le réseau interne
}
```
Ce que Caddy fait **sans que tu l'écrives** : obtenir et renouveler le certificat Let's Encrypt,
rediriger HTTP vers HTTPS (observé : code 308), gérer les WebSockets de Streamlit.

### 5.3 Pourquoi le volume `caddy_data` ?
Il conserve les certificats. Sans lui, chaque recréation redemande un certificat, et Let's Encrypt
limite le nombre de demandes (tu pourrais être bloqué quelques jours).

### 5.4 Pourquoi l'API n'est jamais routée
Le `Caddyfile` ne transmet que vers `frontend`. Le backend (`:8000`) n'est atteignable que par le
frontend, via le réseau interne, avec la clé API. *Vérifié* : `/docs` via Caddy renvoie la page
Streamlit, pas le Swagger de l'API.

### Exercice 5 (gratuit, vérifié) : la répétition de production en local
```powershell
docker run --rm caddy:2 caddy hash-password --plaintext 'essai-local-123'
copy Caddyfile.example Caddyfile ; notepad Caddyfile      # coller le hash
docker compose down
$env:DOMAIN="localhost"; $env:API_KEY="cle-de-test"
docker compose -f docker-compose.prod.yml up -d --build
```
Ouvre `https://localhost` (accepte l'avertissement : le certificat est local). Teste :
```powershell
curl.exe -sk -o NUL -w "%{http_code}" https://localhost/                          # 401
curl.exe -sk -o NUL -w "%{http_code}" -u admin:essai-local-123 https://localhost/ # 200
curl.exe -s  -o NUL -w "%{http_code} -> %{redirect_url}" http://localhost/        # 308 -> https
curl.exe -s -m 3 -o NUL -w "%{http_code}" http://127.0.0.1:8000/health            # 000 : fermé
```
Retour à la normale :
```powershell
docker compose -f docker-compose.prod.yml down
del Caddyfile
docker compose up -d
```

---

## 6. Configuration et secrets

### 6.1 Le principe (« 12-factor »)
Le **même code** tourne partout (dev, test, prod). Ce qui change, c'est la **configuration**,
fournie de l'extérieur par des variables d'environnement. Jamais de clé dans le code.

### 6.2 Dans le projet
- `.env.example` : le modèle, committé, sans valeurs secrètes.
- `.env` : tes vraies valeurs, **ignoré par git** (`.gitignore`) et **exclu des images**
  (`.dockerignore`).
- Le compose passe les variables aux conteneurs.

### 6.3 Si un secret fuite (réflexe à avoir)
1. **Révoque / change la clé tout de suite** chez le fournisseur. C'est l'étape qui compte.
2. Supprimer le fichier de git ne suffit pas : l'historique garde l'ancienne version.
3. Ensuite seulement, nettoie l'historique (outils : `git filter-repo`) et ajoute un scanner
   (`gitleaks`) à la CI.

### 6.4 En entreprise
Les secrets vivent dans un **gestionnaire de secrets** (HashiCorp Vault, Azure Key Vault, AWS
Secrets Manager), injectés au démarrage, avec droits d'accès, journal d'audit et rotation.

---

## 7. Sécurité

### 7.1 Réduire la surface d'attaque
Chaque chose exposée est une porte à surveiller. Dans le projet :
| Mesure | Fichier / endroit |
|---|---|
| Un seul point d'entrée exposé (Caddy) | `docker-compose.prod.yml` |
| Pare-feu : seuls 22, 80, 443 | `ufw` sur le serveur |
| Conteneurs non-root | `USER appuser` dans les Dockerfile |
| Ports de dev limités à `127.0.0.1` | `docker-compose.yml` |
| Image minimale (`-slim`) | moins de logiciels = moins de failles |

### 7.2 Un piège célèbre
**Docker contourne `ufw`.** Un port publié avec `ports:` est ouvert au monde même si `ufw` le
bloque, parce que Docker modifie directement les règles réseau. C'est pour cela qu'en production
backend et frontend n'ont aucun `ports`.

### 7.3 Deux niveaux d'authentification, à ne pas confondre
- **Le site** (mot de passe Caddy) : qui peut utiliser l'interface.
- **L'API** (`X-API-Key`) : qui peut appeler le backend directement.
La clé API ne protège pas le site (le frontend l'envoie lui-même). Il faut les deux.

### 7.4 Ce qu'on valide côté code (`api/main.py`)
Extensions autorisées, taille maximale de 20 Mo, nom de fichier nettoyé contre le *path
traversal* (`../../.env`), comparaison de la clé en temps constant (`secrets.compare_digest`),
erreurs du fournisseur LLM traduites en 502 sans fuite de détails internes.

### 7.5 Spécifique aux applications LLM
- **Les données quittent ton infrastructure** à chaque appel (ici Groq). Question RGPD à régler.
- **Prompt injection** : un document peut contenir « ignore tes instructions ». Nos prompts disent
  de traiter les documents comme des données, et la conformité vérifie les preuves en code
  (un `COMPLIANT` sans source valide est rétrogradé). Ce n'est pas une garantie absolue.

### 7.6 Hygiène continue
Mettre à jour les dépendances (Dependabot), scanner les images et paquets (`trivy`, `pip-audit`),
appliquer les mises à jour de sécurité du système (`unattended-upgrades`).

---

## 8. Les données : volumes et sauvegardes

### 8.1 Sans état (*stateless*) contre avec état (*stateful*)
Un conteneur sans état peut être détruit et recréé à volonté. Ici, l'état est dans deux endroits :
- `data/raw` : les **documents d'origine** (irremplaçables) ;
- `var/chroma` : l'**index** (reconstructible en réindexant les documents, mais long).
Tout le reste (code, images) se reconstruit depuis git.

### 8.2 La règle 3-2-1
**3** copies des données, sur **2** supports différents, dont **1** hors site. Une sauvegarde sur
le même serveur ne protège pas d'une perte du serveur.

### 8.3 Deux sigles à connaître
- **RPO** (*Recovery Point Objective*) : combien de données peut-on perdre ? Sauvegarde
  quotidienne = jusqu'à 24 h perdues.
- **RTO** (*Recovery Time Objective*) : combien de temps pour revenir en service ?

### 8.4 Pourquoi arrêter le backend pendant la sauvegarde
La base d'index est un fichier SQLite. Le copier pendant qu'elle est écrite peut donner un fichier
corrompu. `scripts/backup.sh` arrête brièvement le backend, sauvegarde, puis le relance (même en
cas d'erreur, grâce à `trap`).

### 8.5 La règle d'or
Une sauvegarde qui n'a **jamais été restaurée** n'est pas une sauvegarde. Teste la restauration.

### Exercice 6 (gratuit)
Simule sauvegarde et restauration sans outil spécial :
```powershell
docker compose stop backend
Compress-Archive -Path data\raw, var\chroma -DestinationPath $env:TEMP\sauvegarde.zip -Force
docker compose start backend
Expand-Archive $env:TEMP\sauvegarde.zip -DestinationPath $env:TEMP\restore-test -Force
dir $env:TEMP\restore-test\raw           # tes documents doivent y être
```
Question : que contient-on de plus avec restic ? (chiffrement, déduplication, historique,
envoi vers un autre serveur.)

---

## 9. Observer : logs, santé, métriques

### 9.1 Les trois piliers
- **Logs** : ce que l'application raconte (`docker logs`). Ici, format texte avec niveau et heure.
- **Métriques** : des nombres dans le temps (latence, erreurs, mémoire).
- **Traces** : le chemin d'une requête à travers les services (avancé).

### 9.2 Santé : vivant contre prêt
- *Liveness* : le processus tourne-t-il ? (redémarrer s'il est bloqué)
- *Readiness* : peut-il traiter des requêtes ? (ne lui envoyer du trafic que s'il l'est)
Notre `/health` répond « le processus est vivant ». Comme les modèles se chargent avant que l'API
écoute, il signifie aussi « prêt ». Pour aller plus loin on ajouterait un `/ready` qui teste aussi
la base et la clé Groq.

### 9.3 Ce qu'on surveille pour une application LLM
Latence par endpoint, taux d'erreurs 5xx, erreurs du fournisseur (502), **tokens consommés**
(coût), espace disque, mémoire (les modèles sont gourmands), fraîcheur de la dernière sauvegarde.

### 9.4 Alertes
Une métrique que personne ne regarde ne sert à rien : on définit des seuils (« erreurs > 5 % pendant
5 min ») qui préviennent par e-mail ou messagerie.

### Exercice 7 (gratuit, vérifié)
```powershell
docker compose ps                                           # colonne STATUS : (healthy)
docker inspect -f "{{json .State.Health}}" proposal-rag-backend   # l'historique des tests
docker logs -f proposal-rag-backend                         # suivre les logs en direct
```
Pendant que les logs défilent, pose une question dans l'interface et retrouve la trace.

---

## 10. CI/CD : automatiser

### 10.1 Les deux sigles
- **CI** (*Continuous Integration*) : à chaque modification, lancer automatiquement lint, tests,
  build. On détecte les régressions avant qu'elles arrivent en production.
- **CD** (*Continuous Delivery / Deployment*) : déployer automatiquement ce qui a passé la CI.

### 10.2 Notre `.github/workflows/ci.yml`
| Job | Ce qu'il fait |
|---|---|
| `test` | Installe les dépendances, lance `ruff` puis `pytest` (50 tests, sans réseau ni modèle) |
| `docker` | Construit les deux images (sans les publier), seulement si `test` est vert |

Il se déclenche à chaque `push` sur `main` et à chaque *pull request*.

### 10.3 Ce qui manque pour une vraie CD
1. **Étiqueter** les images (par exemple avec le hash du commit) et les **publier** dans un registre.
2. **Se connecter au serveur** et lancer la mise à jour. Exemple de principe (non testé ici) :
   ```yaml
   deploy:
     needs: docker
     if: github.ref == 'refs/heads/main'
     runs-on: ubuntu-latest
     steps:
       - uses: appleboy/ssh-action@v1
         with:
           host: ${{ secrets.SERVER_HOST }}
           username: deploy
           key: ${{ secrets.SERVER_SSH_KEY }}
           script: |
             cd /opt/proposal-rag && git pull
             docker compose -f docker-compose.prod.yml up -d --build
   ```
   Les identifiants sont dans les **secrets GitHub**, jamais dans le dépôt.

### 10.4 Environnements et retour en arrière
- **dev** → **staging** (copie de la prod pour tester) → **production**.
- **Rollback** : si la nouvelle version casse, on redéploie l'image précédente. D'où l'intérêt
  d'images **versionnées** (jamais seulement `latest`).

### 10.5 Stratégies de déploiement
| Stratégie | Idée |
|---|---|
| Remplacement simple | On arrête l'ancienne, on démarre la nouvelle (courte coupure). Notre cas. |
| Rolling | On remplace les instances une par une |
| Blue/green | Deux environnements : on bascule le trafic d'un coup |
| Canary | La nouvelle version reçoit 5 % du trafic d'abord |

---

## 11. Le déploiement sur un vrai serveur, pas à pas

Chaque étape avec **son pourquoi**.

| # | Action | Pourquoi |
|---|---|---|
| 1 | `ssh-keygen`, copier la clé publique chez l'hébergeur | S'authentifier sans mot de passe |
| 2 | Créer un serveur Ubuntu, au moins 4 Go de RAM | Les modèles ML consomment de la mémoire |
| 3 | Créer l'enregistrement DNS A vers l'IP, vérifier avec `nslookup` | Let's Encrypt doit trouver ton domaine |
| 4 | `ssh root@IP`, puis `apt update && apt upgrade -y` | Correctifs de sécurité |
| 5 | `ufw allow OpenSSH`, `80`, `443`, puis `ufw enable` | Fermer tout le reste (SSH **avant** d'activer) |
| 6 | `curl -fsSL https://get.docker.com \| sh` | Installer Docker |
| 7 | `git clone ... /opt/proposal-rag` | Récupérer le code |
| 8 | `mkdir -p var/chroma data/raw` puis `chown -R 1000:1000 var data` | Droits d'écriture du conteneur |
| 9 | Créer `.env` (`GROQ_API_KEY`, `API_KEY`, `DOMAIN`) et `chmod 600 .env` | Configuration et secrets |
| 10 | Générer le hash, `cp Caddyfile.example Caddyfile`, y coller le hash | Mot de passe devant le site |
| 11 | `docker compose -f docker-compose.prod.yml up -d --build` | Construire et démarrer |
| 12 | `logs -f caddy` : chercher « certificate obtained » | Vérifier le HTTPS |
| 13 | Ouvrir l'URL, importer des documents, poser une question | Test de bout en bout |
| 14 | Configurer `scripts/backup.sh` en tâche planifiée, **tester une restauration** | Ne pas perdre les données |

Mise à jour ensuite : `git pull` puis la commande de l'étape 11.

**Pour un test gratuit ou quasi** : un serveur facturé à l'heure détruit le soir et un sous-domaine
DuckDNS. Pense à **supprimer** le serveur (pas seulement l'éteindre), le sous-domaine, et à
révoquer les clés créées pour l'occasion.

---

## 12. Et en entreprise ?

### 12.1 La correspondance Compose → Kubernetes
Kubernetes (K8s) est un orchestrateur : il gère des conteneurs sur plusieurs machines.

| Dans notre compose | Dans Kubernetes |
|---|---|
| un `service` | un **Deployment** (N copies) + un **Service** (adresse interne stable) |
| `ports` / Caddy | un **Ingress** (+ **cert-manager** pour les certificats) |
| `.env` | **ConfigMap** (config) et **Secret** (secrets) |
| `volumes` | **PersistentVolumeClaim** |
| `healthcheck` | sondes **liveness** et **readiness** |
| `restart: unless-stopped` | le contrôleur recrée les pods tout seul |
| un seul serveur | plusieurs nœuds, mise à l'échelle automatique |

### 12.2 Services managés
Les entreprises préfèrent souvent ne pas gérer elles-mêmes les serveurs :
- conteneurs managés : Azure Container Apps, AWS ECS/Fargate, Google Cloud Run ;
- base de données managée avec sauvegardes automatiques ;
- registre d'images, gestionnaire de secrets, journalisation centralisée.
Les mêmes images Docker fonctionnent dans tous ces environnements.

### 12.3 Infrastructure as Code
Au lieu de cliquer dans une console, on **décrit l'infrastructure dans du code** (Terraform,
Pulumi, Bicep), versionné dans git, relu, rejoué à l'identique. Avantages : reproductibilité,
relecture, audit, destruction et recréation à la demande.

### 12.4 Ce qu'une entreprise ajoute
SSO avec rôles, WAF (pare-feu applicatif), réseau privé et segmentation, conformité (RGPD,
journaux d'audit), plan de reprise après sinistre, supervision 24/7, gestion des coûts.

### 12.5 Le raisonnement à garder
« Notre solution couvre les mêmes besoins avec des outils plus simples. En entreprise, je
remplacerais chaque brique par son équivalent managé, selon le cloud et les contraintes. »

---

## 13. Diagnostiquer une panne

### 13.1 La méthode : remonter la chaîne, couche par couche
```
 DNS → réseau/pare-feu → reverse proxy → application → dépendances → données
```
Pour chaque couche, une question et une commande :

| Couche | Question | Commande |
|---|---|---|
| DNS | Le nom pointe-t-il vers le bon serveur ? | `nslookup rfp.mondomaine.com` |
| Réseau | Le port est-il joignable ? | `curl -v http://IP` ; `ufw status` |
| Proxy | Caddy a-t-il un certificat, des erreurs ? | `docker compose -f docker-compose.prod.yml logs caddy` |
| Application | Les services sont-ils sains ? | `docker compose ps` ; `docker logs backend` |
| Dépendances | Groq répond-il ? Clé valide ? | recherche d'erreurs 401/429/502 dans les logs |
| Ressources | Disque ou mémoire pleins ? | `df -h` ; `free -h` ; `docker stats` |
| Données | Droits des volumes ? | `ls -ln var data` (uid 1000) |

### 13.2 Pannes fréquentes
| Symptôme | Cause probable |
|---|---|
| Site inaccessible | DNS non propagé, port 80/443 fermé (pare-feu local **et** de l'hébergeur) |
| Avertissement de certificat | Certificat pas encore émis (défi ACME échoué) |
| Backend redémarre en boucle | Droits sur `var`, mémoire insuffisante, clé manquante |
| 401 partout | `API_KEY` différente entre frontend et backend |
| 502 sur `/ask` | Fournisseur LLM indisponible, quota ou clé invalide |
| Conteneur arrêté après un `docker kill` | Normal : arrêt manuel, pas de redémarrage |
| Disque plein | Anciennes images : `docker system prune` (avec prudence) |

### 13.3 Réflexes
Lis le **premier** message d'erreur, pas le dernier. Reproduis. Change **une seule chose** à la fois.
Note ce qui a marché : c'est ton futur runbook.

---

## 14. Les exercices (labs)

Fais-les dans l'ordre. Les labs 1 à 7 sont gratuits et se font sur ton PC.

| Lab | Objectif | Module |
|---|---|---|
| 1 | DNS, ports, certificat observés avec `nslookup`, `curl`, `Get-NetTCPConnection` | 1 |
| 2 | Entrer dans un conteneur, lire les logs, voir `id` non-root | 2, 3 |
| 3 | Prouver que `backend` se résout en IP depuis le frontend :<br>`docker compose exec -T frontend python -c "import socket; print(socket.gethostbyname('backend'))"` | 4 |
| 4 | Données persistantes contre éphémères (voir exercice 4) | 4, 8 |
| 5 | Répétition de production avec Caddy en local (voir exercice 5) | 5 |
| 6 | Sauvegarde et restauration manuelles (voir exercice 6) | 8 |
| 7 | Pannes provoquées (voir ci-dessous) | 4, 9, 13 |
| 8 | Optimiser l'image : supprimer la couche dupliquée de 585 Mo | 3 |
| 9 | Casser un test et regarder la CI devenir rouge sur GitHub | 10 |
| 10 | (Optionnel) Un vrai déploiement sur un VPS facturé à l'heure, puis destruction | 11 |

### Lab 7 : provoquer des pannes pour les comprendre
```powershell
# a) Un conteneur qui plante est relancé (vérifié : plusieurs redémarrages en quelques secondes)
docker run -d --name crashy --restart unless-stopped alpine sh -c "sleep 2; exit 1"
Start-Sleep 12 ; docker inspect -f "redemarrages={{.RestartCount}}" crashy ; docker rm -f crashy

# b) Un arrêt manuel n'est PAS relancé (vérifié)
docker run -d --name calme --restart unless-stopped alpine sleep 600
docker stop calme ; docker inspect -f "etat={{.State.Status}}" calme ; docker rm -f calme

# c) Mauvaise clé API (si API_KEY est définie) : 401 attendu
curl.exe -s -o NUL -w "%{http_code}" -H "X-API-Key: faux" http://127.0.0.1:8000/documents
```
Pour chaque panne, écris : ce que tu as observé, ce que tu as cru qu'il se passerait, et l'écart.

### Lab 8 : optimiser l'image
Objectif : retrouver environ 585 Mo. Piste : dans `Dockerfile.backend`, déplacer `USER appuser`
avant le `RUN python -c "...modèles..."` avec `HF_HOME` dans un dossier créé et possédé par
`appuser`, puis retirer le `chown -R /app` ou le limiter au strict nécessaire. Reconstruis, compare
`docker images` et relance `pytest` et le test de bout en bout.

---

## 15. Préparer l'entretien

### Le récit en six phrases
1. J'empaquette backend et frontend dans des images Docker : même comportement partout.
2. Un reverse proxy est le seul composant exposé : HTTPS automatique et authentification.
3. Les secrets passent par l'environnement, jamais par git ; l'API est protégée par une clé.
4. Des healthchecks conditionnent les démarrages et le redémarrage automatique.
5. Les données sont sur des volumes sauvegardés hors du serveur, restauration testée.
6. Une CI exécute lint, tests et build avant toute mise en production.

### Ce que tu peux dire honnêtement
« J'ai conçu et validé une configuration de production, que j'ai testée en local de bout en bout
(HTTPS, authentification, ports internes fermés, healthchecks). Je sais comment la déployer sur un
serveur et ce que je changerais en entreprise. » Ne prétends jamais avoir mis en production ce que
tu n'as pas déployé en vrai.

### Questions et pistes de réponse
| Question | Idée de réponse |
|---|---|
| Pourquoi Docker ? | Reproductibilité, isolation, déploiement identique partout |
| Pourquoi un reverse proxy ? | Un seul point d'entrée : HTTPS, authentification, journaux ; il cache les services internes |
| Comment gères-tu les secrets ? | Variables d'environnement, `.env` ignoré par git ; en entreprise, un gestionnaire de secrets |
| Et si le serveur tombe ? | Redémarrage automatique pour les pannes de processus, sauvegardes hors site ; pour la haute dispo, plusieurs instances derrière un load balancer |
| Comment éviter de casser la prod ? | CI obligatoire, staging, images versionnées, rollback par redéploiement |
| Comment passer à l'échelle ? | Connaître les limites : index local SQLite, appels LLM synchrones. Base vectorielle managée, file de tâches, plusieurs instances |
| Où vont les données ? | Chez le fournisseur LLM ; régler le RGPD ou changer de fournisseur (interface `LLM` prévue) |
| Qu'est-ce qui t'a surpris ? | Docker contourne `ufw` ; `docker kill` ne relance pas un conteneur `unless-stopped` ; une couche Docker ne peut que s'ajouter |

---

## 16. Glossaire

| Terme | Sens |
|---|---|
| **A (enregistrement)** | Ligne DNS qui relie un nom à une IP |
| **ACME** | Protocole d'obtention automatique de certificats (Let's Encrypt) |
| **Bind mount** | Dossier de l'hôte monté dans un conteneur |
| **CI / CD** | Intégration continue / livraison (ou déploiement) continu |
| **Conteneur** | Instance en marche d'une image |
| **DNS** | Annuaire nom → IP |
| **Healthcheck** | Test périodique qui dit si un service est sain |
| **Image** | Modèle figé d'un conteneur |
| **IaC** | Infrastructure décrite dans du code (Terraform) |
| **Ingress** | Entrée HTTP(S) d'un cluster Kubernetes |
| **Reverse proxy** | Intermédiaire placé devant une application |
| **Registre** | Entrepôt d'images (Docker Hub, GHCR, ECR) |
| **Rollback** | Retour à la version précédente |
| **RPO / RTO** | Perte de données tolérée / temps de remise en service |
| **SSH** | Connexion chiffrée à un serveur |
| **Staging** | Environnement de pré-production |
| **TLS / HTTPS** | Chiffrement des échanges web |
| **VPS** | Serveur virtuel loué |
| **Volume** | Stockage qui survit au conteneur |
| **WAF** | Pare-feu applicatif web |
