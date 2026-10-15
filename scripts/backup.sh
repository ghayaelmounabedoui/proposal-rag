#!/bin/bash
# Sauvegarde chiffrée de data/raw (documents) et var/chroma (index) avec restic.
#
# Configuration : créer /root/.proposal-rag-backup.env (chmod 600) contenant par exemple :
#   export RESTIC_REPOSITORY="sftp:utilisateur@autre-serveur:/sauvegardes/proposal-rag"
#   export RESTIC_PASSWORD_FILE="/root/.restic-pass"
#   # Pour un stockage S3 / Backblaze, ajouter aussi :
#   # export AWS_ACCESS_KEY_ID="..."
#   # export AWS_SECRET_ACCESS_KEY="..."
#
# Planification : voir README, section Déploiement.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/proposal-rag}"
COMPOSE="docker compose -f ${APP_DIR}/docker-compose.prod.yml"

# shellcheck disable=SC1091
source /root/.proposal-rag-backup.env

cd "$APP_DIR"

# Le backend est arrêté quelques secondes pour ne pas copier la base SQLite
# pendant qu'elle est en cours d'écriture. Il redémarre quoi qu'il arrive.
trap '$COMPOSE start backend' EXIT
$COMPOSE stop backend

restic backup "$APP_DIR/data/raw" "$APP_DIR/var/chroma"

# Garder 7 sauvegardes quotidiennes, 4 hebdomadaires et 6 mensuelles.
restic forget --keep-daily 7 --keep-weekly 4 --keep-monthly 6 --prune
