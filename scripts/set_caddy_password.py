"""Définit le mot de passe de l'utilisateur 'admin' dans le Caddyfile.

Le mot de passe est saisi sans écho, haché par Caddy (bcrypt), puis le hash est écrit
directement dans le fichier : pas de copier-coller, donc pas d'erreur de recopie.

Usage (depuis la racine du projet) :
    python scripts/set_caddy_password.py
"""

import getpass
import re
import subprocess
import sys
from pathlib import Path

CADDYFILE = Path("Caddyfile")
# Ligne "<utilisateur> <hash>" dans le bloc basic_auth.
LINE = re.compile(r"^(\s*admin\s+)(\S+)", re.MULTILINE)


def main() -> int:
    if not CADDYFILE.exists():
        print("Caddyfile introuvable : copiez d'abord Caddyfile.example en Caddyfile.")
        return 1

    password = getpass.getpass("Nouveau mot de passe (rien ne s'affiche) : ")
    if len(password) < 8:
        print("Mot de passe trop court (8 caractères minimum).")
        return 1
    if getpass.getpass("Confirmez : ") != password:
        print("Les deux saisies sont différentes.")
        return 1

    result = subprocess.run(
        ["docker", "run", "--rm", "caddy:2", "caddy", "hash-password", "--plaintext", password],
        capture_output=True,
        text=True,
        check=False,
    )
    hashed = result.stdout.strip()
    if result.returncode != 0 or len(hashed) != 60 or not hashed.startswith("$2"):
        print("Échec de la génération du hash :", result.stderr.strip() or hashed)
        return 1

    content = CADDYFILE.read_text(encoding="utf-8")
    if not LINE.search(content):
        print("Ligne 'admin <hash>' introuvable dans le Caddyfile.")
        return 1

    # Fonction de remplacement : évite que les '$' du hash soient lus comme des références.
    updated = LINE.sub(lambda m: m.group(1) + hashed, content, count=1)
    CADDYFILE.write_text(updated, encoding="utf-8", newline="")
    print("Mot de passe mis à jour (hash de 60 caractères).")
    print("Relancez Caddy : docker compose -f docker-compose.prod.yml up -d --force-recreate caddy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
