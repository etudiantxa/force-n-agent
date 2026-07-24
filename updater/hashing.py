"""
Module de hashing pour la détection des mises à jour du site FORCE-N.

Principe :
- Chaque page du corpus a un hash SHA-256 calculé à partir de son contenu.
- Ce hash est stocké dans un fichier JSON (hash_store.json), avec l'URL
  comme clé.
- Lors d'une vérification périodique, on refetch chaque page, on
  recalcule son hash, et on le compare à celui stocké : s'ils diffèrent,
  la page a été modifiée depuis la dernière vérification.
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HASH_STORE_PATH = "updater/hash_store.json"


def compute_hash(content: str) -> str:
    """
    Calcule le hash SHA-256 d'un contenu texte.
    On utilise le contenu nettoyé (pas le HTML brut) pour éviter que des
    changements insignifiants (espace, attribut HTML) ne déclenchent une
    fausse détection de changement.
    """
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_hash_store(path: str = HASH_STORE_PATH) -> dict:
    """
    Charge le fichier de hashes stockés.
    Retourne un dictionnaire vide si le fichier n'existe pas encore
    (premier lancement du projet).
    """
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_hash_store(hash_store: dict, path: str = HASH_STORE_PATH) -> None:
    """Sauvegarde le fichier de hashes sur disque."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hash_store, f, ensure_ascii=False, indent=2)


def initialize_hash_store(cleaned_corpus_path: str = "data/cleaned_corpus.json") -> dict:
    """
    Construit le hash_store initial à partir du corpus nettoyé fourni
    par l'encadrant. C'est l'état de référence : la "photo" du site au
    moment où le corpus a été extrait.

    Structure du hash_store :
    {
        "https://force-n.sn/...": {
            "hash": "abc123...",
            "title": "...",
            "category": "...",
            "last_checked": "2026-07-21T10:00:00"
        },
        ...
    }
    """
    with open(cleaned_corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    hash_store = {}
    now = datetime.now(timezone.utc).isoformat()

    for page in corpus:
        url = page["url"]
        hash_store[url] = {
            "hash": compute_hash(page["content"]),
            "title": page["title"],
            "category": page["category"],
            "last_checked": now,
        }

    save_hash_store(hash_store)
    print(f"Hash store initialisé avec {len(hash_store)} pages.")
    print(f"Sauvegardé dans : {HASH_STORE_PATH}")

    return hash_store


if __name__ == "__main__":
    initialize_hash_store()