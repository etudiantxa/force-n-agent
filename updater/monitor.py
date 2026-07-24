"""
Module de surveillance des mises à jour du site FORCE-N.

Ce script :
1. Vérifie le fichier robots.txt du site avant tout accès
2. Refetch chaque page connue (avec un délai entre les requêtes)
3. Recalcule le hash du nouveau contenu et le compare à l'ancien
4. Retourne la liste des pages modifiées, avec leur nouveau contenu
5. Journalise chaque vérification (fichier de log)
"""

import time
import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path

from updater.hashing import compute_hash, load_hash_store, save_hash_store

# Délai (en secondes) entre deux requêtes, pour ne pas surcharger le serveur
DELAY_BETWEEN_REQUESTS = 2

LOG_PATH = "updater/update_log.txt"

# User-Agent identifiant clairement notre agent (bonne pratique de scraping responsable)
HEADERS = {
    "User-Agent": "FORCE-N-Agent-Educatif/1.0 (projet etudiant, usage non commercial)"
}


def is_allowed_by_robots(url: str) -> bool:
    """
    Vérifie que l'URL peut être accédée selon le robots.txt du site.
    Contrainte obligatoire du sujet : respecter le robots.txt.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        # Si le robots.txt est inaccessible, on adopte une approche prudente :
        # on autorise l'accès mais on log l'incident pour le documenter.
        log_event(f"Impossible de lire robots.txt pour {robots_url}, accès autorisé par défaut.")
        return True


def fetch_page_text(url: str) -> str:
    """
    Récupère le contenu texte d'une page (retire les balises HTML),
    de la même manière que le nettoyage initial, pour que les hash
    soient comparables.
    """
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text(separator=" ").strip()


def log_event(message: str) -> None:
    """Ajoute une ligne horodatée au journal des vérifications."""
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def check_for_updates() -> list[dict]:
    """
    Parcourt toutes les pages du hash_store, refetch chacune, compare
    son nouveau hash à l'ancien, et retourne la liste des pages modifiées.

    Chaque page modifiée retournée a la forme :
    {
        "url": "...",
        "title": "...",
        "category": "...",
        "new_content": "...",
        "old_hash": "...",
        "new_hash": "..."
    }
    """
    hash_store = load_hash_store()
    changed_pages = []

    log_event(f"Début de la vérification ({len(hash_store)} pages à contrôler)")

    for url, stored_info in hash_store.items():
        if not is_allowed_by_robots(url):
            log_event(f"Accès refusé par robots.txt : {url}")
            continue

        try:
            new_content = fetch_page_text(url)
            new_hash = compute_hash(new_content)

            if new_hash != stored_info["hash"]:
                changed_pages.append({
                    "url": url,
                    "title": stored_info["title"],
                    "category": stored_info["category"],
                    "new_content": new_content,
                    "old_hash": stored_info["hash"],
                    "new_hash": new_hash,
                })
                log_event(f"CHANGEMENT DÉTECTÉ : {url}")

                # On met à jour le hash_store tout de suite avec le nouveau hash
                hash_store[url]["hash"] = new_hash
                hash_store[url]["last_checked"] = datetime.now(timezone.utc).isoformat()
            else:
                hash_store[url]["last_checked"] = datetime.now(timezone.utc).isoformat()

        except requests.RequestException as e:
            log_event(f"ERREUR lors du fetch de {url} : {e}")

        # Délai obligatoire entre deux requêtes (contrainte du sujet)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    save_hash_store(hash_store)
    log_event(f"Vérification terminée. {len(changed_pages)} page(s) modifiée(s).")

    return changed_pages


if __name__ == "__main__":
    changes = check_for_updates()

    if changes:
        print(f"\n{len(changes)} page(s) modifiée(s) détectée(s) :")
        for page in changes:
            print(f"  - {page['title']} ({page['url']})")
    else:
        print("\nAucune modification détectée depuis la dernière vérification.")