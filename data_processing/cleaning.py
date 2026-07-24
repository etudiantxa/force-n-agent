"""
Module de nettoyage du corpus FORCE-N.

Ce script :
1. Charge le corpus JSON brut
2. Retire les balises HTML résiduelles et décode les entités HTML
3. Normalise les espaces et les caractères
4. Supprime les doublons (contenu identique)
5. Sauvegarde un corpus nettoyé, prêt pour le chunking
"""

import json
import re
import html
from pathlib import Path
from bs4 import BeautifulSoup


def clean_text(raw_text: str) -> str:
    """
    Nettoie un texte brut potentiellement issu de scraping :
    - retire les balises HTML
    - décode les entités HTML (&eacute; -> é)
    - normalise les espaces multiples
    """
    # 1. Décoder les entités HTML (&eacute; -> é, &amp; -> &, etc.)
    text = html.unescape(raw_text)

    # 2. Retirer les balises HTML résiduelles (<p>, <br>, etc.)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 3. Normaliser les espaces multiples, tabulations, retours à la ligne
    text = re.sub(r"\s+", " ", text)

    # 4. Retirer les espaces en début/fin
    text = text.strip()

    return text


def load_corpus(path: str) -> list[dict]:
    """Charge le corpus JSON brut depuis un fichier."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_corpus(raw_corpus: list[dict]) -> list[dict]:
    """
    Applique le nettoyage à chaque page du corpus et supprime les doublons
    (basé sur le contenu nettoyé, pas le titre — deux pages peuvent avoir
    des titres différents mais un contenu copié-collé identique).
    """
    cleaned = []
    seen_contents = set()
    duplicates_count = 0

    for page in raw_corpus:
        cleaned_content = clean_text(page.get("content", ""))
        cleaned_title = clean_text(page.get("title", ""))

        # Détection de doublon : on ignore la casse et les espaces
        content_key = cleaned_content.lower().strip()

        if not cleaned_content:
            # On ignore les pages vides après nettoyage
            continue

        if content_key in seen_contents:
            duplicates_count += 1
            continue

        seen_contents.add(content_key)

        cleaned.append({
            "title": cleaned_title,
            "content": cleaned_content,
            "category": page.get("category", "inconnu"),
            "url": page.get("url", ""),
            "last_modified": page.get("last_modified", ""),
        })

    print(f"Pages en entrée : {len(raw_corpus)}")
    print(f"Doublons supprimés : {duplicates_count}")
    print(f"Pages nettoyées en sortie : {len(cleaned)}")

    return cleaned


def save_cleaned_corpus(cleaned_corpus: list[dict], output_path: str) -> None:
    """Sauvegarde le corpus nettoyé au format JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_corpus, f, ensure_ascii=False, indent=2)
    print(f"Corpus nettoyé sauvegardé dans : {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "data/sample_corpus.json"
    OUTPUT_PATH = "data/cleaned_corpus.json"

    raw = load_corpus(INPUT_PATH)
    cleaned = clean_corpus(raw)
    save_cleaned_corpus(cleaned, OUTPUT_PATH)

    # Aperçu du résultat sur la première page
    if cleaned:
        print("\n--- Aperçu de la première page nettoyée ---")
        print(f"Titre   : {cleaned[0]['title']}")
        print(f"Contenu : {cleaned[0]['content'][:200]}...")