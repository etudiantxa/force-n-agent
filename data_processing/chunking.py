"""
Module de chunking du corpus FORCE-N.

Découpe chaque page nettoyée en chunks (morceaux de texte) avec un
chevauchement, tout en conservant les métadonnées (source, catégorie,
date) sur chaque chunk. C'est ce chunking qui sera ensuite vectorisé
et indexé dans la base ChromaDB.
"""

import json
from pathlib import Path

# Taille cible d'un chunk, en nombre de caractères (approximation simple
# et suffisante pour un projet éducatif ; 1 token ≈ 4 caractères en français)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte en morceaux de taille ~chunk_size caractères,
    avec un chevauchement d'~overlap caractères entre chaque morceau.

    Le chevauchement évite qu'une information à cheval sur deux chunks
    ne soit "coupée" et perde son sens pour le retriever.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        if end >= len(text):
            break

        # On avance de (chunk_size - overlap) pour créer le chevauchement
        start += chunk_size - overlap

    return chunks


def chunk_corpus(cleaned_corpus: list[dict]) -> list[dict]:
    """
    Découpe chaque page du corpus nettoyé en chunks, en conservant
    les métadonnées sur chaque chunk individuel.

    Chaque chunk final a la forme :
    {
        "chunk_id": "identifiant unique",
        "text": "le texte du chunk",
        "title": "titre de la page d'origine",
        "category": "catégorie",
        "url": "url source",
        "last_modified": "date"
    }
    """
    all_chunks = []

    for page_idx, page in enumerate(cleaned_corpus):
        text_chunks = split_text_into_chunks(page["content"])

        for chunk_idx, chunk_text in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"page{page_idx}_chunk{chunk_idx}",
                "text": chunk_text,
                "title": page["title"],
                "category": page["category"],
                "url": page["url"],
                "last_modified": page["last_modified"],
            })

    print(f"Pages en entrée : {len(cleaned_corpus)}")
    print(f"Chunks générés : {len(all_chunks)}")

    return all_chunks


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Sauvegarde les chunks au format JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks sauvegardés dans : {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "data/cleaned_corpus.json"
    OUTPUT_PATH = "data/chunks.json"

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        cleaned_corpus = json.load(f)

    chunks = chunk_corpus(cleaned_corpus)
    save_chunks(chunks, OUTPUT_PATH)

    # Aperçu du résultat
    if chunks:
        print("\n--- Aperçu du premier chunk ---")
        print(f"chunk_id : {chunks[0]['chunk_id']}")
        print(f"titre    : {chunks[0]['title']}")
        print(f"texte    : {chunks[0]['text'][:150]}...")