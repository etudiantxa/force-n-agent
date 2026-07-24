"""
Module de mise à jour ciblée (targeted update) de la base ChromaDB.

Quand updater/monitor.py détecte qu'une page a changé, ce module :
1. Nettoie le nouveau contenu (même logique que data_processing/cleaning.py)
2. Le découpe en chunks (même logique que data_processing/chunking.py)
3. Supprime les anciens chunks de cette page dans ChromaDB
4. Ajoute les nouveaux chunks à leur place

Ça évite de devoir reconstruire toute la base vectorielle à chaque
mise à jour détectée : seule la page modifiée est retraitée.
"""

from data_processing.cleaning import clean_text
from data_processing.chunking import split_text_into_chunks
from rag_pipeline.vectorstore import get_chroma_client, get_embedding_function, COLLECTION_NAME
from updater.monitor import check_for_updates


def reindex_page(collection, page: dict) -> None:
    """
    Retraite une page modifiée et met à jour ses chunks dans ChromaDB.

    page doit contenir : url, title, category, new_content
    """
    url = page["url"]

    # 1. Nettoyage du nouveau contenu
    cleaned_content = clean_text(page["new_content"])

    # 2. Découpage en chunks
    text_chunks = split_text_into_chunks(cleaned_content)

    # 3. Suppression des anciens chunks de cette page
    #    (on utilise un filtre sur les métadonnées : tous les chunks dont
    #    l'URL correspond à cette page)
    collection.delete(where={"url": url})

    # 4. Ajout des nouveaux chunks
    new_ids = [f"{url}_chunk{i}" for i in range(len(text_chunks))]
    new_metadatas = [
        {
            "title": page["title"],
            "category": page["category"],
            "url": url,
            "last_modified": "mise_a_jour_automatique",
        }
        for _ in text_chunks
    ]

    collection.add(ids=new_ids, documents=text_chunks, metadatas=new_metadatas)

    print(f"  Réindexé : {page['title']} ({len(text_chunks)} chunk(s))")


def run_full_update_cycle() -> None:
    """
    Cycle complet de mise à jour :
    1. Détecte les pages modifiées (monitor.check_for_updates)
    2. Réindexe uniquement ces pages dans ChromaDB
    """
    changed_pages = check_for_updates()

    if not changed_pages:
        print("Aucune mise à jour à appliquer à la base vectorielle.")
        return

    client = get_chroma_client()
    embedding_fn = get_embedding_function()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"\nRéindexation de {len(changed_pages)} page(s) modifiée(s) :")
    for page in changed_pages:
        reindex_page(collection, page)

    print("\nMise à jour de la base vectorielle terminée.")


if __name__ == "__main__":
    run_full_update_cycle()