"""
Module de vectorisation (embeddings) et de gestion de la base vectorielle ChromaDB.

Ce module :
1. Charge les chunks générés par data_processing/chunking.py
2. Génère un embedding (vecteur) pour chaque chunk avec un modèle
   multilingue (adapté au français)
3. Persiste tout dans une base ChromaDB sur disque
4. Fournit une fonction `search()` pour interroger la base et vérifier
   que le retriever fonctionne bien, avant même de brancher un LLM
"""

import json
import os

# Le modèle d'embeddings est déjà téléchargé et mis en cache localement
# après le premier lancement. On force le mode hors-ligne pour éviter que
# la librairie ne tente de vérifier une mise à jour à chaque exécution
# (ce qui provoque une erreur si le réseau est instable ou coupé).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # le modèle est déjà en cache localement
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # empêche aussi transformers de vérifier en ligne

import chromadb
from chromadb.utils import embedding_functions

# Modèle d'embeddings : léger, tourne sur CPU, support natif du français
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "force_n_knowledge_base"


def get_embedding_function():
    """
    Retourne la fonction d'embedding utilisée par ChromaDB.
    sentence-transformers télécharge le modèle une seule fois
    (~470 Mo) puis le réutilise depuis le cache local.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_chroma_client():
    """Retourne un client ChromaDB persistant (sauvegardé sur disque)."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def build_vectorstore(chunks_path: str = "data/chunks.json") -> None:
    """
    Construit (ou reconstruit) la base vectorielle à partir des chunks.
    Si la collection existe déjà, elle est supprimée puis recréée
    pour éviter les doublons lors de tests répétés.
    """
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    # On repart d'une collection propre à chaque construction complète
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # la collection n'existait pas encore, rien à faire

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    # ChromaDB attend des listes séparées : ids, textes, métadonnées
    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "title": chunk["title"],
            "category": chunk["category"],
            "url": chunk["url"],
            "last_modified": chunk["last_modified"],
        }
        for chunk in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"{len(chunks)} chunks vectorisés et indexés dans ChromaDB.")
    print(f"Base persistée dans : {CHROMA_DB_PATH}")


def search(query: str, n_results: int = 3) -> list[dict]:
    """
    Interroge la base vectorielle avec une question en langage naturel
    et retourne les chunks les plus pertinents (par similarité sémantique).
    """
    client = get_chroma_client()
    embedding_fn = get_embedding_function()
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    results = collection.query(query_texts=[query], n_results=n_results)

    formatted_results = []
    for i in range(len(results["documents"][0])):
        formatted_results.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return formatted_results


if __name__ == "__main__":
    # 1. Construction de la base à partir des chunks
    build_vectorstore()

    # 2. Test manuel : quelques questions pour vérifier la pertinence
    test_queries = [
        "Quelles sont les conditions d'admission ?",
        "Comment financer le programme ?",
        "Quels sont les temoignages d'anciens etudiants ?",
    ]

    for query in test_queries:
        print(f"\n=== Question : {query} ===")
        results = search(query, n_results=2)
        for r in results:
            print(f"  [{r['metadata']['category']}] {r['metadata']['title']}")
            print(f"  distance : {r['distance']:.4f}")
            print(f"  extrait  : {r['text'][:120]}...")