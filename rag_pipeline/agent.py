"""
Module de l'agent conversationnel RAG.

Ce script connecte :
1. Le retriever ChromaDB (rag_pipeline/vectorstore.py) — pour récupérer
   les chunks pertinents par rapport à une question
2. Un LLM (Gemini, via l'API gratuite Google AI Studio) — pour générer
   une réponse en langage naturel à partir de ces chunks

Le prompt est conçu pour limiter les hallucinations : le LLM doit
s'appuyer uniquement sur le contexte fourni, citer sa source, et dire
clairement qu'il ne sait pas si l'information n'est pas dans le contexte.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.vectorstore import search

load_dotenv()  # charge les variables du fichier .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Le prompt système : c'est ici qu'on force le comportement anti-hallucination
SYSTEM_PROMPT = """Tu es l'assistant conversationnel officiel du programme FORCE-N \
(Formations Ouvertes pour le Renforcement des Compétences, de l'Emploi et de \
l'Entrepreneuriat dans le Numérique), un programme de l'Université Numérique \
Cheikh Hamidou KANE financé par la Fondation Mastercard.

Règles strictes que tu dois toujours respecter :
1. Réponds UNIQUEMENT à partir des informations présentes dans le CONTEXTE ci-dessous.
2. Si le contexte ne contient pas assez d'informations pour répondre, dis \
clairement : "Je n'ai pas cette information dans ma base de connaissances actuelle. \
Je vous invite à consulter directement force-n.sn ou à contacter l'équipe FORCE-N."
3. N'invente JAMAIS d'information (dates, chiffres, conditions) qui n'est pas \
explicitement dans le contexte.
4. À la fin de ta réponse, cite systématiquement la ou les sources utilisées, \
sous la forme : "Source(s) : [titre de la page] (catégorie : ...)".
5. Précise que tu es un assistant IA et non un représentant officiel humain de FORCE-N \
si l'utilisateur te pose une question qui semble l'exiger.
6. Réponds en français, de manière claire et concise.

CONTEXTE :
{context}
"""

USER_PROMPT = "Question de l'utilisateur : {question}"


def get_gemini_llm():
    """Retourne le client LLM Gemini configuré (fournisseur principal)."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY manquante. Vérifie que ton fichier .env contient "
            "bien la clé et qu'il est à la racine du projet."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,  # basse température = réponses plus factuelles, moins créatives
    )


def get_grok_llm():
    """
    Retourne le client LLM Grok (fournisseur de secours).
    L'API Grok est compatible OpenAI SDK : il suffit de changer le
    base_url et la clé API, aucune librairie spécifique nécessaire.
    """
    if not GROK_API_KEY:
        raise ValueError(
            "GROK_API_KEY manquante. Vérifie que ton fichier .env contient "
            "bien la clé et qu'il est à la racine du projet."
        )
    return ChatOpenAI(
        model="grok-4.3",
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
        temperature=0.2,
    )


# Ordre de préférence : Gemini en premier (tiers gratuit le plus généreux),
# Grok en secours si Gemini échoue (limite de taux atteinte, erreur API, etc.)
LLM_PROVIDERS = [
    ("Gemini", get_gemini_llm),
    ("Grok", get_grok_llm),
]


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Transforme les chunks récupérés par le retriever en un texte structuré
    que le LLM peut utiliser comme contexte, avec la source de chaque passage.
    """
    parts = []
    for chunk in retrieved_chunks:
        meta = chunk["metadata"]
        parts.append(
            f"[Source : {meta['title']} | Catégorie : {meta['category']} | "
            f"URL : {meta['url']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


async def stream_agent_response(question: str, n_results: int = 3):
    """
    Version streaming du pipeline agent, pour l'affichage progressif
    (token par token) dans Chainlit. Applique le même fallback entre
    fournisseurs que ask_agent() : si le streaming Gemini échoue avant
    même de commencer, on bascule sur Grok.
    """
    retrieved_chunks = search(question, n_results=n_results)
    context = format_context(retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    last_error = None

    for provider_name, get_llm_fn in LLM_PROVIDERS:
        try:
            llm = get_llm_fn()
            chain = prompt | llm
            async for chunk in chain.astream({"context": context, "question": question}):
                text = extract_text(chunk.content)
                if text:
                    yield text
            return  # streaming terminé avec succès, pas besoin d'essayer le fallback
        except Exception as e:
            print(f"[Fallback] {provider_name} a échoué en streaming ({e}). Tentative avec le fournisseur suivant...")
            last_error = e
            continue

    raise RuntimeError(f"Tous les fournisseurs LLM ont échoué. Dernière erreur : {last_error}")


def ask_agent(question: str, n_results: int = 3) -> str:
    """
    Pipeline complet : retrieval + génération, avec fallback entre
    fournisseurs LLM. Si Gemini échoue (limite de taux, erreur API,
    clé manquante...), l'agent bascule automatiquement sur Grok.
    """
    retrieved_chunks = search(question, n_results=n_results)
    context = format_context(retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])

    last_error = None

    for provider_name, get_llm_fn in LLM_PROVIDERS:
        try:
            llm = get_llm_fn()
            chain = prompt | llm
            response = chain.invoke({"context": context, "question": question})
            return extract_text(response.content)
        except Exception as e:
            print(f"[Fallback] {provider_name} a échoué ({e}). Tentative avec le fournisseur suivant...")
            last_error = e
            continue

    # Si tous les fournisseurs ont échoué
    raise RuntimeError(
        f"Tous les fournisseurs LLM ont échoué. Dernière erreur : {last_error}"
    )


def extract_text(content) -> str:
    """
    Normalise le contenu de la réponse en texte simple, que le LLM
    renvoie une chaîne (la plupart des modèles) ou une liste de blocs
    structurés (certains modèles Gemini 3).
    """
    if isinstance(content, list):
        text_parts = [
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return content


if __name__ == "__main__":
    test_questions = [
        "Quelles sont les conditions d'admission au programme FORCE-N ?",
        "Qui finance le programme FORCE-N ?",
        "Quel est le salaire moyen des alumni ?",  # question piège : pas dans le contexte
    ]

    for question in test_questions:
        print(f"\n=== Question : {question} ===")
        answer = ask_agent(question)
        print(answer)