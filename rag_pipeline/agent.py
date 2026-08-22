"""
Module de l'agent conversationnel RAG.

Ce script connecte :
1. Le retriever ChromaDB (rag_pipeline/vectorstore.py) — pour récupérer
   les chunks pertinents par rapport à une question
2. Un LLM (Gemini, avec fallback automatique vers Grok) — pour générer
   une réponse en langage naturel à partir de ces chunks

Le prompt est conçu pour limiter les hallucinations : le LLM doit
s'appuyer uniquement sur le contexte fourni, citer sa source UNIQUEMENT
quand il répond réellement à partir du contexte, et dire clairement
qu'il ne sait pas si l'information n'est pas dans le contexte (sans
alors inventer de fausse source).

Ce module fournit aussi classify_email_intent(), qui remplace une
détection par mots-clés (trop fragile) par une classification légère
via le LLM lui-même : plus fiable pour distinguer une vraie demande
d'envoi d'e-mail d'une simple question contenant incidemment un mot
comme "candidater" ou "mail".
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.vectorstore import search

load_dotenv()  # charge les variables du fichier .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Le prompt système : c'est ici qu'on force le comportement anti-hallucination.
# Mis à jour pour inclure l'historique de conversation.
SYSTEM_PROMPT_WITH_HISTORY = """Tu es l'assistant conversationnel officiel du programme FORCE-N \
(Formations Ouvertes pour le Renforcement des Compétences, de l'Emploi et de \
l'Entrepreneuriat dans le Numérique), un programme de l'Université Numérique \
Cheikh Hamidou KANE financé par la Fondation Mastercard.

Règles strictes que tu dois toujours respecter :
1. Réponds UNIQUEMENT à partir des informations présentes dans le CONTEXTE ci-dessous OU DANS L'HISTORIQUE DE LA CONVERSATION.
2. Si le contexte ET l'historique ne contiennent pas assez d'informations pour répondre, dis \
clairement : "Je n'ai pas cette information dans ma base de connaissances actuelle. \
Je vous invite à consulter directement force-n.sn ou à contacter l'équipe FORCE-N." \
Dans ce cas précis, NE MENTIONNE AUCUNE SOURCE : n'ajoute pas de ligne "Source(s) :".
3. N'invente JAMAIS d'information (dates, chiffres, conditions) qui n'est pas \
explicitement dans le contexte ou dans l'historique.
4. UNIQUEMENT si tu as répondu à partir d'informations réelles du contexte ou de \
l'historique, cite systématiquement la ou les sources utilisées à la fin de ta \
réponse, sous la forme : "Source(s) : [titre de la page] (catégorie : ...)". \
Ne mets jamais de source placeholder ou vide.
5. Précise que tu es un assistant IA et non un représentant officiel humain de FORCE-N \
si l'utilisateur te pose une question qui semble l'exiger.
6. Réponds en français, de manière claire et concise.

HISTORIQUE DE LA CONVERSATION :
{history}

CONTEXTE :
{context}
"""

# Ancien prompt, sans historique, conservé pour les fonctions qui ne l'utilisent pas
SYSTEM_PROMPT_NO_HISTORY = """Tu es l'assistant conversationnel officiel du programme FORCE-N \
(Formations Ouvertes pour le Renforcement des Compétences, de l'Emploi et de \
l'Entrepreneuriat dans le Numérique), un programme de l'Université Numérique \
Cheikh Hamidou KANE financé par la Fondation Mastercard.

Règles strictes que tu dois toujours respecter :
1. Réponds UNIQUEMENT à partir des informations présentes dans le CONTEXTE ci-dessous.
2. Si le contexte ne contient pas assez d'informations pour répondre, dis \
clairement : "Je n'ai pas cette information dans ma base de connaissances actuelle. \
Je vous invite à consulter directement force-n.sn ou à contacter l'équipe FORCE-N." \
Dans ce cas précis, NE MENTIONNE AUCUNE SOURCE : n'ajoute pas de ligne "Source(s) :".
3. N'invente JAMAIS d'information (dates, chiffres, conditions) qui n'est pas \
explicitement dans le contexte.
4. UNIQUEMENT si tu as répondu à partir d'informations réelles du contexte, cite \
systématiquement la ou les sources utilisées à la fin de ta réponse, sous la forme : \
"Source(s) : [titre de la page] (catégorie : ...)". Ne mets jamais de source \
placeholder ou vide.
5. Précise que tu es un assistant IA et non un représentant officiel humain de FORCE-N \
si l'utilisateur te pose une question qui semble l'exiger.
6. Réponds en français, de manière claire et concise.

CONTEXTE :
{context}
"""

USER_PROMPT = "Question de l'utilisateur : {question}"

# Prompt de classification d'intention (utilisé par classify_email_intent,
# disponible si tu veux revenir à une détection par LLM plutôt que par mots-clés)
INTENT_SYSTEM_PROMPT = """Tu classifies l'intention d'un message envoyé à un \
assistant du programme FORCE-N. Réponds par UN SEUL MOT :

- "email" si l'utilisateur exprime clairement l'intention de COMPOSER ou \
ENVOYER un e-mail (candidature, demande d'information, contact) via l'agent.
- "question" pour tout le reste — y compris les questions qui parlent DE \
candidature, de contact, ou d'e-mail sans demander explicitement d'en envoyer \
un (ex : "comment candidater ?", "quel est l'e-mail de contact ?", "qui peut \
candidater ?" sont des questions, PAS des demandes d'envoi).

Réponds uniquement par "email" ou "question", sans ponctuation ni explication."""


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


def format_history(chat_history: list[tuple[str, str]]) -> str:
    """
    Convertit la liste d'historique [('role', 'message'), ...] en une chaîne
    lisible par le LLM.
    """
    formatted_history = ""
    for role, content in chat_history:
        if role == "user":
            formatted_history += f"Human: {content}\n"
        elif role == "assistant":
            formatted_history += f"Assistant: {content}\n"
    return formatted_history


def extract_text(content) -> str:
    """
    Normalise le contenu de la réponse en texte simple, que le LLM
    renvoie une chaîne (la plupart des modèles) ou une liste de blocs
    structurés (certains modèles Gemini 3).

    CETTE FONCTION MANQUAIT DANS LA VERSION PRÉCÉDENTE — c'est elle qui
    causait le crash "name 'extract_text' is not defined", puisqu'elle
    est appelée par plusieurs fonctions ci-dessous sans jamais avoir été
    redéfinie après l'ajout de la gestion de l'historique.
    """
    if isinstance(content, list):
        text_parts = [
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()
    return content


def get_conversation_chain_with_history():
    """
    Crée et configure le pipeline de conversation LangChain avec historique.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_WITH_HISTORY),
        ("user", USER_PROMPT),
    ])

    last_error = None
    for provider_name, get_llm_fn in LLM_PROVIDERS:
        try:
            llm = get_llm_fn()
            chain = prompt | llm
            print(f"[INFO] Utilisation du fournisseur LLM : {provider_name}")
            return chain
        except Exception as e:
            print(f"[Fallback] {provider_name} a échoué ({e}). Tentative avec le fournisseur suivant...")
            last_error = e
            continue

    raise RuntimeError(f"Tous les fournisseurs LLM ont échoué. Dernière erreur : {last_error}")


def get_conversation_chain_no_history():
    """
    Crée et configure le pipeline de conversation LangChain SANS historique.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_NO_HISTORY),
        ("user", USER_PROMPT),
    ])

    last_error = None
    for provider_name, get_llm_fn in LLM_PROVIDERS:
        try:
            llm = get_llm_fn()
            chain = prompt | llm
            print(f"[INFO] Utilisation du fournisseur LLM (sans hist): {provider_name}")
            return chain
        except Exception as e:
            print(f"[Fallback] {provider_name} a échoué ({e}). Tentative avec le fournisseur suivant...")
            last_error = e
            continue

    raise RuntimeError(f"Tous les fournisseurs LLM ont échoué. Dernière erreur : {last_error}")


async def classify_email_intent(message: str) -> bool:
    """
    Détermine si un message exprime une vraie intention d'envoyer un
    e-mail, via une classification légère par LLM plutôt qu'une simple
    recherche de mots-clés.

    En cas d'échec de l'appel LLM (API indisponible, etc.), on retombe
    prudemment sur False : mieux vaut traiter le message comme une
    question normale que de déclencher par erreur le workflow e-mail.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_PROMPT),
        ("user", "{message}"),
    ])

    for provider_name, get_llm_fn in LLM_PROVIDERS:
        try:
            llm = get_llm_fn()
            chain = prompt | llm
            response = await chain.ainvoke({"message": message})
            classification = extract_text(response.content).strip().lower()
            return "email" in classification
        except Exception as e:
            print(f"[Fallback] {provider_name} a échoué pour la classification d'intention ({e}).")
            continue

    return False


async def stream_agent_response_with_history(question: str, chat_history: list[tuple[str, str]], n_results: int = 3):
    """
    Version streaming du pipeline agent avec historique de conversation.
    """
    retrieved_chunks = await asyncio.to_thread(search, question, n_results)
    context = format_context(retrieved_chunks)
    history_str = format_history(chat_history)

    llm_chain = get_conversation_chain_with_history()
    inputs = {"question": question, "context": context, "history": history_str}

    async for chunk in llm_chain.astream(inputs):
        text = extract_text(chunk.content)
        if text:
            yield text


async def stream_agent_response(question: str, n_results: int = 3):
    """
    Version streaming du pipeline agent, SANS historique (conservée pour
    compatibilité / tests isolés).
    """
    retrieved_chunks = await asyncio.to_thread(search, question, n_results)
    context = format_context(retrieved_chunks)

    llm_chain = get_conversation_chain_no_history()

    async for chunk in llm_chain.astream({"context": context, "question": question}):
        text = extract_text(chunk.content)
        if text:
            yield text


async def ask_agent_with_history(question: str, chat_history: list[tuple[str, str]], n_results: int = 3) -> str:
    """
    Pipeline complet (non-streaming) : retrieval + génération, avec
    gestion de l'historique de conversation.
    """
    retrieved_chunks = await asyncio.to_thread(search, question, n_results)
    context = format_context(retrieved_chunks)
    history_str = format_history(chat_history)

    llm_chain = get_conversation_chain_with_history()
    inputs = {"question": question, "context": context, "history": history_str}

    response = await llm_chain.ainvoke(inputs)
    return extract_text(response.content)


async def ask_agent(question: str, n_results: int = 3) -> str:
    """
    Pipeline complet (non-streaming) SANS historique, avec fallback
    entre fournisseurs LLM.
    """
    retrieved_chunks = await asyncio.to_thread(search, question, n_results)
    context = format_context(retrieved_chunks)

    llm_chain = get_conversation_chain_no_history()
    response = await llm_chain.ainvoke({"context": context, "question": question})
    return extract_text(response.content)


if __name__ == "__main__":
    import asyncio as _asyncio

    async def _run_tests():
        test_questions = [
            "Quelles sont les conditions d'admission au programme FORCE-N ?",
            "Qui finance le programme FORCE-N ?",
            "Quel est le salaire moyen des alumni ?",  # question piège : pas dans le contexte
        ]
        for question in test_questions:
            print(f"\n=== Question : {question} ===")
            try:
                answer = await ask_agent(question)
                print(answer)
            except RuntimeError as e:
                print(f"Erreur : {e}")

    _asyncio.run(_run_tests())