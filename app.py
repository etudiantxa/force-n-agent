"""
Interface Chainlit de l'agent conversationnel FORCE-N.

Version 2 (complète) :
- Agent RAG conversationnel avec streaming (comme la version 1)
- Détection de l'intention d'envoyer un e-mail
- Collecte conversationnelle des informations (destinataire, objet, corps)
- Brouillon affiché avec boutons de confirmation/annulation
- Envoi SMTP réel uniquement après confirmation explicite de l'utilisateur
- Scheduler de surveillance des mises à jour démarré en arrière-plan
- Gestion de l'historique de conversation (ajoutée dans cette version)
"""

import os
import chainlit as cl
from rag_pipeline.agent import stream_agent_response_with_history
from rag_pipeline.vectorstore import get_embedding_function, get_chroma_client
from email_tool.tool import draft_email, send_confirmed_email
from updater.scheduler import start_scheduler

# Préchargement du modèle d'embeddings et de la connexion ChromaDB au
# démarrage de l'application (pas à la première question posée), pour
# que le premier échange avec un utilisateur soit rapide.
print("Préchargement du modèle d'embeddings...")
try:
    get_embedding_function()
    get_chroma_client()
    print("Préchargement terminé.")
except Exception as e:
    print(f"Erreur lors du préchargement : {e}")

# Démarrage du scheduler de surveillance des mises à jour, en tâche de
# fond, une seule fois au lancement de l'application.
try:
    start_scheduler()
except Exception as e:
    print(f"Erreur lors du démarrage du scheduler : {e}")

# Mots-clés pour détecter l'intention d'envoyer un e-mail.
# Approche équilibrée entre sensibilité et spécificité pour éviter :
# - les faux positifs (ex: "candidater" -> "candidat")
# - les dépendances supplémentaires (pas de LLM nécessaire pour cette détection)
#
# On cherche des expressions spécifiques qui indiquent clairement l'intention d'envoyer un email,
# plutôt que des sous-chaînes génériques. Cela réduit les faux positifs tout en restant simple.
# On normalise le texte pour gérer les variantes comme "e-mail", "email", "courriel".
EMAIL_KEYWORDS = [
    "envoyer un email",  # Couvre : envoyer un email, envoyer un e-mail
    "envoyer un courriel",
    "envoie un email",
    "envoie un courriel",
    "envoi email",       # Couvre : envoi email, envoi e-mail
    "envoi courriel",
    "ecrire à",          # Couvre : écrire à, écris à
    "contacter",         # Couvre : contacter, contacter l'équipe
    "postuler à",        # Couvre : postuler à, postuler pour
    # "candidature à",     # RETIRÉ : trop générique, causait des faux positifs comme "je veux envoyer un e-mail de candidature"
    "mail à",            # Couvre : mail à, email à, e-mail à, courriel à
    "courriel à",
]

def normalize_text_for_keywords(text: str) -> str:
    """Normalise le texte pour la détection de mots-clés : retire la ponctuation et remplace '-' par ' '."""
    import re
    # Remplacer les traits d'union par des espaces pour matcher "e-mail" comme "email"
    text = text.replace('-', ' ')
    # Retirer la ponctuation de base (optionnel, mais peut aider)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Mettre en minuscule
    return text.lower()

def is_email_request(text: str) -> bool:
    """Détecte si le message de l'utilisateur exprime l'intention d'envoyer un e-mail."""
    # Normaliser le texte pour la recherche de mots-clés
    normalized_text = normalize_text_for_keywords(text)
    # Vérifier si l'un des mots-clés spécifiques est dans le message normalisé
    for keyword in EMAIL_KEYWORDS:
        if keyword in normalized_text:
            return True
    return False  # Si aucun mot-clé n'est trouvé, ce n'est pas une demande d'email


async def collect_and_confirm_email():
    """
    Workflow conversationnel de collecte des informations d'e-mail,
    suivi d'un brouillon et d'une demande de confirmation explicite
    via boutons (pas juste une réponse textuelle "oui"/"non", plus
    fiable et sans ambiguïté d'interprétation).
    """
    to_response = await cl.AskUserMessage(
        content="À quelle adresse veux-tu envoyer cet e-mail ?", timeout=180
    ).send()
    if not to_response:
        await cl.Message(content="Envoi annulé (pas de réponse reçue à temps).").send()
        return
    to = to_response["output"]

    subject_response = await cl.AskUserMessage(
        content="Quel est l'objet de l'e-mail ?", timeout=180
    ).send()
    if not subject_response:
        await cl.Message(content="Envoi annulé (pas de réponse reçue à temps).").send()
        return
    subject = subject_response["output"]

    body_response = await cl.AskUserMessage(
        content="Que veux-tu écrire dans le message ?", timeout=300
    ).send()
    if not body_response:
        await cl.Message(content="Envoi annulé (pas de réponse reçue à temps).").send()
        return
    body = body_response["output"]

    draft_preview = draft_email.invoke({"to": to, "subject": subject, "body": body})

    actions = [
        cl.Action(
            name="confirm_send_email",
            payload={"to": to, "subject": subject, "body": body},
            label="✅ Confirmer l'envoi",
        ),
        cl.Action(
            name="cancel_send_email",
            payload={},
            label="❌ Annuler",
        ),
    ]

    await cl.Message(content=draft_preview, actions=actions).send()


@cl.action_callback("confirm_send_email")
async def on_confirm_send_email(action: cl.Action):
    """
    Appelée uniquement quand l'utilisateur clique sur "Confirmer l'envoi".
    C'est le seul point du programme où un e-mail est réellement envoyé.
    """
    result = send_confirmed_email(
        to=action.payload["to"],
        subject=action.payload["subject"],
        body=action.payload["body"],
    )
    await cl.Message(content=result["message"]).send()


@cl.action_callback("cancel_send_email")
async def on_cancel_send_email(action: cl.Action):
    """Appelée quand l'utilisateur clique sur "Annuler" : aucun envoi n'a lieu."""
    await cl.Message(content="Envoi annulé, aucun e-mail n'a été envoyé.").send()


@cl.on_chat_start
async def start():
    """Initialise la session utilisateur avec un historique vide et affiche le message de bienvenue."""
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content=(
            "👋 Bonjour ! Je suis l'assistant IA du programme **FORCE-N**.\n\n"
            "Je peux répondre à tes questions sur les formations, les conditions "
            "d'admission, les partenaires et les opportunités du programme. "
            "Je peux aussi t'aider à envoyer un e-mail de candidature ou de contact.\n\n"
            "⚠️ *Je suis un assistant IA et non un représentant officiel de FORCE-N. "
            "Pour toute démarche officielle, consulte force-n.sn.*"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Appelé à chaque message envoyé par l'utilisateur.
    Met à jour l'historique, route soit vers le workflow e-mail, soit vers l'agent RAG (streaming avec historique).
    """
    # Récupérer l'historique de la session
    chat_history = cl.user_session.get("chat_history", [])
    
    # Ajouter le message de l'utilisateur à l'historique
    chat_history.append(("user", message.content))

    if is_email_request(message.content):
        await collect_and_confirm_email()
        # Ne pas oublier de remettre à jour l'historique même après un workflow d'email
        cl.user_session.set("chat_history", chat_history)
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    # Appeler la nouvelle fonction qui gère l'historique
    async for token in stream_agent_response_with_history(message.content, chat_history):
        await response_msg.stream_token(token)

    # Ajouter la réponse de l'assistant à l'historique
    chat_history.append(("assistant", response_msg.content))
    cl.user_session.set("chat_history", chat_history)

    await response_msg.update()