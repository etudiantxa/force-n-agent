"""
Interface Chainlit de l'agent conversationnel FORCE-N.

- Agent RAG conversationnel avec streaming
- Classification d'intention par LLM (pas de mots-clés) pour distinguer
  une vraie demande d'envoi d'e-mail d'une simple question
- Collecte conversationnelle des informations (destinataire, objet, corps)
- Brouillon affiché avec boutons de confirmation/annulation
- Envoi SMTP réel uniquement après confirmation explicite de l'utilisateur
- Scheduler de surveillance des mises à jour démarré en arrière-plan
"""

import chainlit as cl
from rag_pipeline.agent import stream_agent_response, classify_email_intent
from rag_pipeline.vectorstore import get_embedding_function, get_chroma_client
from email_tool.tool import draft_email, send_confirmed_email
from updater.scheduler import start_scheduler

# Préchargement du modèle d'embeddings et de la connexion ChromaDB au
# démarrage de l'application (pas à la première question posée), pour
# que le premier échange avec un utilisateur soit rapide.
print("Préchargement du modèle d'embeddings...")
get_embedding_function()
get_chroma_client()
print("Préchargement terminé.")

# Démarrage du scheduler de surveillance des mises à jour, en tâche de
# fond, une seule fois au lancement de l'application.
start_scheduler()


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
    to = to_response["output"].strip()

    subject_response = await cl.AskUserMessage(
        content="Quel est l'objet de l'e-mail ?", timeout=180
    ).send()
    if not subject_response:
        await cl.Message(content="Envoi annulé (pas de réponse reçue à temps).").send()
        return
    subject = subject_response["output"].strip()

    body_response = await cl.AskUserMessage(
        content="Que veux-tu écrire dans le message ?", timeout=300
    ).send()
    if not body_response:
        await cl.Message(content="Envoi annulé (pas de réponse reçue à temps).").send()
        return
    body = body_response["output"].strip()

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
    """Message de bienvenue affiché à l'ouverture d'une nouvelle conversation."""
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
    Route soit vers le workflow e-mail, soit vers l'agent RAG (streaming),
    selon une classification d'intention par LLM (pas de mots-clés).
    """
    wants_email = await classify_email_intent(message.content)

    if wants_email:
        await collect_and_confirm_email()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    async for token in stream_agent_response(message.content):
        await response_msg.stream_token(token)

    await response_msg.update()