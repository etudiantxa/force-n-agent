"""
Outil LangChain pour l'envoi d'e-mails.

Ce tool est conçu pour être utilisé en deux temps depuis Chainlit :
1. L'agent (ou l'utilisateur) prépare un brouillon avec draft_email()
2. Une fois que l'utilisateur a cliqué sur "Confirmer l'envoi" dans
   l'interface, send_confirmed_email() est appelée pour envoyer réellement.

Cette séparation garantit qu'aucun e-mail n'est jamais envoyé sans une
action de confirmation explicite de l'utilisateur (exigence éthique du sujet).
"""

from langchain_core.tools import tool
from email_tool.sender import send_email


@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """
    Prépare un brouillon d'e-mail à partir des informations collectées
    en conversation (destinataire, objet, corps du message).

    Utilise cet outil quand l'utilisateur souhaite envoyer un e-mail
    (candidature, demande d'information, contact avec l'équipe FORCE-N).
    Ne PAS envoyer l'e-mail directement : ce tool retourne uniquement un
    aperçu du brouillon, qui devra être validé par l'utilisateur avant
    tout envoi réel.
    """
    draft = (
        f"**Brouillon d'e-mail préparé :**\n\n"
        f"**À :** {to}\n"
        f"**Objet :** {subject}\n\n"
        f"{body}\n\n"
        f"Merci de confirmer si tu souhaites envoyer cet e-mail tel quel, "
        f"ou si tu veux le modifier."
    )
    return draft


def send_confirmed_email(to: str, subject: str, body: str) -> dict:
    """
    Envoie réellement l'e-mail. Cette fonction ne doit être appelée
    qu'APRÈS confirmation explicite de l'utilisateur (ex : clic sur un
    bouton "Confirmer l'envoi" dans Chainlit) — jamais directement par
    l'agent de sa propre initiative.
    """
    return send_email(to=to, subject=subject, body=body)


if __name__ == "__main__":
    # Test manuel du brouillon (sans envoi réel)
    preview = draft_email.invoke({
        "to": "contact@force-n.sn",
        "subject": "Demande d'information - Certificat en IA",
        "body": "Bonjour,\n\nJe souhaiterais obtenir plus d'informations sur "
                "les conditions d'admission au certificat en Intelligence "
                "Artificielle et LLM.\n\nCordialement.",
    })
    print(preview)