"""
Module d'envoi d'e-mails via SMTP.

Utilise smtplib (bibliothèque standard Python, aucune dépendance externe)
pour envoyer des e-mails via un serveur SMTP configurable (Gmail par défaut).
"""

import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


def send_email(to: str, subject: str, body: str) -> dict:
    """
    Envoie un e-mail via le serveur SMTP configuré.

    Retourne un dictionnaire {"success": bool, "message": str} plutôt que
    de lever une exception directement : ça permet à l'agent conversationnel
    d'informer l'utilisateur proprement en cas d'échec, sans planter.
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        return {
            "success": False,
            "message": "Configuration SMTP manquante. Vérifie SMTP_USER et "
                        "SMTP_PASSWORD dans ton fichier .env.",
        }

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()  # chiffre la connexion avant d'envoyer les identifiants
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        return {
            "success": True,
            "message": f"E-mail envoyé avec succès à {to}.",
        }

    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "Échec de l'authentification SMTP. Vérifie que tu "
                        "utilises bien un mot de passe d'application Gmail, "
                        "pas ton mot de passe habituel.",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Erreur lors de l'envoi de l'e-mail : {e}",
        }


if __name__ == "__main__":
    # Test manuel : envoie un e-mail de test à toi-même
    # (remplace par ta propre adresse pour vérifier la réception)
    test_recipient = SMTP_USER  # envoi à toi-même par défaut, le plus sûr pour tester

    result = send_email(
        to=test_recipient,
        subject="Test - Agent FORCE-N",
        body="Ceci est un e-mail de test envoyé depuis le module email_tool. "
             "Si tu reçois ce message, la configuration SMTP fonctionne correctement.",
    )

    print(result["message"])