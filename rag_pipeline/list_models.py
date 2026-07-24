"""
Utilitaire : liste les modèles Gemini actuellement disponibles pour ta clé API.

À lancer une seule fois pour savoir quel nom de modèle utiliser dans
rag_pipeline/agent.py. Les noms de modèles évoluent régulièrement côté
Google, donc mieux vaut vérifier directement plutôt que deviner.
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Modèles disponibles pour ta clé API, supportant generateContent :\n")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f"  - {model.name}")