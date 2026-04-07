import logfire
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ImageUrl, DocumentUrl

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TOOL OUTPUT — Ce que les tools peuvent retourner

Les tools ne sont pas limités à renvoyer du texte brut.
Ils peuvent retourner tout ce que Pydantic sait sérialiser en JSON,
ainsi que du contenu multimodal (images, documents, audio, vidéo)
selon ce que le modèle supporte.

Types de retour possibles :
- str, int, float, bool        → sérialisés directement
- datetime, date                → sérialisés en ISO 8601
- BaseModel (Pydantic)          → sérialisé en JSON automatiquement
- ImageUrl / DocumentUrl        → contenu multimodal envoyé au modèle
- BinaryContent                 → contenu binaire (image, audio…)

Note sur la compatibilité :
- Certains modèles (ex: Gemini) supportent nativement les retours
  semi-structurés (dict, objets…).
- D'autres (ex: OpenAI) attendent du texte, mais savent très bien
  extraire le sens d'un JSON sérialisé automatiquement.
- Si un objet Python est retourné et que le modèle attend du texte,
  Pydantic AI le sérialise en JSON pour vous.
"""


# =====================================================================
# PARTIE 1 : Retour de types simples (datetime, BaseModel)
# =====================================================================

# Un tool peut retourner un datetime → le modèle reçoit la date en ISO 8601.
# Un tool peut retourner un BaseModel → le modèle reçoit le JSON correspondant.

class User(BaseModel):
    name: str
    age: int


agent_data = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant qui fournit des informations. "
        "Utilise les outils disponibles pour répondre aux questions."
    ),
)


@agent_data.tool_plain
def get_current_time() -> datetime:
    """Retourne l'heure actuelle."""
    now = datetime.now()
    logger.info(f"Heure actuelle → {now}")
    return now


@agent_data.tool_plain
def get_user() -> User:
    """Retourne les informations de l'utilisateur connecté."""
    user = User(name='Alice', age=28)
    logger.info(f"Utilisateur → {user}")
    return user


# =====================================================================
# PARTIE 2 : Retour de contenu multimodal (ImageUrl, DocumentUrl)
# =====================================================================

# Les tools peuvent renvoyer des URLs vers des images ou documents.
# Le modèle peut alors "voir" ou "lire" ce contenu pour formuler
# sa réponse — à condition qu'il supporte l'entrée multimodale.

agent_multimodal = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant visuel capable d'analyser des images "
        "et des documents. Décris ce que tu vois."
    ),
)


@agent_multimodal.tool_plain
def get_company_logo() -> ImageUrl:
    """Récupère le logo de l'entreprise."""
    logger.info("Récupération du logo")
    return ImageUrl(url='https://iili.io/3Hs4FMg.png')


@agent_multimodal.tool_plain
def get_document() -> DocumentUrl:
    """Récupère le document PDF de référence."""
    logger.info("Récupération du document PDF")
    return DocumentUrl(
        url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'
    )


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Retour datetime ---
    logger.info("=== Quelle heure est-il ? ===")
    result = agent_data.run_sync("Quelle heure est-il ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Retour BaseModel ---
    logger.info("=== Info utilisateur ===")
    result = agent_data.run_sync("Comment s'appelle l'utilisateur connecté ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Retour ImageUrl ---
    logger.info("=== Analyse du logo ===")
    result = agent_multimodal.run_sync("Quel est le nom de l'entreprise sur le logo ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Retour DocumentUrl ---
    logger.info("=== Contenu du document ===")
    result = agent_multimodal.run_sync("Quel est le contenu principal du document ?")
    logger.success(f"Réponse : {result.output}")
