import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_core import to_jsonable_python

from pydantic_ai import Agent, ModelMessagesTypeAdapter

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MESSAGES SÉRIALISATION — Stocker et charger les messages en JSON

Pour persister l'historique de conversation (en base, sur disque,
entre Python et JS/TS), on utilise un TypeAdapter de Pydantic.

Pydantic AI exporte ModelMessagesTypeAdapter pour ça.

Workflow :
1. Obtenir les messages → result.all_messages()
2. Convertir en objets Python sérialisables → to_jsonable_python()
3. Stocker (JSON, base de données, fichier)
4. Recharger → ModelMessagesTypeAdapter.validate_python()
5. Réutiliser → agent.run_sync(..., message_history=loaded_messages)

Cas d'usage :
- Persistence de conversations
- Evals et tests
- Partage entre Python et JavaScript/TypeScript
- Backup / audit trail
"""


# =====================================================================
# PARTIE 1 : Sérialisation avec ModelMessagesTypeAdapter
# =====================================================================

# to_jsonable_python convertit les messages en objets Python standards.
# ModelMessagesTypeAdapter.validate_python les reconvertit en messages.

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant. Réponds en français.',
)


def demo_serialization():
    """Sérialiser / désérialiser des messages."""
    result = agent.run_sync("Quelle est la capitale de la France ?")
    logger.info(f"Output : {result.output}")

    # Étape 1 : Récupérer les messages
    history = result.all_messages()
    logger.info(f"Messages : {len(history)}")

    # Étape 2 : Convertir en objets Python sérialisables
    as_python = to_jsonable_python(history)
    logger.info(f"Type Python : {type(as_python)}")
    logger.info(f"Aperçu : {str(as_python)[:200]}...")

    # Étape 3 : Reconvertir en messages Pydantic AI
    restored = ModelMessagesTypeAdapter.validate_python(as_python)
    logger.info(f"Messages restaurés : {len(restored)}")

    # Étape 4 : Réutiliser dans un nouveau run
    result2 = agent.run_sync(
        "Et celle de l'Allemagne ?",
        message_history=restored,
    )
    logger.success(f"Suite de conversation : {result2.output}")


# =====================================================================
# PARTIE 2 : Stocker en fichier JSON
# =====================================================================

import json
import tempfile
from pathlib import Path


def demo_file_storage():
    """Sauvegarder et recharger un historique depuis un fichier JSON."""
    result = agent.run_sync("Donne-moi 3 faits sur Paris.")
    history = result.all_messages()

    # Sauvegarder en fichier
    as_python = to_jsonable_python(history)
    tmp_file = Path(tempfile.gettempdir()) / "pydantic_ai_history.json"
    tmp_file.write_text(json.dumps(as_python, indent=2, ensure_ascii=False))
    logger.info(f"Historique sauvegardé dans : {tmp_file}")

    # Recharger depuis le fichier
    loaded_data = json.loads(tmp_file.read_text())
    restored = ModelMessagesTypeAdapter.validate_python(loaded_data)
    logger.info(f"Messages rechargés : {len(restored)}")

    # Continuer la conversation
    result2 = agent.run_sync(
        "Donne-moi 3 faits de plus.",
        message_history=restored,
    )
    logger.success(f"Suite : {result2.output}")

    # Nettoyer
    tmp_file.unlink()


# =====================================================================
# PARTIE 3 : Réutiliser des messages avec un modèle différent
# =====================================================================

# Le format de messages est indépendant du modèle.
# On peut passer des messages d'un modèle à un autre.


def demo_cross_model():
    """Réutiliser des messages entre modèles différents."""
    # Run 1 avec Claude
    result1 = agent.run_sync("Raconte-moi une blague.")
    logger.info(f"Claude : {result1.output}")

    # Run 2 avec un autre modèle, même historique
    # (décommenté si vous avez accès à un autre modèle)
    # result2 = agent.run_sync(
    #     "Explique la blague",
    #     model='gateway/openai:gpt-4o',
    #     message_history=result1.new_messages(),
    # )
    # logger.info(f"GPT-4o : {result2.output}")

    # Alternative : même modèle mais ça montre le pattern
    result2 = agent.run_sync(
        "Explique la blague",
        message_history=result1.new_messages(),
    )
    logger.success(f"Explication : {result2.output}")

    # Vérifier les messages
    all_msgs = result2.all_messages()
    for i, msg in enumerate(all_msgs):
        model_name = getattr(msg, 'model_name', 'N/A')
        logger.info(f"  Message {i}: {type(msg).__name__} (model: {model_name})")


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Sérialisation basique ---
    logger.info("=== Sérialisation ModelMessagesTypeAdapter ===")
    demo_serialization()

    # --- Démo 2 : Stockage fichier ---
    logger.info("=== Stockage en fichier JSON ===")
    demo_file_storage()

    # --- Démo 3 : Cross-model ---
    logger.info("=== Messages cross-model ===")
    demo_cross_model()
