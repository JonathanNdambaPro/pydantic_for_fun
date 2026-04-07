import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MESSAGES CONVERSATION — Maintenir le contexte entre plusieurs runs

L'usage principal de l'historique de messages est de maintenir
le contexte à travers plusieurs runs d'agent (conversation).

Pour réutiliser des messages, on les passe au paramètre
message_history de Agent.run / run_sync / run_stream.

Points clés :
- Si message_history est défini et non vide, le system prompt
  n'est PAS régénéré (on suppose qu'il est dans l'historique)
- new_messages() → messages du run courant uniquement
- all_messages() → historique complet + messages du run courant
- Le format est indépendant du modèle → on peut changer de modèle
  entre les runs
"""


# =====================================================================
# PARTIE 1 : Conversation basique avec message_history
# =====================================================================

# On passe les messages du premier run au second pour maintenir
# le contexte de la conversation.

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant. Réponds en français, de manière concise.',
)


def demo_conversation():
    """Conversation en 3 tours avec historique."""
    # Tour 1
    result1 = agent.run_sync("Je m'appelle Alice.")
    logger.info(f"Tour 1 : {result1.output}")

    # Tour 2 : on passe l'historique du tour 1
    result2 = agent.run_sync(
        "Comment je m'appelle ?",
        message_history=result1.new_messages(),
    )
    logger.info(f"Tour 2 : {result2.output}")

    # Tour 3 : on passe l'historique complet (tour 1 + 2)
    result3 = agent.run_sync(
        "Et toi, c'est quoi ton nom ?",
        message_history=result2.all_messages(),
    )
    logger.info(f"Tour 3 : {result3.output}")

    # Vérifier l'historique complet
    all_msgs = result3.all_messages()
    logger.info(f"Total messages après 3 tours : {len(all_msgs)}")


# =====================================================================
# PARTIE 2 : new_messages() vs all_messages()
# =====================================================================

# new_messages() → seulement les messages du run courant
# all_messages() → tout l'historique + messages courants
# La différence est importante quand on chaîne les runs.


def demo_new_vs_all():
    """Illustre la différence entre new_messages et all_messages."""
    result1 = agent.run_sync("Bonjour !")
    logger.info(f"Tour 1 - new_messages : {len(result1.new_messages())} messages")
    logger.info(f"Tour 1 - all_messages : {len(result1.all_messages())} messages")
    # new = all ici car pas d'historique passé

    result2 = agent.run_sync(
        "Ça va ?",
        message_history=result1.all_messages(),
    )
    logger.info(f"Tour 2 - new_messages : {len(result2.new_messages())} messages")
    logger.info(f"Tour 2 - all_messages : {len(result2.all_messages())} messages")
    # new = 2 (request + response du tour 2)
    # all = 4 (tour 1 + tour 2)


# =====================================================================
# PARTIE 3 : Conversation avec streaming
# =====================================================================

import asyncio


async def demo_streamed_conversation():
    """Conversation en streaming avec historique."""
    # Tour 1 (sync pour simplifier)
    result1 = agent.run_sync("Donne-moi 3 noms de fruits.")
    logger.info(f"Tour 1 : {result1.output}")

    # Tour 2 (streamé) avec historique
    async with agent.run_stream(
        "Maintenant, donne-moi des légumes.",
        message_history=result1.all_messages(),
    ) as result2:
        async for text in result2.stream_text():
            pass  # Consommer le stream

        logger.info(f"Tour 2 (streamé) : dernier texte reçu")
        logger.info(f"Total messages : {len(result2.all_messages())}")


# =====================================================================
# PARTIE 4 : Boucle de chat interactive
# =====================================================================

# Pattern classique : boucle qui accumule l'historique.


def demo_chat_loop():
    """Simule une boucle de chat avec 3 messages prédéfinis."""
    messages_to_send = [
        "Salut ! Parle-moi de Python.",
        "Quels sont ses avantages ?",
        "Merci, résume en une phrase.",
    ]

    history = []

    for user_msg in messages_to_send:
        logger.info(f"User : {user_msg}")
        result = agent.run_sync(user_msg, message_history=history)
        logger.success(f"Agent : {result.output}")

        # Accumuler l'historique
        history = result.all_messages()

    logger.info(f"Historique final : {len(history)} messages")


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Conversation basique ---
    logger.info("=== Conversation basique ===")
    demo_conversation()

    # --- Démo 2 : new_messages vs all_messages ---
    logger.info("=== new_messages vs all_messages ===")
    demo_new_vs_all()

    # --- Démo 3 : Conversation streamée ---
    logger.info("=== Conversation streamée ===")
    asyncio.run(demo_streamed_conversation())

    # --- Démo 4 : Boucle de chat ---
    logger.info("=== Boucle de chat ===")
    demo_chat_loop()
