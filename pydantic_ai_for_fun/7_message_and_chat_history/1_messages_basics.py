import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MESSAGES BASICS — Accéder aux messages échangés pendant un run

Après un run d'agent, on peut accéder aux messages échangés via
l'objet résultat (RunResult ou StreamedRunResult).

Méthodes disponibles :
- all_messages()      → tous les messages (y compris des runs précédents)
- new_messages()      → uniquement les messages du run courant
- all_messages_json() → variante JSON bytes
- new_messages_json() → variante JSON bytes

Structure des messages :
- ModelRequest  → requête envoyée au modèle (contient des parts)
  - UserPromptPart    → le prompt utilisateur
  - SystemPromptPart  → instructions système
  - ToolReturnPart    → résultat d'un outil
  - RetryPromptPart   → demande de retry
- ModelResponse → réponse du modèle
  - TextPart          → texte brut
  - ToolCallPart      → appel d'outil

Note StreamedRunResult :
- Les messages ne sont complets qu'APRÈS la fin du stream
- Il faut avoir appelé stream_output(), stream_text(), get_output(), etc.
- stream_text(delta=True) ne construit PAS le message final
"""


# =====================================================================
# PARTIE 1 : Accéder aux messages d'un RunResult (sync)
# =====================================================================

# Agent.run_sync retourne un RunResult avec all_messages() et new_messages().

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant. Réponds en français.',
)


def demo_run_result():
    """Montre comment accéder aux messages après un run sync."""
    result = agent.run_sync("Raconte-moi une blague.")

    # La sortie finale
    logger.info(f"Output : {result.output}")

    # Tous les messages du run
    all_msgs = result.all_messages()
    logger.info(f"Nombre de messages : {len(all_msgs)}")

    for i, msg in enumerate(all_msgs):
        logger.info(f"  Message {i}: {type(msg).__name__}")
        for part in msg.parts:
            logger.info(f"    Part: {type(part).__name__} → {str(part)[:100]}")

    # Juste les nouveaux messages (même chose ici car pas d'historique)
    new_msgs = result.new_messages()
    logger.info(f"Nouveaux messages : {len(new_msgs)}")

    # Usage (tokens consommés)
    logger.info(f"Usage : {result.usage()}")


# =====================================================================
# PARTIE 2 : Messages en JSON
# =====================================================================

# all_messages_json() et new_messages_json() retournent des bytes JSON.
# Utile pour sérialiser / stocker / transférer.


def demo_messages_json():
    """Montre la sérialisation JSON des messages."""
    result = agent.run_sync("Dis bonjour")

    # Variante JSON bytes
    json_bytes = result.all_messages_json()
    logger.info(f"JSON bytes (taille) : {len(json_bytes)} octets")
    logger.info(f"JSON aperçu : {json_bytes[:200]}...")

    new_json = result.new_messages_json()
    logger.info(f"New messages JSON (taille) : {len(new_json)} octets")


# =====================================================================
# PARTIE 3 : Messages d'un StreamedRunResult (async)
# =====================================================================

# Avec run_stream, les messages ne sont complets qu'après le stream.

import asyncio


async def demo_streamed_messages():
    """Montre l'accès aux messages pendant et après un stream."""
    async with agent.run_stream("Dis bonjour en 3 langues") as result:
        # Avant la fin du stream : messages incomplets
        msgs_before = result.all_messages()
        logger.info(f"Messages AVANT stream : {len(msgs_before)}")
        # → Seulement le ModelRequest, pas encore la réponse

        # Consommer le stream
        async for text in result.stream_text():
            pass  # On consomme tout

        # Après la fin du stream : messages complets
        msgs_after = result.all_messages()
        logger.info(f"Messages APRÈS stream : {len(msgs_after)}")
        # → ModelRequest + ModelResponse complet

        for i, msg in enumerate(msgs_after):
            logger.info(f"  Message {i}: {type(msg).__name__}")


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : RunResult messages ---
    logger.info("=== Messages d'un RunResult ===")
    demo_run_result()

    # --- Démo 2 : Messages en JSON ---
    logger.info("=== Messages JSON ===")
    demo_messages_json()

    # --- Démo 3 : StreamedRunResult messages ---
    logger.info("=== Messages d'un StreamedRunResult ===")
    asyncio.run(demo_streamed_messages())
