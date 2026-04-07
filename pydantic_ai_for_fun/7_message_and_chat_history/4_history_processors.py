import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext, TextPart, UserPromptPart

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HISTORY PROCESSORS — Modifier l'historique avant envoi au modèle

Les history processors interceptent et modifient l'historique de
messages AVANT chaque requête au modèle. Paramètre history_processors
sur Agent, ou via la capability HistoryProcessor.

Points clés :
- Les processors REMPLACENT l'historique (pas de copie auto)
- Appliqués dans l'ordre de la liste
- Sync ou async acceptés
- Peuvent accepter un RunContext optionnel
- Attention : les tool calls/returns doivent rester appairés
- new_messages() peut être affecté si on réordonne/supprime

Cas d'usage :
- Filtrer des infos sensibles (privacy)
- Limiter le nombre de messages (économie tokens)
- Résumer les vieux messages (préserver contexte)
- Logique de traitement custom
"""


# =====================================================================
# PARTIE 1 : Filtre simple — Garder seulement les ModelRequest
# =====================================================================

# On supprime toutes les ModelResponse pour ne garder que les requêtes.
# Le modèle ne voit que les questions, pas ses propres réponses.


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Supprime toutes les réponses du modèle de l'historique."""
    filtered = [msg for msg in messages if isinstance(msg, ModelRequest)]
    logger.info(f"[filter] {len(messages)} → {len(filtered)} messages (responses supprimées)")
    return filtered


agent_filtered = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    history_processors=[filter_responses],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Garder les N derniers messages
# =====================================================================

# Utile pour les longues conversations : on garde seulement
# les messages récents pour économiser des tokens.


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Garde seulement les 5 derniers messages."""
    max_messages = 5
    if len(messages) > max_messages:
        logger.warning(f"[recent] Troncature : {len(messages)} → {max_messages} messages")
        return messages[-max_messages:]
    return messages


agent_recent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    history_processors=[keep_recent_messages],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 3 : Processor avec RunContext
# =====================================================================

# Un processor peut recevoir un RunContext pour accéder aux infos
# du run courant (usage tokens, dépendances, etc.).


def context_aware_processor(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Tronque l'historique si trop de tokens consommés."""
    current_tokens = ctx.usage.total_tokens
    logger.info(f"[context] Tokens courants : {current_tokens}")

    if current_tokens > 1000:
        logger.warning("[context] Tokens élevés → on garde les 3 derniers messages")
        return messages[-3:]
    return messages


agent_context = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    history_processors=[context_aware_processor],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 4 : Processors multiples (chaînés)
# =====================================================================

# Plusieurs processors s'appliquent dans l'ordre de la liste.
# Le résultat du premier est passé au second.


def log_message_count(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Logge le nombre de messages (sans modifier)."""
    logger.info(f"[log] {len(messages)} messages dans l'historique")
    return messages


def keep_last_three(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Garde les 3 derniers messages."""
    return messages[-3:] if len(messages) > 3 else messages


agent_chained = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    history_processors=[log_message_count, keep_last_three],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 5 : Résumer les vieux messages avec un LLM
# =====================================================================

# On utilise un agent moins cher pour résumer les anciens messages.
# Attention : les tool calls/returns doivent rester appairés.

summarize_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="""Résume cette conversation en français.
Omets les bavardages, concentre-toi sur le contenu technique.""",
)


async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Résume les 10 plus anciens messages si l'historique est long."""
    threshold = 10
    if len(messages) > threshold:
        oldest = messages[:threshold]
        logger.info(f"[summarize] Résumé des {threshold} premiers messages")
        summary = await summarize_agent.run(message_history=oldest)
        # On retourne le résumé + les messages récents
        return summary.new_messages() + messages[-1:]
    return messages


agent_summarize = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    history_processors=[summarize_old_messages],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 6 : Créer manuellement des messages pour les tests
# =====================================================================

# Les messages sont de simples dataclasses, on peut les créer
# manuellement pour les tests ou la simulation.


def demo_manual_messages():
    """Crée un historique de messages à la main."""
    fake_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Quelle est la capitale du Japon ?")]),
        ModelResponse(parts=[TextPart(content="La capitale du Japon est Tokyo.")]),
        ModelRequest(parts=[UserPromptPart(content="Et sa population ?")]),
        ModelResponse(parts=[TextPart(content="Tokyo a environ 14 millions d'habitants.")]),
    ]

    # Utiliser cet historique fabriqué dans un run
    agent = Agent(
        'gateway/anthropic:claude-sonnet-4-6',
        instructions="Tu es un assistant. Réponds en français.",
    )
    result = agent.run_sync(
        "Résume ce qu'on a dit.",
        message_history=fake_history,
    )
    logger.success(f"Résumé : {result.output}")


# =====================================================================
# PARTIE 7 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Filtre responses ---
    logger.info("=== Filter responses ===")
    history = [
        ModelRequest(parts=[UserPromptPart(content="Bonjour")]),
        ModelResponse(parts=[TextPart(content="Salut !")]),
    ]
    result = agent_filtered.run_sync("Comment ça va ?", message_history=history)
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Keep recent ---
    logger.info("=== Keep recent messages ===")
    result1 = agent_recent.run_sync("Message 1")
    result2 = agent_recent.run_sync("Message 2", message_history=result1.all_messages())
    result3 = agent_recent.run_sync("Message 3", message_history=result2.all_messages())
    logger.success(f"Réponse : {result3.output}")
    logger.info(f"Messages finaux : {len(result3.all_messages())}")

    # --- Démo 3 : Processors chaînés ---
    logger.info("=== Processors chaînés ===")
    result = agent_chained.run_sync("Bonjour !", message_history=history)
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Messages manuels ---
    logger.info("=== Messages manuels ===")
    demo_manual_messages()
