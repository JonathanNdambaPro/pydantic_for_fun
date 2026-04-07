import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import Hooks

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS EVENT STREAM — Intercepter le flux d'événements en streaming

Les event stream hooks permettent d'intercepter les événements
pendant un run streamé :

1. run_event_stream → wrappe le flux complet (async generator)
2. event            → hook de commodité, se déclenche pour chaque événement

Cas d'usage :
- Compteur d'événements / métriques de streaming
- Logging détaillé du flux
- Transformation d'événements à la volée
- Monitoring temps réel du streaming
"""


# =====================================================================
# PARTIE 1 : Hook event — Compteur d'événements
# =====================================================================

# Le hook 'event' se déclenche pour chaque événement individuel
# pendant un run streamé. Simple et pratique.

hooks_event = Hooks()
event_count = 0


@hooks_event.on.event
async def count_events(
    ctx: RunContext[None], event: AgentStreamEvent
) -> AgentStreamEvent:
    """Compte chaque événement du stream."""
    global event_count
    event_count += 1
    return event


agent_event = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_event],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Hook event — Logging détaillé
# =====================================================================

# On peut loguer le type et le contenu de chaque événement.

hooks_detailed = Hooks()
event_types: list[str] = []


@hooks_detailed.on.event
async def log_event_details(
    ctx: RunContext[None], event: AgentStreamEvent
) -> AgentStreamEvent:
    """Logge le type de chaque événement."""
    event_type = type(event).__name__
    event_types.append(event_type)
    logger.info(f"[event] Type: {event_type}")
    return event


agent_detailed = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_detailed],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 3 : Combiner event hooks avec d'autres hooks
# =====================================================================

# On peut combiner les event hooks avec des hooks model_request
# pour avoir une vue complète du lifecycle.

hooks_combo = Hooks()
lifecycle_log: list[str] = []


@hooks_combo.on.before_model_request
async def log_model_start(ctx, request_context):
    lifecycle_log.append("model:start")
    return request_context


@hooks_combo.on.after_model_request
async def log_model_end(ctx, *, request_context, response):
    lifecycle_log.append("model:end")
    return response


@hooks_combo.on.event
async def log_stream_event(
    ctx: RunContext[None], event: AgentStreamEvent
) -> AgentStreamEvent:
    lifecycle_log.append(f"event:{type(event).__name__}")
    return event


agent_combo = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_combo],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # --- Démo 1 : Compteur d'événements ---
        logger.info("=== Event hook (compteur) ===")
        global event_count
        event_count = 0
        async with agent_event.run_stream("Raconte une blague courte") as stream:
            result = await stream.get_output()
        logger.success(f"Réponse : {result}")
        logger.info(f"Nombre d'événements : {event_count}")

        # --- Démo 2 : Logging détaillé des types ---
        logger.info("=== Event hook (types détaillés) ===")
        event_types.clear()
        async with agent_detailed.run_stream("Dis bonjour") as stream:
            result = await stream.get_output()
        logger.success(f"Réponse : {result}")
        logger.info(f"Types d'événements : {event_types}")

        # --- Démo 3 : Combo lifecycle + events ---
        logger.info("=== Combo model hooks + event hooks ===")
        lifecycle_log.clear()
        async with agent_combo.run_stream("Salut !") as stream:
            result = await stream.get_output()
        logger.success(f"Réponse : {result}")
        logger.info(f"Lifecycle : {lifecycle_log}")

    asyncio.run(main())
