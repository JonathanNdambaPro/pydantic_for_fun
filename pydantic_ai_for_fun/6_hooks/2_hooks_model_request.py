import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS MODEL REQUEST — Intercepter les appels au modèle LLM

Les model request hooks se déclenchent autour de chaque appel au LLM.
ModelRequestContext regroupe : model, messages, model_settings,
et model_request_parameters.

Trois niveaux d'interception :
- before_model_request  → avant l'appel (modifier la requête)
- after_model_request   → après l'appel (inspecter/modifier la réponse)
- model_request (wrap)  → entourer l'appel complet (setup/teardown)

Fonctionnalités avancées :
- Changer le modèle à la volée via request_context.model
- Sauter l'appel avec SkipModelRequest(response)
- before_* → dans l'ordre d'enregistrement
- after_*  → en ordre inverse
"""


# =====================================================================
# PARTIE 1 : before_model_request — Avant l'appel
# =====================================================================

# Inspecter ou modifier la requête avant qu'elle parte au modèle.

hooks_before = Hooks()


@hooks_before.on.before_model_request
async def inspect_messages(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Inspecte et logge les messages avant envoi."""
    nb_messages = len(request_context.messages)
    logger.info(f"[before] {nb_messages} messages dans la requête")
    # On peut modifier request_context.messages ici si besoin
    return request_context


agent_before = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_before],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : after_model_request — Après l'appel
# =====================================================================

# Inspecter ou modifier la réponse après réception.

hooks_after = Hooks()


@hooks_after.on.after_model_request
async def inspect_response(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """Inspecte la réponse du modèle."""
    logger.info(f"[after] Réponse avec {len(response.parts)} parties")
    for i, part in enumerate(response.parts):
        logger.info(f"  Part {i}: {type(part).__name__}")
    return response


agent_after = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_after],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 3 : model_request (wrap) — Middleware complet
# =====================================================================

# hooks.on.model_request correspond à wrap_model_request.
# On wrappe l'appel entier pour faire du setup/teardown.

hooks_wrap = Hooks()


@hooks_wrap.on.model_request
async def timed_request(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    """Mesure le temps d'exécution de chaque appel au modèle."""
    import time

    start = time.time()
    response = await handler(request_context)
    duration = time.time() - start
    logger.info(f"[wrap] Appel modèle terminé en {duration:.2f}s")
    return response


agent_wrap = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_wrap],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 4 : Combiner before + after + wrap
# =====================================================================

# Plusieurs hooks sur la même instance, tous se combinent.
# Ordre : before → wrap(avant handler) → appel → wrap(après handler) → after

hooks_combo = Hooks()


@hooks_combo.on.before_model_request
async def combo_before(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    logger.info("[combo] 1. before_model_request")
    return request_context


@hooks_combo.on.model_request
async def combo_wrap(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    logger.info("[combo] 2. wrap - avant handler")
    response = await handler(request_context)
    logger.info("[combo] 4. wrap - après handler")
    return response


@hooks_combo.on.after_model_request
async def combo_after(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    logger.info("[combo] 3. after_model_request (ordre inverse)")
    return response


agent_combo = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_combo],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : before_model_request ---
    logger.info("=== before_model_request ===")
    result = agent_before.run_sync("Bonjour")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : after_model_request ---
    logger.info("=== after_model_request ===")
    result = agent_after.run_sync("Dis bonjour")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : wrap (timing) ---
    logger.info("=== wrap_model_request (timing) ===")
    result = agent_wrap.run_sync("Salut !")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Combo before + wrap + after ---
    logger.info("=== Combo before + wrap + after ===")
    result = agent_combo.run_sync("Hello")
    logger.success(f"Réponse : {result.output}")
