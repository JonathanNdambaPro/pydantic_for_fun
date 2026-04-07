import asyncio
import time
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks, HookTimeoutError, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse, ToolCallPart

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WRAP HOOKS & TIMEOUTS — Middleware et limites de temps

Wrap hooks permettent d'entourer une opération avec du setup/teardown.
Dans hooks.on, le préfixe wrap_ est omis :
- hooks.on.model_request     → wrap_model_request
- hooks.on.run               → wrap_run
- hooks.on.node_run           → wrap_node_run
- hooks.on.tool_validate      → wrap_tool_validate
- hooks.on.tool_execute       → wrap_tool_execute

Ordre avec multi-hooks :
- before_* → dans l'ordre d'enregistrement
- after_*  → en ordre inverse
- wrap_*   → middleware nestés (premier = couche externe)

Timeouts :
- Chaque hook peut avoir un timeout en secondes
- Si dépassé → HookTimeoutError avec hook_name et timeout
- Via décorateur : @hooks.on.before_model_request(timeout=5.0)
- Via constructeur : timeout dans les kwargs
"""


# =====================================================================
# PARTIE 1 : Wrap hook — Middleware model request
# =====================================================================

# Le wrap hook reçoit un handler qu'on appelle pour exécuter l'opération.
# On peut faire du setup avant et du teardown après.

hooks_wrap = Hooks()
wrap_log: list[str] = []


@hooks_wrap.on.model_request
async def measure_request(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    """Middleware qui mesure le temps et logge avant/après."""
    wrap_log.append("before")
    start = time.time()

    response = await handler(request_context)

    duration = time.time() - start
    wrap_log.append("after")
    logger.info(f"[wrap] Requête modèle : {duration:.2f}s")
    return response


agent_wrap = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_wrap],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 2 : Wrap hooks nestés — Middleware en oignon
# =====================================================================

# Quand plusieurs wrap hooks sont enregistrés, ils se nestent.
# Le premier enregistré est la couche externe (comme des middlewares).

hooks_nested = Hooks()
nested_log: list[str] = []


@hooks_nested.on.model_request
async def outer_layer(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    """Couche externe du middleware."""
    nested_log.append("outer:enter")
    response = await handler(request_context)
    nested_log.append("outer:exit")
    return response


@hooks_nested.on.model_request
async def inner_layer(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    """Couche interne du middleware."""
    nested_log.append("inner:enter")
    response = await handler(request_context)
    nested_log.append("inner:exit")
    return response


agent_nested = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_nested],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 3 : Wrap hook sur tool_execute
# =====================================================================

hooks_tool_wrap = Hooks()


@hooks_tool_wrap.on.tool_execute
async def wrap_tool(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    handler: Any,
) -> Any:
    """Wrappe l'exécution de chaque outil avec du timing."""
    start = time.time()
    logger.info(f"[wrap_tool] Début {call.tool_name}")

    result = await handler(args)

    duration = time.time() - start
    logger.info(f"[wrap_tool] Fin {call.tool_name} en {duration:.4f}s")
    return result


agent_tool_wrap = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_tool_wrap],
    instructions="Tu es un assistant.",
)


@agent_tool_wrap.tool_plain
def slow_search(query: str) -> str:
    """Simule une recherche lente."""
    import time as t

    t.sleep(0.1)
    return f"Résultats pour '{query}': 42 résultats trouvés"


# =====================================================================
# PARTIE 4 : Timeouts sur les hooks
# =====================================================================

# Chaque hook peut avoir un timeout. Si le hook dépasse le temps
# imparti, une HookTimeoutError est levée.

hooks_timeout = Hooks()


@hooks_timeout.on.before_model_request(timeout=5.0)
async def quick_check(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Vérification rapide avec timeout de 5 secondes."""
    await asyncio.sleep(0.01)  # Rapide, pas de timeout
    logger.info("[timeout] Vérification OK (< 5s)")
    return request_context


# Exemple de hook qui dépasserait le timeout (pour la démo)
hooks_slow = Hooks()


@hooks_slow.on.before_model_request(timeout=0.01)
async def too_slow(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Hook volontairement trop lent → HookTimeoutError."""
    await asyncio.sleep(10)  # Sera interrompu par le timeout
    return request_context  # pragma: no cover


agent_timeout = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_timeout],
    instructions="Tu es un assistant.",
)

agent_slow = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_slow],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Wrap hook simple ---
    logger.info("=== Wrap hook (timing) ===")
    result = agent_wrap.run_sync("Bonjour")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Wrap log : {wrap_log}")

    # --- Démo 2 : Wrap hooks nestés ---
    logger.info("=== Wrap hooks nestés (oignon) ===")
    result = agent_nested.run_sync("Salut")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Nested log : {nested_log}")
    # Attendu : outer:enter → inner:enter → inner:exit → outer:exit

    # --- Démo 3 : Wrap tool ---
    logger.info("=== Wrap tool_execute ===")
    result = agent_tool_wrap.run_sync("Cherche 'pydantic ai hooks'")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Timeout OK ---
    logger.info("=== Timeout hook (OK) ===")
    result = agent_timeout.run_sync("Test timeout")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 5 : Timeout dépassé ---
    logger.info("=== Timeout hook (dépassé → HookTimeoutError) ===")
    try:
        agent_slow.run_sync("Test slow")
    except HookTimeoutError as e:
        logger.error(f"Hook timeout : {e.hook_name} après {e.timeout}s")
