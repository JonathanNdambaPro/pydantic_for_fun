import asyncio
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks, WrapModelRequestHandler
from pydantic_ai.messages import ModelResponse, ToolCallPart

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS — Intercepter le comportement de l'agent avec des décorateurs

Hooks est le moyen le plus simple d'ajouter des lifecycle hooks
sans sous-classer AbstractCapability. Juste des décorateurs.

Quand utiliser Hooks vs AbstractCapability :
- Hooks               → logging, metrics, intercepteurs rapides, scripts
- AbstractCapability   → réutilisable, combine tools + hooks + instructions

Enregistrement des hooks :
1. Via décorateur    → @hooks.on.before_model_request
2. Via constructeur  → Hooks(before_model_request=my_fn)

Points d'interception (même noms que AbstractCapability) :
- Run            → before_run, after_run, run (wrap), run_error
- Node           → before_node_run, after_node_run, node_run, node_run_error
- Model request  → before_model_request, after_model_request, model_request, model_request_error
- Tool validate  → before_tool_validate, after_tool_validate, tool_validate, tool_validate_error
- Tool execute   → before_tool_execute, after_tool_execute, tool_execute, tool_execute_error
- Tools prepare  → prepare_tools
- Event stream   → run_event_stream, event

Fonctionnalités :
- timeout        → limite de temps par hook (HookTimeoutError si dépassé)
- tools=[...]    → filtrer les hooks de tools par nom
- wrap hooks     → middleware complet (before + after en un seul hook)
- ModelRetry     → demander au modèle de réessayer depuis un hook
- Sync et async  → les deux sont acceptés

Ordre d'exécution (multi-hooks) :
- before_* → dans l'ordre d'enregistrement
- after_*  → en ordre inverse
- wrap_*   → middleware (premier = couche externe)
"""


# =====================================================================
# PARTIE 1 : Hooks via décorateurs (@hooks.on.*)
# =====================================================================

hooks_logging = Hooks()


@hooks_logging.on.before_model_request
async def log_request(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Logge chaque requête envoyée au modèle."""
    logger.info(f"[hook] Envoi de {len(request_context.messages)} messages au modèle")
    return request_context


@hooks_logging.on.after_model_request
async def log_response(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """Logge chaque réponse reçue du modèle."""
    logger.info(f"[hook] Réponse reçue : {len(response.parts)} parties")
    return response


agent_logged = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_logging],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Hooks via constructeur
# =====================================================================

# Alternative aux décorateurs : passer les fonctions au constructeur.


async def log_request_alt(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    logger.info(f"[constructeur] {len(request_context.messages)} messages")
    return request_context


agent_constructor = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[Hooks(before_model_request=log_request_alt)],
)


# =====================================================================
# PARTIE 3 : Wrap hooks — middleware complet
# =====================================================================

# hooks.on.model_request correspond à wrap_model_request.
# On wrappe l'opération pour faire du setup/teardown.

hooks_wrap = Hooks()


@hooks_wrap.on.model_request
async def timed_request(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    handler: WrapModelRequestHandler,
) -> ModelResponse:
    """Mesure le temps de chaque appel au modèle."""
    import time

    start = time.time()
    response = await handler(request_context)
    duration = time.time() - start
    logger.info(f"[wrap] Appel modèle : {duration:.2f}s")
    return response


agent_wrapped = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_wrap],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 4 : Tool hooks avec filtre par nom
# =====================================================================

# On peut cibler des hooks sur des tools spécifiques via tools=[...].

hooks_tools = Hooks()
audit_log: list[str] = []


@hooks_tools.on.before_tool_execute(tools=["send_email"])
async def audit_email(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Audite uniquement les appels à send_email."""
    audit_log.append(f"AUDIT: {call.tool_name}({args})")
    logger.warning(f"[audit] {call.tool_name} appelé avec {args}")
    return args


agent_audit = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_tools],
    instructions="Tu es un assistant email.",
)


@agent_audit.tool_plain
def send_email(to: str, subject: str) -> str:
    """Envoie un email."""
    return f"Email envoyé à {to} : {subject}"


@agent_audit.tool_plain
def read_inbox() -> str:
    """Lit la boîte de réception."""
    return "3 emails non lus."


# =====================================================================
# PARTIE 5 : Timeout sur les hooks
# =====================================================================

# Chaque hook peut avoir un timeout. Si dépassé → HookTimeoutError.

hooks_timeout = Hooks()


@hooks_timeout.on.before_model_request(timeout=5.0)
async def quick_check(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Vérification rapide avec timeout de 5 secondes."""
    # Si cette vérification prend plus de 5s → HookTimeoutError
    await asyncio.sleep(0.01)
    logger.info("[timeout] Vérification OK (< 5s)")
    return request_context


# =====================================================================
# PARTIE 6 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Hooks de logging ---
    logger.info("=== Hooks de logging (décorateurs) ===")
    result = agent_logged.run_sync("Bonjour, comment ça va ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Wrap hook (timing) ---
    logger.info("=== Wrap hook (timing) ===")
    result = agent_wrapped.run_sync("Dis bonjour")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Tool hook filtré ---
    logger.info("=== Tool hook filtré (audit email) ===")
    result = agent_audit.run_sync("Envoie un email à alice@test.com avec le sujet 'Bonjour'")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Audit log : {audit_log}")
