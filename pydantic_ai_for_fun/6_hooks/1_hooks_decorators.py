import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelResponse

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS DÉCORATEURS — Enregistrer des hooks avec @hooks.on.*

Les Hooks interceptent le comportement de l'agent à chaque étape
d'un run (requêtes modèle, appels d'outils, streaming) via de
simples décorateurs ou arguments au constructeur. Pas de sous-classe.

Hooks est le moyen recommandé pour ajouter des lifecycle hooks
d'application (logging, metrics, validation légère).

Deux façons d'enregistrer :
1. Via décorateur    → @hooks.on.before_model_request
2. Via constructeur  → Hooks(before_model_request=my_fn)

Points clés :
- Plusieurs hooks sur le même événement → exécutés dans l'ordre
- Sync et async sont acceptés (sync wrappé automatiquement)
- Décorateur nu ou avec paramètres (timeout, tools)
"""


# =====================================================================
# PARTIE 1 : Quick start — Hooks via décorateurs
# =====================================================================

# On crée une instance Hooks, on enregistre via @hooks.on.*,
# puis on passe l'instance comme capability de l'agent.

hooks_logging = Hooks()


@hooks_logging.on.before_model_request
async def log_request(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Logge le nombre de messages envoyés au modèle."""
    logger.info(f"[before] Envoi de {len(request_context.messages)} messages au modèle")
    return request_context


@hooks_logging.on.after_model_request
async def log_response(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """Logge la réponse reçue du modèle."""
    logger.info(f"[after] Réponse reçue : {len(response.parts)} parties")
    return response


agent_decorator = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_logging],
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Hooks via constructeur (kwargs)
# =====================================================================

# Alternative aux décorateurs : passer les fonctions directement
# au constructeur Hooks. Utile pour des hooks simples ou one-liner.


async def log_request_constructor(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    logger.info(f"[constructeur] {len(request_context.messages)} messages")
    return request_context


agent_constructor = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[Hooks(before_model_request=log_request_constructor)],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 3 : Décorateur avec paramètres
# =====================================================================

# On peut passer des paramètres au décorateur : timeout, tools, etc.

hooks_params = Hooks()


# Décorateur nu (sans parenthèses)
@hooks_params.on.before_model_request
async def bare_hook(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    logger.info("[bare] Hook sans paramètres")
    return request_context


# Décorateur avec paramètre timeout
@hooks_params.on.after_model_request(timeout=5.0)
async def timed_hook(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    logger.info("[timed] Hook avec timeout de 5s")
    return response


agent_params = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_params],
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Hooks via décorateurs ---
    logger.info("=== Hooks via décorateurs ===")
    result = agent_decorator.run_sync("Bonjour, comment ça va ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Hooks via constructeur ---
    logger.info("=== Hooks via constructeur ===")
    result = agent_constructor.run_sync("Dis bonjour")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Décorateur avec paramètres ---
    logger.info("=== Décorateur avec paramètres ===")
    result = agent_params.run_sync("Salut !")
    logger.success(f"Réponse : {result.output}")
