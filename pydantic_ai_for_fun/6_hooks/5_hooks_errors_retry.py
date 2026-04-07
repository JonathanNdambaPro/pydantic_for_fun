from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS ERREURS & RETRY — Gestion d'erreurs et relance du modèle

Error hooks (*_error) utilisent la sémantique raise-to-propagate,
return-to-recover :
- Raise l'erreur originale → propagation (défaut)
- Raise une autre exception → transformation de l'erreur
- Return un résultat       → suppression de l'erreur (recovery)

ModelRetry depuis les hooks :
- Même exception que dans les tools et output validators
- Model request hooks → RetryPromptPart envoyé au modèle
- Tool hooks          → tool retry prompt
- Compte contre output_retries (model) ou max_retries (tool)

Types de hooks d'erreur :
- run_error, node_run_error
- model_request_error
- tool_validate_error, tool_execute_error
"""


# =====================================================================
# PARTIE 1 : Error hooks — Recovery d'erreur
# =====================================================================

# Un error hook peut intercepter une erreur et décider de :
# - la propager (raise), la transformer, ou la supprimer (return)

hooks_error = Hooks()
error_log: list[str] = []


@hooks_error.on.tool_execute_error
async def recover_tool_error(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    error: Exception,
) -> str:
    """Intercepte les erreurs d'outils et retourne un fallback."""
    error_log.append(f"error: {call.tool_name} → {error}")
    logger.warning(f"[error] {call.tool_name} a échoué : {error}")
    # Return un résultat → supprime l'erreur (recovery)
    return f"Erreur récupérée : l'outil {call.tool_name} n'est pas disponible"


agent_error = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_error],
    instructions="Tu es un assistant. Si un outil échoue, explique l'erreur.",
)


@agent_error.tool_plain
def risky_operation(action: str) -> str:
    """Opération qui peut échouer."""
    if action == "crash":
        raise RuntimeError("Boom ! L'opération a planté.")
    return f"Opération '{action}' réussie"


# =====================================================================
# PARTIE 2 : ModelRetry depuis after_model_request
# =====================================================================

# On peut relancer le modèle si la réponse ne convient pas.
# L'original est préservé dans l'historique pour que le modèle
# voie ce qu'il a dit.

hooks_retry = Hooks()
retry_count = 0


@hooks_retry.on.after_model_request
async def check_for_placeholders(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    response: ModelResponse,
) -> ModelResponse:
    """Rejette les réponses contenant des placeholders."""
    global retry_count
    response_text = str(response.parts)
    if "PLACEHOLDER" in response_text or "TODO" in response_text:
        retry_count += 1
        logger.warning(f"[retry] Placeholder détecté ! Retry #{retry_count}")
        raise ModelRetry("La réponse contient du texte placeholder. Fournis des données réelles.")
    return response


agent_retry = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_retry],
    instructions="Tu es un assistant. Réponds toujours avec des données concrètes, jamais de placeholders.",
)


# =====================================================================
# PARTIE 3 : ModelRetry depuis tool hooks
# =====================================================================

# ModelRetry dans les tool hooks est converti en tool retry prompt.
# Compte contre max_retries de l'outil.

hooks_tool_retry = Hooks()


@hooks_tool_retry.on.after_tool_execute
async def validate_tool_result(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: Any,
) -> Any:
    """Vérifie que le résultat de l'outil est valide."""
    if result and "erreur" in str(result).lower():
        logger.warning(f"[tool_retry] Résultat suspect pour {call.tool_name}")
        raise ModelRetry(f"L'outil {call.tool_name} a retourné une erreur. Essaie avec d'autres paramètres.")
    return result


agent_tool_retry = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_tool_retry],
    instructions="Tu es un assistant avec des outils.",
)


@agent_tool_retry.tool_plain
def search_database(query: str) -> str:
    """Cherche dans la base de données."""
    if len(query) < 3:
        return "Erreur : requête trop courte"
    return f"Résultats pour '{query}': utilisateur trouvé"


# =====================================================================
# PARTIE 4 : model_request_error — Transformer les erreurs modèle
# =====================================================================

hooks_model_error = Hooks()


@hooks_model_error.on.model_request_error
async def handle_model_error(
    ctx: RunContext[None],
    *,
    request_context: ModelRequestContext,
    error: Exception,
) -> ModelResponse:
    """Intercepte les erreurs du modèle et retourne un fallback."""
    logger.error(f"[model_error] Erreur modèle : {error}")
    # On pourrait retourner une ModelResponse de fallback ici
    # ou raise l'erreur pour la propager
    raise error  # Propagation par défaut


agent_model_error = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_model_error],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Error recovery ---
    logger.info("=== Error hooks (recovery) ===")
    result = agent_error.run_sync("Fais l'action 'crash'")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Error log : {error_log}")

    # --- Démo 2 : ModelRetry depuis after_model_request ---
    logger.info("=== ModelRetry (after_model_request) ===")
    result = agent_retry.run_sync("Donne-moi 3 faits intéressants sur Python")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : ModelRetry depuis tool hooks ---
    logger.info("=== ModelRetry (tool hooks) ===")
    result = agent_tool_retry.run_sync("Cherche 'pydantic' dans la base")
    logger.success(f"Réponse : {result.output}")
