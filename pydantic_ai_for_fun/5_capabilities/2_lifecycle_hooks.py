from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
LIFECYCLE HOOKS — Intercepter le cycle de vie de l'agent

Les hooks permettent d'intercepter et modifier le comportement à
5 points du cycle de vie, chacun avec 4 variantes :

- before_*     → avant l'action, peut modifier les inputs
- after_*      → après le succès, peut modifier les outputs
- wrap_*       → middleware complet (décide si/comment exécuter)
- on_*_error   → quand l'action échoue, peut récupérer l'erreur

Points d'interception :
1. Run hooks        → avant/après le run complet de l'agent
2. Node hooks       → avant/après chaque nœud du graph (UserPrompt, ModelRequest, CallTools)
3. Model hooks      → avant/après chaque appel au modèle
4. Tool hooks       → avant/après validation ET exécution des tools
5. Event stream     → observer/transformer le flux d'événements en streaming

Flow d'erreur :
  before_X → wrap_X(handler)
    ├─ succès → after_X (modifier le résultat)
    └─ échec  → on_X_error
          ├─ re-raise  → l'erreur se propage
          └─ return    → l'erreur est récupérée

Pour les cas simples, utiliser Hooks (décorateurs) au lieu de
sous-classer AbstractCapability.
"""


# =====================================================================
# PARTIE 1 : Node hooks — logger les nœuds traversés
# =====================================================================

# wrap_node_run s'exécute pour chaque nœud du graph de l'agent.
# Utile pour observer les transitions, ajouter du logging par étape.


@dataclass
class NodeLogger(AbstractCapability[Any]):
    """Logge chaque nœud exécuté pendant un run."""

    nodes: list[str] = field(default_factory=list)

    async def wrap_node_run(
        self,
        ctx: RunContext[Any],
        *,
        node: AgentNode[Any],
        handler: WrapNodeRunHandler[Any],
    ) -> NodeResult[Any]:
        node_name = type(node).__name__
        self.nodes.append(node_name)
        logger.info(f"[node] {node_name}")
        return await handler(node)


# =====================================================================
# PARTIE 2 : Model request hooks — logging des requêtes
# =====================================================================

# wrap_model_request permet d'observer ou timer les appels au modèle.


@dataclass
class RequestLogger(AbstractCapability[Any]):
    """Logge les requêtes et réponses du modèle."""

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        logger.info(
            f"[model] Step {ctx.run_step}, "
            f"{len(request_context.messages)} messages"
        )
        response = await handler(request_context)
        logger.info(f"[model] Réponse : {len(response.parts)} parties")
        return response


# =====================================================================
# PARTIE 3 : Tool hooks — logging des appels de tools
# =====================================================================


@dataclass
class ToolLogger(AbstractCapability[Any]):
    """Logge les appels de tools et leurs résultats."""

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        logger.info(f"[tool] {call.tool_name}({args})")
        result = await handler(args)
        logger.info(f"[tool] Résultat : {result!r}")
        return result


# =====================================================================
# PARTIE 4 : Error hooks — récupération d'erreurs
# =====================================================================

# on_*_error permet de récupérer d'une erreur :
# - raise l'erreur → elle se propage (défaut)
# - raise une autre exception → transforme l'erreur
# - return un résultat → supprime l'erreur, utilise le résultat


@dataclass
class ErrorRecovery(AbstractCapability[Any]):
    """Récupère des erreurs modèle avec une réponse de fallback."""

    errors: list[str] = field(default_factory=list)

    async def on_model_request_error(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        self.errors.append(f"Erreur modèle : {error}")
        logger.error(f"[error] Erreur modèle récupérée : {error}")
        # Retourne une réponse de fallback au lieu de crasher
        return ModelResponse(
            parts=[TextPart(content="Service temporairement indisponible.")]
        )


# =====================================================================
# PARTIE 5 : Guardrail — redaction PII dans les réponses
# =====================================================================

# after_model_request permet de modifier la réponse du modèle.
# Ici on redacte les emails et numéros de téléphone.


@dataclass
class PIIRedaction(AbstractCapability[Any]):
    """Redacte les emails et numéros de téléphone des réponses."""

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                part.content = re.sub(
                    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                    "[EMAIL REDACTÉ]",
                    part.content,
                )
                part.content = re.sub(
                    r"\b\d{2}[-. ]?\d{2}[-. ]?\d{2}[-. ]?\d{2}[-. ]?\d{2}\b",
                    "[TÉLÉPHONE REDACTÉ]",
                    part.content,
                )
        return response


# =====================================================================
# PARTIE 6 : Composition et exécution
# =====================================================================

node_logger = NodeLogger()

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[
        node_logger,
        RequestLogger(),
        ToolLogger(),
        PIIRedaction(),
    ],
    instructions="Tu es un assistant. Réponds en français.",
)


@agent.tool_plain
def get_contact(name: str) -> str:
    """Retourne les coordonnées d'une personne."""
    return f"{name} : jean.dupont@email.com, 06 12 34 56 78"


if __name__ == "__main__":
    # --- Démo 1 : Tous les hooks en action ---
    logger.info("=== Lifecycle hooks complets ===")
    result = agent.run_sync("Donne-moi les coordonnées de Jean Dupont")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Nœuds traversés : {node_logger.nodes}")
