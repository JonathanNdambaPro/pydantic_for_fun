from dataclasses import dataclass
from datetime import datetime
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability, PrepareTools
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CUSTOM CAPABILITIES — Créer ses propres capabilities

Pour créer une capability custom, on sous-classe AbstractCapability
et on override les méthodes dont on a besoin.

Méthodes de configuration (appelées à la construction de l'agent) :
- get_toolset()          → fournir un toolset
- get_builtin_tools()    → fournir des builtin tools
- get_wrapper_toolset()  → wrapper le toolset de l'agent
- get_instructions()     → ajouter des instructions
- get_model_settings()   → ajouter des model settings

Lifecycle :
- for_run(ctx)       → retourner une instance fraîche par run
                       (isolation de l'état mutable entre runs)
- for_run_step(ctx)  → appelé à chaque étape du run

prepare_tools(ctx, tool_defs) :
  → Filtrer ou modifier les tools visibles par le modèle à chaque étape
  → Contrôle la visibilité, pas l'exécution (utiliser les hooks pour ça)

Pour les cas simples, utiliser PrepareTools (capability built-in)
au lieu de sous-classer.
"""


# =====================================================================
# PARTIE 1 : Capability qui fournit des tools
# =====================================================================

# Une capability peut embarquer ses propres tools via un FunctionToolset.

math_toolset = FunctionToolset()


@math_toolset.tool_plain
def add(a: float, b: float) -> float:
    """Additionne deux nombres."""
    logger.info(f"[math] {a} + {b} = {a + b}")
    return a + b


@math_toolset.tool_plain
def multiply(a: float, b: float) -> float:
    """Multiplie deux nombres."""
    logger.info(f"[math] {a} * {b} = {a * b}")
    return a * b


@dataclass
class MathTools(AbstractCapability[Any]):
    """Fournit des opérations mathématiques basiques."""

    def get_toolset(self) -> AgentToolset[Any] | None:
        return math_toolset


# =====================================================================
# PARTIE 2 : Capability qui fournit des instructions dynamiques
# =====================================================================

# get_instructions peut retourner une string statique OU une
# fonction qui prend RunContext et retourne une string.


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """Injecte la date/heure actuelle dans les instructions."""

    def get_instructions(self):
        def _get_time(ctx: RunContext[Any]) -> str:
            return f"La date et l'heure actuelles sont {datetime.now().isoformat()}."

        return _get_time


# =====================================================================
# PARTIE 3 : PrepareTools — filtrer les tools visibles
# =====================================================================

# PrepareTools est une capability built-in qui wrappe une fonction
# de filtrage. Pas besoin de sous-classer.


async def hide_dangerous(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Cache les tools dangereux (delete_, drop_)."""
    return [
        td
        for td in tool_defs
        if not td.name.startswith("delete_") and not td.name.startswith("drop_")
    ]


# Équivalent avec une sous-classe custom :
@dataclass
class HideDangerousTools(AbstractCapability[Any]):
    """Cache les tools dont le nom commence par certains préfixes."""

    hidden_prefixes: tuple[str, ...] = ("delete_", "drop_")

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [
            td
            for td in tool_defs
            if not any(td.name.startswith(p) for p in self.hidden_prefixes)
        ]


# =====================================================================
# PARTIE 4 : Isolation par run (for_run)
# =====================================================================

# Si la capability a de l'état mutable, for_run retourne une
# instance fraîche pour éviter les fuites entre runs.


@dataclass
class RequestCounter(AbstractCapability[Any]):
    """Compte les requêtes modèle par run."""

    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> RequestCounter:
        return RequestCounter()  # Instance fraîche à chaque run

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.count += 1
        logger.info(f"[counter] Requête #{self.count}")
        return request_context


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

# Agent avec plusieurs capabilities custom composées.

counter = RequestCounter()

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[
        MathTools(),
        KnowsCurrentTime(),
        PrepareTools(hide_dangerous),
        counter,
    ],
    instructions="Tu es un assistant. Réponds en français.",
)


@agent.tool_plain
def delete_everything() -> str:
    """Supprime tout. Dangereux !"""
    return "Tout supprimé."


@agent.tool_plain
def get_status() -> str:
    """Retourne le statut du système."""
    return "Système OK."


if __name__ == "__main__":
    # --- Démo 1 : MathTools + KnowsCurrentTime ---
    logger.info("=== Capabilities composées ===")
    result = agent.run_sync("Quelle heure est-il et combien font 7 * 8 ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : PrepareTools (delete_everything est caché) ---
    logger.info("=== Tool dangereux caché ===")
    result = agent.run_sync("Supprime tout")
    logger.success(f"Réponse : {result.output}")
    # Le modèle ne voit pas delete_everything → ne peut pas l'appeler

    # --- Démo 3 : Counter (isolation par run) ---
    logger.info(f"Counter partagé : {counter.count}")
    # Toujours 0 car for_run retourne une instance fraîche
