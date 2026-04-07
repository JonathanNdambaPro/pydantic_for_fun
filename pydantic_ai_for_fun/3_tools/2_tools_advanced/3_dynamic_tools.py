import logfire
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, Tool, ToolDefinition

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
DYNAMIC TOOLS — Outils dynamiques avec `prepare`

Par défaut, tous les tools enregistrés sur un agent sont disponibles
à chaque étape du run. Mais parfois on veut :
- Masquer un tool selon le contexte (permissions, état, deps…)
- Modifier la description ou le schéma d'un tool dynamiquement

C'est le rôle de la fonction `prepare`.

Comment ça marche :
1. À CHAQUE étape du run, avant d'envoyer la liste des tools au modèle,
   Pydantic AI appelle la fonction `prepare` de chaque tool.
2. `prepare` reçoit le RunContext et un ToolDefinition pré-construit.
3. Elle retourne :
   - le ToolDefinition tel quel     → le tool est inclus normalement
   - un ToolDefinition modifié      → le tool est inclus avec les modifs
   - None                           → le tool est EXCLU de cette étape

On peut utiliser `prepare` avec :
- @agent.tool(prepare=...)
- @agent.tool_plain(prepare=...)
- Tool(fn, prepare=...)

La signature de prepare est : ToolPrepareFunc
  async def prepare(ctx: RunContext[T], tool_def: ToolDefinition) -> ToolDefinition | None
"""


# =====================================================================
# PARTIE 1 : Masquer un tool selon les dépendances
# =====================================================================

# Cas d'usage : un tool "admin" qui ne doit être disponible que
# si l'utilisateur a le rôle "admin". Si deps != "admin",
# le modèle ne voit même pas que le tool existe.

agent_admin = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=str,  # Le rôle de l'utilisateur
    instructions=(
        "Tu es un assistant système. "
        "Utilise les outils disponibles pour répondre. "
        "Si tu n'as pas d'outil adapté, dis-le poliment."
    ),
)


async def only_for_admin(
    ctx: RunContext[str], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Ne rend le tool disponible que si le rôle est 'admin'."""
    if ctx.deps == 'admin':
        logger.info("Rôle admin détecté → tool 'reset_system' activé")
        return tool_def
    else:
        logger.info(f"Rôle '{ctx.deps}' → tool 'reset_system' masqué")
        return None  # Le modèle ne verra pas ce tool


@agent_admin.tool(prepare=only_for_admin)
def reset_system(ctx: RunContext[str]) -> str:
    """Réinitialise le système. Réservé aux administrateurs."""
    logger.warning(f"Système réinitialisé par {ctx.deps}")
    return "Système réinitialisé avec succès."


@agent_admin.tool_plain
def get_system_status() -> str:
    """Retourne le statut actuel du système."""
    return "Système opérationnel — charge CPU : 23%, mémoire : 4.2 Go / 16 Go"


# =====================================================================
# PARTIE 2 : Modifier le schéma d'un tool dynamiquement
# =====================================================================

# Cas d'usage : un outil de salutation dont la description du paramètre
# change selon qu'on salue un humain ou une machine.
# On utilise ici le dataclass Tool au lieu du décorateur.


def greet(name: str) -> str:
    """Salue quelqu'un par son nom."""
    logger.info(f"Salutation → {name}")
    return f"Bonjour {name} !"


async def prepare_greet(
    ctx: RunContext[Literal['humain', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Adapte la description du paramètre 'name' selon le type d'entité."""
    description = f"Nom de l'{ctx.deps} à saluer."
    tool_def.parameters_json_schema['properties']['name']['description'] = description
    logger.info(f"Description du param 'name' mise à jour → '{description}'")
    return tool_def


# Création du tool via le dataclass Tool (alternative aux décorateurs)
greet_tool = Tool(greet, prepare=prepare_greet)

agent_greet = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    tools=[greet_tool],
    deps_type=Literal['humain', 'machine'],
    instructions="Tu es un assistant poli. Utilise l'outil pour saluer.",
)


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Tool masqué pour un utilisateur normal ---
    logger.info("=== Utilisateur 'viewer' essaie de reset ===")
    result = agent_admin.run_sync(
        "Réinitialise le système s'il te plaît",
        deps="viewer",
    )
    logger.success(f"Réponse (viewer) : {result.output}")

    # --- Démo 2 : Tool visible pour un admin ---
    logger.info("=== Utilisateur 'admin' reset le système ===")
    result = agent_admin.run_sync(
        "Réinitialise le système s'il te plaît",
        deps="admin",
    )
    logger.success(f"Réponse (admin) : {result.output}")

    # --- Démo 3 : Description dynamique — saluer un humain ---
    logger.info("=== Salutation d'un humain ===")
    result = agent_greet.run_sync("Salue Marie", deps="humain")
    logger.success(f"Réponse (humain) : {result.output}")

    # --- Démo 4 : Description dynamique — saluer une machine ---
    logger.info("=== Salutation d'une machine ===")
    result = agent_greet.run_sync("Salue Robot-X", deps="machine")
    logger.success(f"Réponse (machine) : {result.output}")
