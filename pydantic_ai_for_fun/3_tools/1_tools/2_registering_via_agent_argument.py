import random

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, Tool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
ENREGISTREMENT DE TOOLS VIA L'ARGUMENT `tools` DE L'AGENT

Au lieu des décorateurs @agent.tool / @agent.tool_plain, on peut
enregistrer les tools directement via le paramètre `tools=[...]`
du constructeur Agent.

Avantages :
- Réutilisation : les mêmes fonctions peuvent être partagées entre
  plusieurs agents sans duplication
- Contrôle fin : en wrappant dans Tool(), on peut explicitement
  spécifier `takes_ctx=True/False` au lieu de laisser Pydantic AI
  deviner via l'introspection des paramètres

Deux approches :
1. Passer des fonctions brutes → Pydantic AI détecte automatiquement
   si le premier argument est un RunContext
2. Passer des instances Tool() → on contrôle explicitement takes_ctx
"""

instructions = (
    "Tu es un jeu de dé. Lance le dé et vérifie si le résultat "
    "correspond à la supposition du joueur. Si oui, dis-lui qu'il a gagné. "
    "Utilise le nom du joueur dans ta réponse."
)


# =====================================================================
# PARTIE 1 : Définir les tools comme de simples fonctions
# =====================================================================

# Ces fonctions ne sont rattachées à aucun agent en particulier.
# Elles peuvent être réutilisées par autant d'agents qu'on veut.

def roll_dice() -> str:
    """Lance un dé à six faces et retourne le résultat."""
    result = random.randint(1, 6)  # noqa: S311
    logger.info(f"Dé lancé → {result}")
    return str(result)


def get_player_name(ctx: RunContext[str]) -> str:
    """Récupère le nom du joueur."""
    logger.info(f"Nom du joueur : {ctx.deps}")
    return ctx.deps


# =====================================================================
# PARTIE 2 : Approche A — passage de fonctions brutes
# =====================================================================

# Pydantic AI inspecte automatiquement la signature de chaque fonction.
# Si le premier paramètre est un RunContext → takes_ctx=True (implicite).
# Sinon → takes_ctx=False.

agent_a = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=str,
    tools=[roll_dice, get_player_name],  # Détection automatique du contexte
    instructions=instructions,
)


# =====================================================================
# PARTIE 3 : Approche B — passage d'instances Tool() explicites
# =====================================================================

# En wrappant dans Tool(), on spécifie explicitement takes_ctx.
# C'est plus verbeux mais plus lisible et sans ambiguïté.
# Utile si la détection automatique ne convient pas ou si on veut
# être explicite pour la maintenabilité du code.

agent_b = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=str,
    tools=[
        Tool(roll_dice, takes_ctx=False),       # Pas besoin du contexte
        Tool(get_player_name, takes_ctx=True),   # A besoin du RunContext
    ],
    instructions=instructions,
)


# =====================================================================
# PARTIE 4 : Exécution — les deux agents partagent les mêmes tools
# =====================================================================

if __name__ == "__main__":
    dice_result = {}

    # Agent A : détection automatique du contexte
    logger.info("=== Agent A (auto-détection) — Yashar devine 6 ===")
    dice_result["a"] = agent_a.run_sync("Ma supposition est 6", deps="Yashar")
    logger.success(f"Réponse A : {dice_result['a'].output}")

    # Agent B : Tool() explicite
    logger.info("=== Agent B (Tool explicite) — Anne devine 4 ===")
    dice_result["b"] = agent_b.run_sync("Ma supposition est 4", deps="Anne")
    logger.success(f"Réponse B : {dice_result['b'].output}")
