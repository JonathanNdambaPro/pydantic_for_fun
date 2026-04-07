import random

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
FUNCTION TOOLS (Outils de fonction)

Les function tools permettent au modèle d'exécuter des actions et de
récupérer des informations supplémentaires pour construire sa réponse.

Cas d'usage :
- Quand le modèle doit effectuer une action (lancer un dé, appeler une API…)
- Quand il est impossible de tout mettre dans les instructions
- Quand on veut rendre le comportement plus déterministe en déléguant
  de la logique à du code classique (pas forcément IA)

Il y a plusieurs façons d'enregistrer des tools :
1. @agent.tool        → le tool a accès au RunContext (deps, retry, agent…)
2. @agent.tool_plain  → le tool N'A PAS besoin du contexte
3. tools=[...]        → passage direct de fonctions ou d'instances Tool à l'Agent

Function Tools vs RAG :
Les tools sont le "R" de RAG (Retrieval-Augmented Generation) en plus
général. RAG = recherche vectorielle, tools = n'importe quelle action.

Function Tools vs Structured Output :
Les tools utilisent l'API "tools/functions" du modèle. Cette même API
sert aussi pour le structured output. Un agent peut donc avoir plusieurs
tools : certains appellent des fonctions, d'autres terminent le run
et produisent la sortie finale.
"""


# =====================================================================
# PARTIE 1 : @agent.tool_plain — tool SANS contexte
# =====================================================================

# Un jeu de dé simple. Le tool n'a besoin d'aucune dépendance
# ni info sur le contexte d'exécution → tool_plain suffit.

agent_dice = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=str,  # Le type des deps = le nom du joueur (simple str)
    instructions=(
        "Tu es un jeu de dé. Lance le dé et vérifie si le résultat "
        "correspond à la supposition du joueur. Si oui, dis-lui qu'il a gagné. "
        "Utilise le nom du joueur dans ta réponse."
    ),
)


@agent_dice.tool_plain
def roll_dice() -> str:
    """Lance un dé à six faces et retourne le résultat."""
    result = random.randint(1, 6)
    logger.info(f"Dé lancé → {result}")
    return str(result)


# =====================================================================
# PARTIE 2 : @agent.tool — tool AVEC contexte (RunContext)
# =====================================================================

# Ce tool a besoin du RunContext pour accéder aux dépendances.
# Ici, ctx.deps contient le nom du joueur (str).
@agent_dice.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """Récupère le nom du joueur."""
    logger.info(f"Nom du joueur récupéré : {ctx.deps}")
    return ctx.deps


# =====================================================================
# PARTIE 3 : Enregistrement via le keyword `tools` de l'Agent
# =====================================================================

# Au lieu d'utiliser des décorateurs, on peut passer les tools
# directement à l'Agent via le paramètre `tools=[...]`.
# Les fonctions sont automatiquement wrappées en Tool.

def get_lucky_number() -> str:
    """Retourne un nombre porte-bonheur aléatoire entre 1 et 100."""
    number = random.randint(1, 100)
    logger.info(f"Nombre porte-bonheur → {number}")
    return str(number)


def coin_flip() -> str:
    """Lance une pièce et retourne pile ou face."""
    result = random.choice(["pile", "face"])
    logger.info(f"Pièce lancée → {result}")
    return result


agent_casino = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un croupier de casino amical. "
        "Utilise les outils disponibles pour répondre aux demandes du joueur. "
        "Sois enthousiaste et divertissant."
    ),
    tools=[get_lucky_number, coin_flip],  # Enregistrement direct
)


# =====================================================================
# PARTIE 4 : Exécution — démonstration des différentes approches
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : tool_plain + tool avec contexte ---
    logger.info("=== Jeu de dé avec Anne ===")
    result_dice = agent_dice.run_sync("Ma supposition est 4", deps="Anne")
    logger.success(f"Réponse : {result_dice.output}")

    # --- Démo 2 : tools enregistrés via keyword ---
    logger.info("=== Casino : nombre porte-bonheur ===")
    result_lucky = agent_casino.run_sync("Donne-moi un nombre porte-bonheur !")
    logger.success(f"Réponse : {result_lucky.output}")

    logger.info("=== Casino : pile ou face ===")
    result_coin = agent_casino.run_sync("Lance une pièce pour moi !")
    logger.success(f"Réponse : {result_coin.output}")
