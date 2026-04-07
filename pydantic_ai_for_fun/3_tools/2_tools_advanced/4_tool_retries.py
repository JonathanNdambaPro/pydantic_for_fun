
import asyncio
import random

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRetry

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TOOL EXECUTION & RETRIES — Validation et mécanisme de retry

Quand le modèle appelle un tool, voici ce qui se passe :

1. Les arguments fournis par le LLM sont validés par Pydantic
   (types, champs requis, contraintes…).

2. Si la validation échoue → Pydantic lève une ValidationError.
   Pydantic AI transforme ça en RetryPromptPart et le renvoie au LLM
   pour qu'il corrige ses paramètres et réessaie.

3. Si la validation passe mais que la logique du tool détecte un
   problème → on peut lever ModelRetry manuellement.
   Même mécanisme : le message d'erreur est renvoyé au LLM.

Les deux (ValidationError et ModelRetry) respectent le paramètre
`retries` configuré sur le Tool ou l'Agent.

En résumé :
- ValidationError  → retry AUTOMATIQUE (mauvais types, champs manquants…)
- ModelRetry       → retry EXPLICITE (logique métier, erreur transitoire…)
"""


# =====================================================================
# PARTIE 1 : ModelRetry — retry explicite depuis la logique du tool
# =====================================================================

# Cas d'usage : un outil de recherche qui refuse certaines requêtes
# et guide le LLM vers une meilleure formulation.

agent_search = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant de recherche. "
        "Utilise l'outil de recherche pour trouver des informations. "
        "Si la recherche échoue, reformule ta requête."
    ),
    retries=3,  # Le LLM a droit à 3 tentatives max
)


@agent_search.tool_plain
def search(query: str) -> str:
    """Recherche des informations. La requête doit faire au moins 3 caractères."""
    logger.info(f"Recherche reçue : '{query}'")

    if len(query) < 3:
        # On guide le LLM : "ta requête est trop courte, réessaie"
        raise ModelRetry(
            f"La requête '{query}' est trop courte (min 3 caractères). "
            "Reformule avec plus de détails."
        )

    if query.lower() == "test":
        raise ModelRetry(
            "Le mot 'test' n'est pas une requête valide. "
            "Pose une vraie question."
        )

    logger.success(f"Recherche OK pour : '{query}'")
    return f"Résultats pour '{query}' : 42 documents trouvés."


# =====================================================================
# PARTIE 2 : Simulation d'erreur transitoire avec ModelRetry
# =====================================================================

# Cas d'usage : un outil qui appelle une API externe instable.
# Si l'appel échoue, on lève ModelRetry pour que le LLM réessaie.

agent_api = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant météo. "
        "Utilise l'outil pour obtenir la météo d'une ville."
    ),
    retries=3,
)


@agent_api.tool_plain
def get_weather(city: str) -> str:
    """Récupère la météo pour une ville donnée."""
    logger.info(f"Appel API météo pour : {city}")

    # Simulation d'une API instable (1 chance sur 3 d'échouer)
    if random.random() < 0.33:
        logger.warning("API météo indisponible, retry...")
        raise ModelRetry(
            "L'API météo est temporairement indisponible. "
            "Réessaie avec la même ville."
        )

    return f"Météo à {city} : 22°C, ensoleillé."


# =====================================================================
# PARTIE 3 : Tool Timeout — limiter le temps d'exécution
# =====================================================================

# On peut définir un timeout au niveau de l'agent (par défaut pour
# tous les tools) ou au niveau de chaque tool individuellement.
# Si un tool dépasse son timeout → considéré comme un échec →
# le modèle reçoit "Timed out after X seconds." et ça compte
# comme un retry.

agent_timeout = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant. Utilise les outils disponibles.",
    tool_timeout=30,  # Timeout par défaut : 30s pour tous les tools
    retries=2,
)


@agent_timeout.tool_plain
async def slow_tool() -> str:
    """Ce tool utilise le timeout par défaut de l'agent (30 secondes)."""
    logger.info("slow_tool : traitement en cours...")
    await asyncio.sleep(2)  # Simule un traitement long (mais < 30s → OK)
    return "Traitement terminé avec succès."


@agent_timeout.tool_plain(timeout=5)
async def fast_tool() -> str:
    """Ce tool a son propre timeout (5 secondes) qui override le défaut."""
    logger.info("fast_tool : traitement rapide...")
    await asyncio.sleep(1)  # Rapide → OK
    return "Résultat rapide obtenu."


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : ModelRetry sur requête invalide ---
    logger.info("=== Recherche normale ===")
    result = agent_search.run_sync("Explique-moi le machine learning")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : API avec erreur transitoire ---
    logger.info("=== Météo (peut retry si API instable) ===")
    result = agent_api.run_sync("Quelle est la météo à Paris ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Tools avec timeout ---
    logger.info("=== Tool avec timeout agent (30s) ===")
    result = agent_timeout.run_sync("Utilise le slow_tool")
    logger.success(f"Réponse : {result.output}")

    logger.info("=== Tool avec timeout custom (5s) ===")
    result = agent_timeout.run_sync("Utilise le fast_tool")
    logger.success(f"Réponse : {result.output}")
