from datetime import datetime

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, CombinedToolset, FunctionToolset, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TOOLSETS — Collections de tools réutilisables et composables

Un toolset = un groupe de tools qu'on peut :
- Réutiliser sur plusieurs agents
- Swapper au runtime ou en test
- Composer (combiner, filtrer, renommer, préfixer…)

Quand utiliser un toolset plutôt que @agent.tool :
- Quand on veut partager le même pack de tools entre agents
- Quand les tools viennent d'une source externe (MCP, LangChain…)
- Quand on veut override les tools en test
- Quand on veut filtrer/modifier les tools dynamiquement

Pour des tools simples définis dans ton code → @agent.tool suffit.

FunctionToolset est le toolset le plus courant : il regroupe des
fonctions Python locales, comme @agent.tool mais en réutilisable.
"""


# =====================================================================
# PARTIE 1 : FunctionToolset — créer un pack de tools réutilisable
# =====================================================================

# On crée deux toolsets indépendants qu'on pourra brancher
# sur n'importe quel agent.

weather_toolset = FunctionToolset(
    instructions="Utilise les outils météo pour les questions sur le temps."
)


@weather_toolset.tool_plain
def get_temperature(city: str) -> str:
    """Retourne la température d'une ville."""
    logger.info(f"[météo] Température pour {city}")
    return f"{city} : 22°C"


@weather_toolset.tool_plain
def get_conditions(city: str) -> str:
    """Retourne les conditions météo d'une ville."""
    logger.info(f"[météo] Conditions pour {city}")
    return f"{city} : ensoleillé, vent léger"


datetime_toolset = FunctionToolset(
    instructions="Utilise les outils date/heure pour les questions temporelles."
)
datetime_toolset.add_function(lambda: str(datetime.now()), name="now")


# =====================================================================
# PARTIE 2 : Réutiliser les toolsets sur plusieurs agents
# =====================================================================

# Les deux agents partagent le même weather_toolset.
# Si on modifie le toolset, les deux agents en bénéficient.

agent_meteo = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant météo.",
    toolsets=[weather_toolset],
)

agent_complet = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant polyvalent.",
    toolsets=[weather_toolset, datetime_toolset],
)


# =====================================================================
# PARTIE 3 : Ajouter des toolsets au runtime (par run)
# =====================================================================

# On peut ajouter des toolsets supplémentaires pour un run spécifique
# sans les enregistrer sur l'agent de façon permanente.

extra_toolset = FunctionToolset()


@extra_toolset.tool_plain
def get_joke() -> str:
    """Raconte une blague."""
    return "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils tomberaient dans le bateau."


# agent_meteo.run_sync("Raconte une blague", toolsets=[extra_toolset])
# → agent_meteo a weather_toolset + extra_toolset pour CE run uniquement


# =====================================================================
# PARTIE 4 : Override en test
# =====================================================================

# agent.override(toolsets=[...]) remplace TOUS les toolsets de l'agent.
# Parfait pour les tests : on injecte un mock toolset.

mock_toolset = FunctionToolset()


@mock_toolset.tool_plain
def get_temperature_mock(city: str) -> str:
    """Mock : retourne toujours la même température."""
    return f"{city} : 20°C (mock)"


# with agent_meteo.override(toolsets=[mock_toolset]):
#     result = agent_meteo.run_sync("Météo à Paris ?")
#     # → utilise get_temperature_mock au lieu de get_temperature


# =====================================================================
# PARTIE 5 : Composition — filtrer, préfixer, combiner
# =====================================================================

# .filtered() → exclure certains tools selon une condition
filtered_toolset = weather_toolset.filtered(
    lambda ctx, tool_def: "temperature" not in tool_def.name
)
# → ne garde que get_conditions, exclut get_temperature

# .prefixed() → ajouter un préfixe pour éviter les conflits de noms
combined = CombinedToolset([
    weather_toolset.prefixed("weather"),
    datetime_toolset.prefixed("datetime"),
])
# → tools nommés : weather_get_temperature, weather_get_conditions, datetime_now


# =====================================================================
# PARTIE 6 : Toolset dynamique — changer les tools selon le contexte
# =====================================================================

# @agent.toolset permet de retourner un toolset différent selon les deps.
# Réévalué à chaque étape du run.

agent_dynamic = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=str,  # "meteo" ou "datetime"
    instructions="Tu es un assistant. Utilise les outils disponibles.",
)


@agent_dynamic.toolset
def dynamic_toolset(ctx: RunContext[str]):
    """Retourne un toolset différent selon les deps."""
    if ctx.deps == "meteo":
        return weather_toolset
    else:
        return datetime_toolset


# =====================================================================
# PARTIE 7 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Agent avec un seul toolset ---
    logger.info("=== Agent météo (1 toolset) ===")
    result = agent_meteo.run_sync("Quelle est la température à Lyon ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Agent avec plusieurs toolsets (instructions combinées) ---
    logger.info("=== Agent complet (2 toolsets) ===")
    result = agent_complet.run_sync("Quelle heure est-il ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Toolset supplémentaire au runtime ---
    logger.info("=== Toolset extra au runtime ===")
    result = agent_meteo.run_sync("Raconte une blague", toolsets=[extra_toolset])
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Toolset dynamique selon les deps ---
    logger.info("=== Toolset dynamique (meteo) ===")
    result = agent_dynamic.run_sync("Température à Marseille ?", deps="meteo")
    logger.success(f"Réponse : {result.output}")

    logger.info("=== Toolset dynamique (datetime) ===")
    result = agent_dynamic.run_sync("Quelle heure est-il ?", deps="datetime")
    logger.success(f"Réponse : {result.output}")
