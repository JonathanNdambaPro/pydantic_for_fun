import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import CodeMode

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CODE MODE — Exécution sandboxée de code pour appels d'outils groupés

Code Mode remplace les appels d'outils individuels par un unique outil
run_code qui exécute du Python dans un sandbox Monty. Le modèle écrit
du code qui appelle plusieurs outils avec boucles, conditions, variables
et asyncio.gather — le tout en un seul appel.

Avantages par rapport au tool calling classique :
- 1 appel modèle pour N outils (au lieu de 1 par outil)
- Parallélisme natif via asyncio.gather
- Filtrage, transformation, agrégation côté code
- Historique de conversation plus compact

Points clés :
- Sandbox Monty : pas de classes, pas d'imports tiers
  (stdlib autorisé : sys, typing, asyncio, math, json, re, datetime, os, pathlib)
- L'état REPL persiste entre les appels run_code dans un même run
- La dernière expression est capturée comme valeur de retour
- Observabilité : les appels imbriqués produisent leurs propres spans
"""


# =====================================================================
# PARTIE 1 : Code Mode basique — tous les outils sandboxés
# =====================================================================

# Par défaut, CodeMode(tools='all') encapsule TOUS les outils
# dans le sandbox run_code. Le modèle écrit du Python qui appelle
# les outils comme des fonctions async.


def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}


def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


agent_basic = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode()],
    instructions="Tu es un assistant météo. Réponds en français.",
)

# On enregistre les outils sur l'agent
agent_basic.tool_plain(get_weather)
agent_basic.tool_plain(convert_temp)

# Le modèle peut écrire du code comme :
# paris, tokyo = await asyncio.gather(
#     get_weather(city='Paris'),
#     get_weather(city='Tokyo'),
# )
# paris_c = await convert_temp(fahrenheit=paris['temp_f'])
# tokyo_c = await convert_temp(fahrenheit=tokyo['temp_f'])
# {'paris': paris_c, 'tokyo': tokyo_c}


# =====================================================================
# PARTIE 2 : Sélection des outils — par nom
# =====================================================================

# On peut choisir quels outils passent par le sandbox.
# Les outils non sélectionnés restent disponibles en tool calling classique.


def search(query: str) -> list[str]:
    """Search for documents matching the query."""
    return [f"Result 1 for '{query}'", f"Result 2 for '{query}'"]


def fetch(url: str) -> str:
    """Fetch content from a URL."""
    return f"Content of {url}"


def dangerous_action(command: str) -> str:
    """Execute a dangerous action (excluded from sandbox)."""
    return f"Executed: {command}"


# Seuls search et fetch sont dans le sandbox
agent_selective = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode(tools=['search', 'fetch'])],
    instructions="Tu es un assistant de recherche.",
)

agent_selective.tool_plain(search)
agent_selective.tool_plain(fetch)
agent_selective.tool_plain(dangerous_action)


# =====================================================================
# PARTIE 3 : Sélection par prédicat (fonction)
# =====================================================================

# On peut passer une fonction lambda ou callable pour filtrer
# dynamiquement les outils à sandboxer.

agent_predicate = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode(tools=lambda ctx, td: td.name != 'dangerous_action')],
    instructions="Tu es un assistant sécurisé.",
)

agent_predicate.tool_plain(search)
agent_predicate.tool_plain(fetch)
agent_predicate.tool_plain(dangerous_action)


# =====================================================================
# PARTIE 4 : Sélection par métadonnées
# =====================================================================

# On peut taguer les toolsets avec des métadonnées et filtrer
# via un dict. Utile quand on compose plusieurs toolsets.

search_tools = FunctionToolset(tools=[search, fetch]).with_metadata(code_mode=True)

agent_metadata = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    toolsets=[search_tools],
    capabilities=[CodeMode(tools={'code_mode': True})],
    instructions="Tu es un assistant de recherche avancé.",
)


# =====================================================================
# PARTIE 5 : Valeurs de retour et état REPL
# =====================================================================

# - La dernière expression du code est capturée comme retour
#   (pas besoin de print())
# - Si du texte est imprimé : {"output": "<texte>", "result": <expr>}
# - L'état persiste entre les appels run_code dans le même run
#   (variables, imports, fonctions définies)
# - Passer restart=True dans l'appel pour réinitialiser l'état

agent_repl = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[CodeMode(max_retries=5)],
    instructions=(
        "Tu es un assistant data. Utilise le code mode pour "
        "traiter les données efficacement."
    ),
)

agent_repl.tool_plain(get_weather)
agent_repl.tool_plain(convert_temp)


# =====================================================================
# PARTIE 6 : Observabilité — inspecter les appels imbriqués
# =====================================================================

# Quand instrumenté avec Logfire/OpenTelemetry, chaque appel d'outil
# dans run_code produit son propre span. On peut aussi inspecter
# les métadonnées dans les messages de retour.

from pydantic_ai.messages import ToolReturnPart  # noqa: E402


def inspect_code_mode_calls(result) -> None:
    """Inspecte les appels d'outils effectués via run_code."""
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                tool_calls = part.metadata.get('tool_calls', {})
                tool_returns = part.metadata.get('tool_returns', {})
                logger.info(f"Outils appelés dans run_code : {list(tool_calls.keys())}")
                logger.info(f"Retours : {list(tool_returns.keys())}")


# =====================================================================
# PARTIE 7 : Agent spec (YAML/JSON)
# =====================================================================

# CodeMode fonctionne aussi avec les fichiers de spec YAML :
#
# # agent.yaml
# model: anthropic:claude-sonnet-4-6
# capabilities:
#   - CodeMode: {}
#
# # Avec arguments :
# capabilities:
#   - CodeMode:
#       tools: ['search', 'fetch']
#       max_retries: 5
#
# Chargement en Python :
# agent = Agent.from_file('agent.yaml', custom_capability_types=[CodeMode])


# =====================================================================
# PARTIE 8 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Code Mode basique (outils parallèles) ---
    logger.info("=== Code Mode basique — météo multi-villes ===")
    result = agent_basic.run_sync(
        "Quelle est la météo à Paris et Tokyo, en Celsius ?"
    )
    logger.success(f"Réponse : {result.output[:300]}")

    # Inspection des appels imbriqués
    inspect_code_mode_calls(result)

    # --- Démo 2 : Sélection par nom ---
    logger.info("=== Sélection par nom (search + fetch uniquement) ===")
    result = agent_selective.run_sync(
        "Cherche 'pydantic ai code mode' et récupère le premier résultat."
    )
    logger.success(f"Réponse : {result.output[:300]}")

    # --- Démo 3 : Sélection par métadonnées ---
    logger.info("=== Sélection par métadonnées ===")
    result = agent_metadata.run_sync(
        "Fais une recherche sur 'asyncio gather' et récupère les détails."
    )
    logger.success(f"Réponse : {result.output[:300]}")

    # --- Démo 4 : État REPL persistant ---
    logger.info("=== État REPL persistant (max_retries=5) ===")
    result = agent_repl.run_sync(
        "Récupère la météo de 5 villes françaises, convertis en Celsius, "
        "et donne-moi la moyenne."
    )
    logger.success(f"Réponse : {result.output[:300]}")
