import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
DUCKDUCKGO SEARCH TOOL — Recherche web via DuckDuckGo

Contrairement aux built-in tools (WebSearchTool, CodeExecutionTool…)
qui sont exécutés par le provider, les common tools sont des outils
natifs Pydantic AI exécutés côté application.

duckduckgo_search_tool est un tool de recherche web basé sur l'API
DuckDuckGo. Il fonctionne avec TOUS les providers (pas de dépendance
côté provider, c'est Pydantic AI qui fait la recherche).

Built-in tools vs Common tools :
- Built-in (WebSearchTool)     → exécuté par le provider (Anthropic, OpenAI…)
- Common (duckduckgo_search)   → exécuté par Pydantic AI, fonctionne partout

Avantages de DuckDuckGo :
- Gratuit, pas de clé API
- Fonctionne avec tous les providers
- Pas de tracking / respect de la vie privée

Installation :
  uv add "pydantic-ai-slim[duckduckgo]"
  # ou
  pip install "pydantic-ai-slim[duckduckgo]"
"""


# =====================================================================
# PARTIE 1 : DuckDuckGo Search basique
# =====================================================================

# duckduckgo_search_tool() retourne une fonction tool prête à l'emploi.
# On la passe dans tools=[...] (pas builtin_tools, c'est un tool classique).

agent_ddg = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    tools=[duckduckgo_search_tool()],
    instructions=(
        "Tu es un assistant avec accès à la recherche web via DuckDuckGo. "
        "Cherche sur le web pour répondre aux questions d'actualité. "
        "Réponds en français."
    ),
)


# =====================================================================
# PARTIE 2 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Recherche d'actualités ---
    logger.info("=== Recherche DuckDuckGo ===")
    result = agent_ddg.run_sync(
        "Quelles sont les dernières nouvelles sur l'intelligence artificielle ?"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Recherche factuelle ---
    logger.info("=== Recherche factuelle ===")
    result = agent_ddg.run_sync(
        "Quels sont les 5 films d'animation les plus rentables en 2025 ?"
    )
    logger.success(f"Réponse : {result.output}")
