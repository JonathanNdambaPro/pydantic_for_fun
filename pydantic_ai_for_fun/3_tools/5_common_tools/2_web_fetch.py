import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WEB FETCH TOOL (Common) — Récupérer une page web en markdown

web_fetch_tool est la version common tool de WebFetchTool (built-in).
Il récupère le contenu d'une URL et le convertit en markdown.

Différence avec le built-in WebFetchTool :
- Built-in (WebFetchTool)    → exécuté par le provider (Anthropic, Google)
- Common (web_fetch_tool)    → exécuté par Pydantic AI, fonctionne partout

Avantages :
- Fonctionne avec TOUS les providers (même ceux sans built-in)
- Protection SSRF intégrée (empêche les attaques server-side request forgery)
- Conversion automatique en markdown

La capability WebFetch utilise automatiquement ce common tool comme
fallback quand le provider ne supporte pas le built-in WebFetchTool.
→ Pas besoin de choisir manuellement entre les deux.

Installation :
  uv add "pydantic-ai-slim[web-fetch]"
  # ou
  pip install "pydantic-ai-slim[web-fetch]"
"""


# =====================================================================
# PARTIE 1 : web_fetch_tool basique
# =====================================================================

# web_fetch_tool() retourne un tool prêt à l'emploi.
# Le LLM peut récupérer le contenu de n'importe quelle URL.

agent_fetch = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    tools=[web_fetch_tool()],
    instructions=(
        "Tu es un assistant qui lit des pages web et résume leur contenu. "
        "Réponds en français."
    ),
)


# =====================================================================
# PARTIE 2 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Lire une page web ---
    logger.info("=== Fetch d'une page web ===")
    result = agent_fetch.run_sync("C'est quoi ce site ? https://ai.pydantic.dev")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Résumer un article ---
    logger.info("=== Résumé d'une page ===")
    result = agent_fetch.run_sync(
        "Résume le contenu de https://docs.pydantic.dev/latest/"
    )
    logger.success(f"Réponse : {result.output}")
