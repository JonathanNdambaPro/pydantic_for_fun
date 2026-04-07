import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, WebFetchTool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WEB FETCH TOOL — Récupérer le contenu d'une URL

WebFetchTool permet au LLM de récupérer le contenu d'une page web
et de l'intégrer dans son contexte. Contrairement à WebSearchTool
qui cherche sur le web, WebFetchTool lit une URL spécifique.

Cas d'usage :
- Lire une documentation en ligne
- Extraire le contenu d'un article
- Comparer plusieurs pages web

Providers supportés :
- Anthropic ✅ (support complet, tous les paramètres)
- Google    ✅ (pas de paramètres, limites : 20 URLs/requête, 34 Mo/URL)
- xAI       ❌ (web browsing intégré dans WebSearchTool)
- OpenAI    ❌
- Groq      ❌

Paramètres de configuration (Anthropic uniquement) :
- allowed_domains    → domaines autorisés uniquement
- blocked_domains    → domaines à exclure
- max_uses           → nombre max de fetches par run
- enable_citations   → inclure les citations dans la réponse
- max_content_tokens → limite de tokens par page récupérée

Note : avec Anthropic, on ne peut PAS utiliser blocked_domains ET
allowed_domains en même temps (comme pour WebSearchTool).

Alternative model-agnostic : la capability WebFetch utilise le tool
natif quand supporté, et tombe sur une implémentation locale sinon.
"""


# =====================================================================
# PARTIE 1 : WebFetchTool basique
# =====================================================================

# Le LLM récupère le contenu d'une URL et l'utilise pour répondre.

agent_fetch = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[WebFetchTool()],
    instructions="Tu es un assistant qui lit des pages web. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Configuration avancée (domaines, citations, limites)
# =====================================================================

# On peut restreindre les domaines accessibles, activer les citations,
# et limiter le nombre de fetches par run.

agent_fetch_configured = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[
        WebFetchTool(
            allowed_domains=['ai.pydantic.dev', 'docs.pydantic.dev'],
            max_uses=10,
            enable_citations=True,
            max_content_tokens=50000,
        )
    ],
    instructions=(
        "Tu es un assistant documentation Pydantic. "
        "Tu ne peux lire que les sites Pydantic. "
        "Réponds en français avec des citations."
    ),
)


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Fetch basique ---
    logger.info("=== Fetch d'une page web ===")
    result = agent_fetch.run_sync(
        "C'est quoi ce site ? https://ai.pydantic.dev"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Fetch configuré (domaines restreints + citations) ---
    logger.info("=== Comparaison de documentations ===")
    result = agent_fetch_configured.run_sync(
        "Compare les documentations sur https://ai.pydantic.dev "
        "et https://docs.pydantic.dev"
    )
    logger.success(f"Réponse : {result.output}")
