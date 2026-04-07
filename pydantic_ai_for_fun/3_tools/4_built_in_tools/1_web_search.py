import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, WebSearchTool, WebSearchUserLocation

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WEB SEARCH TOOL — Recherche web native via le provider

Les built-in tools sont des outils natifs fournis par les providers LLM.
Contrairement aux tools custom (@agent.tool), ils sont exécutés
directement par l'infrastructure du provider, pas par Pydantic AI.

WebSearchTool permet à l'agent de chercher sur le web en temps réel.
C'est idéal pour les questions nécessitant des données à jour.

Providers supportés :
- Anthropic        ✅ (support complet)
- OpenAI Responses ✅ (support complet, nécessite openai-responses)
- Google           ✅ (pas de paramètres, limitations avec function tools)
- xAI              ✅ (blocked_domains, allowed_domains)
- Groq             ✅ (support limité, nécessite compound models)
- OpenRouter       ✅ (via plugins)

Paramètres de configuration :
- search_context_size → quantité de contexte ('low', 'medium', 'high')
- user_location       → localisation de l'utilisateur (ville, pays…)
- blocked_domains     → sites à exclure des résultats
- allowed_domains     → sites autorisés uniquement
- max_uses            → limite d'utilisation (Anthropic uniquement)

Note : avec Anthropic, on ne peut PAS utiliser blocked_domains ET
allowed_domains en même temps.

Alternative model-agnostic : la capability WebSearch utilise le tool
natif quand supporté, et tombe sur une implémentation locale sinon.
"""


# =====================================================================
# PARTIE 1 : WebSearchTool basique
# =====================================================================

# Le cas le plus simple : on ajoute WebSearchTool() dans builtin_tools.
# Le provider (ici Anthropic) exécute la recherche côté serveur.

agent_search = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[WebSearchTool()],
    instructions="Tu es un assistant avec accès au web. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Configuration avancée (localisation, filtres)
# =====================================================================

# On peut personnaliser la recherche : localisation de l'utilisateur,
# domaines bloqués, limite d'utilisation…

agent_search_configured = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[
        WebSearchTool(
            user_location=WebSearchUserLocation(
                city='Paris',
                country='FR',
                region='Île-de-France',
                timezone='Europe/Paris',
            ),
            blocked_domains=['example.com', 'spam-site.net'],
            max_uses=5,  # Anthropic uniquement : limite à 5 recherches par run
        )
    ],
    instructions=(
        "Tu es un assistant localisé à Paris. "
        "Utilise la recherche web pour répondre avec des infos à jour. "
        "Réponds en français."
    ),
)


# =====================================================================
# PARTIE 3 : Configuration dynamique selon le contexte
# =====================================================================

# Parfois on veut configurer le tool selon les deps du run.
# On passe une fonction au lieu d'une instance de WebSearchTool.
# Si la fonction retourne None → le tool est désactivé pour ce run.


async def prepared_web_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    """Configure le WebSearchTool dynamiquement selon les deps."""
    location = ctx.deps.get("location")
    if not location:
        logger.info("Pas de localisation → WebSearchTool désactivé")
        return None

    logger.info(f"WebSearchTool configuré pour : {location}")
    return WebSearchTool(
        user_location=WebSearchUserLocation(city=location),
    )


agent_search_dynamic = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[prepared_web_search],
    deps_type=dict,
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Recherche web basique ---
    logger.info("=== Recherche web basique ===")
    result = agent_search.run_sync(
        "Quelle est la plus grosse actu IA cette semaine ?"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Recherche localisée à Paris ---
    logger.info("=== Recherche localisée (Paris) ===")
    result = agent_search_configured.run_sync(
        "Quelle heure est-il et quel temps fait-il ?"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Dynamique avec localisation ---
    logger.info("=== Recherche dynamique (Lyon) ===")
    result = agent_search_dynamic.run_sync(
        "Quels événements ont lieu ce weekend ?",
        deps={"location": "Lyon"},
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : Dynamique sans localisation (tool désactivé) ---
    logger.info("=== Sans localisation (tool désactivé) ===")
    result = agent_search_dynamic.run_sync(
        "Quelle est la capitale de la France ?",
        deps={"location": None},
    )
    logger.success(f"Réponse : {result.output}")
