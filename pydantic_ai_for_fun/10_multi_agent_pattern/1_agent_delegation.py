import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, UsageLimits

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
AGENT DELEGATION — Un agent délègue du travail à un autre via un tool

La délégation d'agent = un agent appelle un autre agent depuis un tool,
puis reprend le contrôle quand l'agent délégué a fini.

5 niveaux de complexité multi-agent :
1. Single agent       → la plupart des cas
2. Agent delegation   → un agent appelle un autre via tool (CE FICHIER)
3. Programmatic hand-off → le code applicatif enchaîne les agents
4. Graph-based        → machine à états pour les cas complexes
5. Deep Agents        → agents autonomes avec planification

Points clés de la délégation :
- Les agents sont stateless et globaux → pas besoin de les mettre
  dans les dépendances
- Passer ctx.usage au run de l'agent délégué pour compter l'usage
  dans le total du parent
- On peut utiliser des modèles différents pour chaque agent
- UsageLimits pour éviter les coûts imprévus
"""


# =====================================================================
# PARTIE 1 : Délégation simple — Agent de sélection + Agent générateur
# =====================================================================

# L'agent de sélection utilise un tool qui appelle l'agent générateur.
# Flux : selection_agent → joke_factory (tool) → generation_agent → retour

# Agent générateur : produit une liste de blagues
joke_generation_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=list[str],
    instructions="Génère des blagues courtes et drôles en français.",
)

# Agent de sélection : choisit la meilleure blague
joke_selection_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Utilise l'outil `joke_factory` pour générer des blagues, "
        "puis choisis la meilleure. Tu dois retourner une seule blague."
    ),
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[None], count: int) -> list[str]:
    """Génère un certain nombre de blagues via l'agent générateur."""
    logger.info(f"[delegation] Demande de {count} blagues à l'agent générateur")
    r = await joke_generation_agent.run(
        f"Génère {count} blagues courtes et drôles.",
        usage=ctx.usage,  # L'usage compte dans le total du parent
    )
    logger.info(f"[delegation] {len(r.output)} blagues reçues")
    return r.output


# =====================================================================
# PARTIE 2 : Délégation avec UsageLimits
# =====================================================================

# On limite le nombre de requêtes et tokens pour éviter les coûts
# imprévus ou les boucles infinies.

# Agent de recherche : cherche des infos
research_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=str,
    instructions="Réponds de manière factuelle et concise en français.",
)

# Agent principal : utilise le researcher
main_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant qui utilise l'outil `research` pour "
        "trouver des informations, puis tu les synthétises."
    ),
)


@main_agent.tool
async def research(ctx: RunContext[None], question: str) -> str:
    """Délègue une recherche à l'agent spécialisé."""
    logger.info(f"[research] Question déléguée : {question}")
    r = await research_agent.run(
        question,
        usage=ctx.usage,
    )
    return r.output


# =====================================================================
# PARTIE 3 : Délégation avec modèles différents
# =====================================================================

# Chaque agent peut utiliser un modèle différent.
# L'usage total est suivi, mais le coût monétaire exact
# ne peut pas être calculé depuis result.usage().

# Agent rapide (modèle léger) pour le brouillon
draft_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=str,
    instructions="Écris un brouillon rapide et concis en français.",
)

# Agent principal (modèle puissant) pour la synthèse finale
editor_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un éditeur. Utilise l'outil `get_draft` pour obtenir "
        "un brouillon, puis améliore-le et retourne la version finale."
    ),
)


@editor_agent.tool
async def get_draft(ctx: RunContext[None], topic: str) -> str:
    """Obtient un brouillon de l'agent rédacteur."""
    logger.info(f"[draft] Demande de brouillon sur : {topic}")
    r = await draft_agent.run(
        f"Écris un court paragraphe sur : {topic}",
        usage=ctx.usage,
    )
    return r.output


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Délégation simple (blagues) ---
    logger.info("=== Délégation simple : sélection de blagues ===")
    result = joke_selection_agent.run_sync(
        "Raconte-moi une blague.",
        usage_limits=UsageLimits(request_limit=5),
    )
    logger.success(f"Meilleure blague : {result.output}")
    logger.info(f"Usage total : {result.usage()}")

    # --- Démo 2 : Délégation avec limites ---
    logger.info("=== Délégation avec UsageLimits ===")
    result = main_agent.run_sync(
        "Quels sont les 3 langages de programmation les plus populaires ?",
        usage_limits=UsageLimits(request_limit=5, total_tokens_limit=2000),
    )
    logger.success(f"Synthèse : {result.output}")
    logger.info(f"Usage total : {result.usage()}")

    # --- Démo 3 : Délégation multi-modèles ---
    logger.info("=== Délégation multi-modèles (draft → edit) ===")
    result = editor_agent.run_sync(
        "Écris un paragraphe sur l'intelligence artificielle.",
        usage_limits=UsageLimits(request_limit=5),
    )
    logger.success(f"Version finale : {result.output}")
    logger.info(f"Usage total : {result.usage()}")
