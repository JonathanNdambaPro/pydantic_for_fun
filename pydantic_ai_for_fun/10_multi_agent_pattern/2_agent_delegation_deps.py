from dataclasses import dataclass

import httpx
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, UsageLimits

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
AGENT DELEGATION DEPS — Délégation avec dépendances partagées

Quand un agent délègue à un autre, l'agent délégué a généralement
besoin des mêmes dépendances (ou d'un sous-ensemble).

Règles :
- L'agent délégué doit avoir le même deps_type ou un sous-ensemble
- On passe ctx.deps au run de l'agent délégué
- Éviter d'initialiser des dépendances dans un tool (lent)
- Réutiliser les connexions du parent (httpx, DB, etc.)

Pattern :
1. Définir un deps_type commun (dataclass)
2. Les deux agents l'utilisent
3. Le tool passe ctx.deps et ctx.usage
"""


# =====================================================================
# PARTIE 1 : Dépendances partagées — dataclass commune
# =====================================================================

# Les deux agents partagent le même type de dépendances.


@dataclass
class AppDeps:
    """Dépendances de l'application : client HTTP + clé API."""

    http_client: httpx.AsyncClient
    api_key: str


# =====================================================================
# PARTIE 2 : Agent délégué — Générateur de blagues avec API
# =====================================================================

# Cet agent utilise un tool qui fait des requêtes HTTP via les deps.

joke_generation_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=AppDeps,
    output_type=list[str],
    instructions=(
        "Utilise l'outil 'get_jokes' pour obtenir des blagues, "
        "puis extrais chaque blague dans une liste."
    ),
)


@joke_generation_agent.tool
async def get_jokes(ctx: RunContext[AppDeps], count: int) -> str:
    """Récupère des blagues depuis une API (simulée)."""
    logger.info(f"[get_jokes] Requête API pour {count} blagues")
    # En réalité, on ferait une vraie requête HTTP :
    # response = await ctx.deps.http_client.get(
    #     'https://api.jokes.com/jokes',
    #     params={'count': count},
    #     headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    # )
    # return response.text

    # Simulation pour la démo
    jokes = [f"Blague #{i+1} : Pourquoi le {i+1} est drôle ? Parce que." for i in range(count)]
    return "\n".join(jokes)


# =====================================================================
# PARTIE 3 : Agent parent — Sélecteur de blagues
# =====================================================================

# L'agent parent utilise le même deps_type et passe ctx.deps au délégué.

joke_selection_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=AppDeps,
    instructions=(
        "Utilise l'outil `joke_factory` pour générer des blagues, "
        "puis choisis la meilleure. Retourne une seule blague."
    ),
)


@joke_selection_agent.tool
async def joke_factory(ctx: RunContext[AppDeps], count: int) -> list[str]:
    """Délègue la génération de blagues à l'agent spécialisé."""
    logger.info(f"[joke_factory] Délégation pour {count} blagues")
    r = await joke_generation_agent.run(
        f"Génère {count} blagues.",
        deps=ctx.deps,   # Passer les dépendances du parent
        usage=ctx.usage,  # Compter l'usage dans le total
    )
    return r.output


# =====================================================================
# PARTIE 4 : Pattern avec sous-ensemble de deps
# =====================================================================

# L'agent délégué peut avoir un deps_type plus simple.


@dataclass
class FullDeps:
    """Dépendances complètes du parent."""

    http_client: httpx.AsyncClient
    api_key: str
    db_connection: str  # Pas besoin pour l'agent délégué


@dataclass
class ApiDeps:
    """Sous-ensemble de dépendances pour l'agent délégué."""

    http_client: httpx.AsyncClient
    api_key: str


# Agent délégué avec des deps réduites
api_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=ApiDeps,
    output_type=str,
    instructions="Utilise les outils pour faire des requêtes API.",
)

# Agent parent avec des deps complètes
orchestrator_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=FullDeps,
    instructions="Tu es un orchestrateur. Utilise les outils disponibles.",
)


@orchestrator_agent.tool
async def call_api(ctx: RunContext[FullDeps], query: str) -> str:
    """Délègue un appel API à l'agent spécialisé."""
    # On extrait le sous-ensemble de deps nécessaire
    api_deps = ApiDeps(
        http_client=ctx.deps.http_client,
        api_key=ctx.deps.api_key,
    )
    r = await api_agent.run(query, deps=api_deps, usage=ctx.usage)
    return r.output


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        async with httpx.AsyncClient() as client:
            # --- Démo 1 : Délégation avec deps partagées ---
            logger.info("=== Délégation avec dépendances partagées ===")
            deps = AppDeps(http_client=client, api_key="demo-key-123")

            result = await joke_selection_agent.run(
                "Raconte-moi une blague.",
                deps=deps,
                usage_limits=UsageLimits(request_limit=8),
            )
            logger.success(f"Meilleure blague : {result.output}")
            logger.info(f"Usage total : {result.usage()}")

            # --- Démo 2 : Délégation avec sous-ensemble de deps ---
            logger.info("=== Délégation avec sous-ensemble de deps ===")
            full_deps = FullDeps(
                http_client=client,
                api_key="demo-key-456",
                db_connection="postgresql://localhost/mydb",
            )

            result = await orchestrator_agent.run(
                "Fais une recherche sur Python",
                deps=full_deps,
                usage_limits=UsageLimits(request_limit=5),
            )
            logger.success(f"Résultat : {result.output}")
            logger.info(f"Usage total : {result.usage()}")

    asyncio.run(main())
