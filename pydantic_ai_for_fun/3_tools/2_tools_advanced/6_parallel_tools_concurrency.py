import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
PARALLEL TOOL CALLS & CONCURRENCY

Quand le LLM retourne plusieurs tool calls dans une seule réponse,
Pydantic AI les exécute EN PARALLÈLE par défaut (asyncio.create_task).

Règles de performance :
- Tools async   → exécutés sur l'event loop (concurrent, léger)
- Tools sync    → offloadés dans des threads (via anyio.to_thread)
- Préférer async sauf pour du blocking I/O ou du CPU-bound (numpy…)

Forcer l'exécution séquentielle :
- sequential=True sur un tool spécifique
- agent.parallel_tool_call_execution_mode('sequential') sur tout l'agent

Thread executor (prod / FastAPI) :
Par défaut, les threads pour les tools sync sont créés à la volée.
En prod, ça peut faire exploser la mémoire. Solution :
- ThreadExecutor(executor) en capability par agent
- Agent.using_thread_executor(executor) en context manager global

Limiter les tool calls :
UsageLimits(tool_calls_limit=N) permet de plafonner le nombre
d'appels de tools dans un run (évite les boucles infinies).
Les output tools (structured output) ne sont PAS comptés.
"""


# =====================================================================
# PARTIE 1 : Tools async exécutés en parallèle
# =====================================================================

# Quand le LLM demande la météo ET les news en même temps,
# les deux tools tournent en parallèle → temps total = max(t1, t2)

agent_parallel = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant qui donne la météo et les actualités. "
        "Appelle les deux outils en même temps pour être plus rapide."
    ),
)


@agent_parallel.tool_plain
async def get_weather(city: str) -> str:
    """Récupère la météo d'une ville."""
    logger.info(f"[météo] Début pour {city}...")
    await asyncio.sleep(2)  # Simule un appel API de 2s
    logger.info(f"[météo] Fin pour {city}")
    return f"Météo à {city} : 22°C, ensoleillé."


@agent_parallel.tool_plain
async def get_news(topic: str) -> str:
    """Récupère les dernières actualités sur un sujet."""
    logger.info(f"[news] Début pour {topic}...")
    await asyncio.sleep(2)  # Simule un appel API de 2s
    logger.info(f"[news] Fin pour {topic}")
    return f"Actu {topic} : nouvelle découverte majeure annoncée aujourd'hui."


# =====================================================================
# PARTIE 2 : Forcer l'exécution séquentielle
# =====================================================================

# Parfois un tool modifie un état partagé et ne doit PAS tourner
# en parallèle avec d'autres. On utilise sequential=True.

counter = 0  # État partagé (simulé)

agent_sequential = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant compteur. "
        "Utilise les outils pour incrémenter et lire le compteur."
    ),
)


@agent_sequential.tool_plain(sequential=True)
async def increment_counter() -> str:
    """Incrémente le compteur global de 1. Exécution séquentielle obligatoire."""
    global counter
    logger.info(f"[compteur] Avant : {counter}")
    await asyncio.sleep(0.5)  # Simule un traitement
    counter += 1
    logger.info(f"[compteur] Après : {counter}")
    return f"Compteur incrémenté → {counter}"


@agent_sequential.tool_plain
async def read_counter() -> str:
    """Lit la valeur actuelle du compteur."""
    return f"Compteur = {counter}"


# =====================================================================
# PARTIE 3 : Context manager pour forcer le séquentiel sur tout l'agent
# =====================================================================

# Alternative au sequential=True par tool : on peut forcer le mode
# séquentiel pour TOUS les tools d'un agent via un context manager.

# with agent_parallel.parallel_tool_call_execution_mode('sequential'):
#     result = agent_parallel.run_sync("Météo à Lyon et news tech")
#     # → get_weather puis get_news, PAS en parallèle


# =====================================================================
# PARTIE 4 : Limiter le nombre de tool calls (UsageLimits)
# =====================================================================

# UsageLimits(tool_calls_limit=N) empêche l'agent de faire plus de
# N appels de tools dans un run. Utile pour éviter les boucles.
# Note : les output tools (structured output) ne sont pas comptés.

agent_limited = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant curieux qui aime utiliser ses outils. "
        "Réponds aux questions en utilisant les outils disponibles."
    ),
)


@agent_limited.tool_plain
def ping(message: str) -> str:
    """Répond pong avec le message."""
    logger.info(f"[ping] {message}")
    return f"pong: {message}"


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Exécution parallèle (les deux tools ~2s au total, pas ~4s) ---
    logger.info("=== Parallel : météo + news en même temps ===")
    result = agent_parallel.run_sync(
        "Donne-moi la météo à Paris et les news sur l'IA"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Exécution séquentielle (increment est protégé) ---
    logger.info("=== Sequential : incrémenter le compteur ===")
    result = agent_sequential.run_sync("Incrémente le compteur deux fois puis lis-le")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Limiter les tool calls ---
    logger.info("=== Limité à 2 tool calls max ===")
    try:
        result = agent_limited.run_sync(
            "Fais 5 pings différents",
            usage_limits=UsageLimits(tool_calls_limit=2),
        )
        logger.success(f"Réponse : {result.output}")
    except Exception as e:
        logger.error(f"Limite atteinte : {e}")
