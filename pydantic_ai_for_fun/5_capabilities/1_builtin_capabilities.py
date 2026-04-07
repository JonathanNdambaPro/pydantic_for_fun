import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks, Thinking, WebFetch, WebSearch

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CAPABILITIES — Unités de comportement réutilisables et composables

Une capability = un bundle qui peut fournir :
- Des tools (via toolsets ou builtin tools)
- Des lifecycle hooks (intercepter requêtes, tools, runs…)
- Des instructions (statiques ou dynamiques)
- Des model settings (statiques ou par étape)

Au lieu de passer plein d'arguments séparés à l'Agent, on regroupe
le comportement dans des capabilities réutilisables.

Capabilities built-in fournies par Pydantic AI :
- Thinking        → active le mode réflexion du modèle
- Hooks           → enregistre des hooks via décorateurs (le plus simple)
- WebSearch       → recherche web (builtin si supporté, fallback local)
- WebFetch        → fetch d'URL (builtin si supporté, fallback local)
- ImageGeneration → génération d'images (builtin + fallback)
- MCP             → connexion MCP (builtin si supporté, connexion directe sinon)
- PrepareTools    → filtre/modifie les tools visibles par étape
- PrefixTools     → préfixe les noms de tools d'une capability
- ThreadExecutor  → thread pool custom pour les tools sync en prod

Composition :
Quand on passe plusieurs capabilities, elles sont combinées :
- before_* → dans l'ordre (cap1 → cap2 → cap3)
- after_*  → en ordre inverse (cap3 → cap2 → cap1)
- wrap_*   → en middleware (cap1 wrappe cap2 wrappe cap3)
"""


# =====================================================================
# PARTIE 1 : Thinking — mode réflexion
# =====================================================================

# Active le "thinking/reasoning" du modèle. Le plus simple pour
# activer la réflexion cross-provider.

agent_thinking = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[Thinking(effort='high')],
    instructions="Tu es un assistant de recherche. Sois exhaustif.",
)


# =====================================================================
# PARTIE 2 : Hooks — lifecycle hooks via décorateurs
# =====================================================================

# Le moyen le plus simple d'intercepter des événements sans
# sous-classer AbstractCapability. On décore des fonctions.

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(
    ctx: RunContext[None], request_context: ModelRequestContext
) -> ModelRequestContext:
    """Logge chaque requête envoyée au modèle."""
    agent_name = ctx.agent.name if ctx.agent else "unknown"
    logger.info(
        f"[{agent_name}] Envoi de {len(request_context.messages)} messages"
    )
    return request_context


agent_hooked = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    name="agent_hooked",
    capabilities=[hooks],
    instructions="Tu es un assistant.",
)


# =====================================================================
# PARTIE 3 : Provider-adaptive tools (WebSearch, WebFetch)
# =====================================================================

# WebSearch et WebFetch utilisent automatiquement le builtin du
# provider quand supporté, sinon un fallback local (DuckDuckGo,
# markdownify). Ton agent fonctionne partout sans changer le code.

agent_adaptive = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[
        WebSearch(),   # Builtin Anthropic, fallback DuckDuckGo
        WebFetch(),    # Builtin Anthropic, fallback markdownify
    ],
    instructions="Tu es un assistant avec accès au web. Réponds en français.",
)

# Forcer builtin uniquement (erreur si le provider ne supporte pas) :
# WebSearch(local=False)

# Forcer local uniquement (ne jamais utiliser le builtin) :
# WebSearch(builtin=False)

# Contraintes de domaine (nécessite le builtin pour WebSearch) :
# WebSearch(allowed_domains=['example.com'])
# WebFetch(allowed_domains=['example.com'])  # OK en local aussi


# =====================================================================
# PARTIE 4 : Combiner plusieurs capabilities
# =====================================================================

# Les capabilities se composent naturellement.
# Instructions, settings, tools sont fusionnés automatiquement.

agent_complet = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
        hooks,  # Réutilise les mêmes hooks
    ],
    instructions="Tu es un assistant de recherche.",
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Thinking ---
    logger.info("=== Thinking (effort=high) ===")
    result = agent_thinking.run_sync("Explique la relativité générale simplement")
    logger.success(f"Réponse : {result.output[:100]}...")

    # --- Démo 2 : Hooks (logging automatique) ---
    logger.info("=== Hooks (logging) ===")
    result = agent_hooked.run_sync("Bonjour")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Provider-adaptive WebSearch ---
    logger.info("=== WebSearch adaptatif ===")
    result = agent_adaptive.run_sync("Quelle est la dernière actu IA ?")
    logger.success(f"Réponse : {result.output[:100]}...")
