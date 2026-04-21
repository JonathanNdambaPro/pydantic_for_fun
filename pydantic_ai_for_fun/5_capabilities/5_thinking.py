import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models.anthropic import AnthropicModelSettings

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
THINKING — Mode réflexion (reasoning) des modèles

Le thinking permet au modèle de "réfléchir" étape par étape avant
de donner sa réponse finale. Utile pour les tâches complexes :
raisonnement logique, maths, code, analyse multi-étapes.

Par défaut désactivé. Deux façons de l'activer :
1. Via model_settings={'thinking': ...}  → cross-provider, le plus simple
2. Via capabilities=[Thinking(...)]      → même effet, plus explicite

Niveaux d'effort (unified) :
- True        → active avec le niveau par défaut du provider
- False       → désactive (ignoré sur les modèles always-on)
- 'minimal' / 'low' / 'medium' / 'high' / 'xhigh'
  → niveau spécifique (mappé au plus proche si non supporté)

Traduction par provider :
- Anthropic (Opus 4.6+) : adaptive thinking
- Anthropic (anciens)    : budget_tokens
- OpenAI                 : reasoning_effort
- Google (Gemini 3+)     : thinking_level
- Google (Gemini 2.5)    : thinking_budget
- Groq                   : reasoning_format
- xAI                    : reasoning_effort ('low' / 'high' seulement)

Les settings provider-spécifiques prennent la priorité quand les
deux sont définis.
"""


# =====================================================================
# PARTIE 1 : Unified thinking — cross-provider (le plus simple)
# =====================================================================

# La façon la plus simple d'activer le thinking, peu importe le
# provider. Pydantic AI traduit automatiquement vers le format natif.

# Via model_settings :
agent_unified = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    model_settings={'thinking': 'high'},
    instructions="Tu es un assistant de raisonnement logique.",
)

# Via la capability Thinking (même effet, plus explicite) :
agent_capability = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[Thinking(effort='high')],
    instructions="Tu es un assistant de raisonnement logique.",
)


# =====================================================================
# PARTIE 2 : Anthropic — extended thinking (modèles anciens)
# =====================================================================

# Pour les modèles comme claude-sonnet-4-5 ou claude-opus-4-5,
# on utilise le mode "enabled" avec un budget de tokens.
# ATTENTION : déprécié sur Opus 4.6, supprimé sur Opus 4.7+.

agent_extended = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    model_settings=AnthropicModelSettings(
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
    ),
    instructions="Tu es un assistant analytique.",
)


# =====================================================================
# PARTIE 3 : Anthropic — adaptive thinking (Opus 4.6+)
# =====================================================================

# Adaptive thinking : le modèle décide dynamiquement quand et
# combien réfléchir selon la complexité de la requête.
# Remplace extended thinking sur les modèles récents.
# Active aussi automatiquement l'interleaved thinking.

agent_adaptive = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    model_settings=AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive'},
        anthropic_effort='high',  # Contrôle l'effort global (indépendant du thinking)
    ),
    instructions="Tu es un assistant de recherche approfondie.",
)

# Opus 4.7 supporte aussi effort='xhigh' pour les tâches très complexes.


# =====================================================================
# PARTIE 4 : Anthropic — interleaved thinking
# =====================================================================

# L'interleaved thinking permet au modèle de réfléchir entre
# chaque étape (entre les tool calls par exemple).
# Sur les modèles anciens, nécessite un header beta explicite.

agent_interleaved = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    model_settings=AnthropicModelSettings(
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 10000},
        extra_headers={'anthropic-beta': 'interleaved-thinking-2025-05-14'},
    ),
    instructions="Tu es un assistant qui réfléchit étape par étape.",
)


# =====================================================================
# PARTIE 5 : OpenAI — reasoning effort
# =====================================================================

# Pour les modèles OpenAI (o1, o3, gpt-5+), le thinking est
# contrôlé via reasoning_effort. Les tags <think> sont convertis
# automatiquement en ThinkingPart.

# Via le unified setting (recommandé) :
# agent_openai = Agent(
#     'openai:o3',
#     model_settings={'thinking': 'high'},
# )

# Via les OpenAI Responses API (settings natifs) :
# from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
# model = OpenAIResponsesModel('gpt-5.2')
# settings = OpenAIResponsesModelSettings(
#     openai_reasoning_effort='low',
#     openai_reasoning_summary='detailed',
# )
# agent_openai_native = Agent(model, model_settings=settings)


# =====================================================================
# PARTIE 6 : Google — thinking config
# =====================================================================

# Pour Gemini 3+ : thinking_level (HIGH, MEDIUM, LOW)
# Pour Gemini 2.5 : thinking_budget (nombre de tokens)

# from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
# model = GoogleModel('gemini-3-pro-preview')
# settings = GoogleModelSettings(
#     google_thinking_config={'include_thoughts': True}
# )
# agent_google = Agent(model, model_settings=settings)


# =====================================================================
# PARTIE 7 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Unified thinking (cross-provider) ---
    logger.info("=== Unified thinking (effort=high) ===")
    result = agent_unified.run_sync(
        "Quel est le plus petit nombre entier positif divisible par 1, 2, 3, ..., 10 ?"
    )
    logger.success(f"Réponse : {result.output[:200]}...")

    # --- Démo 2 : Thinking capability ---
    logger.info("=== Thinking capability ===")
    result = agent_capability.run_sync(
        "Si A implique B et B implique C, et que A est vrai, que peut-on déduire ?"
    )
    logger.success(f"Réponse : {result.output[:200]}...")

    # --- Démo 3 : Adaptive thinking ---
    logger.info("=== Adaptive thinking (Anthropic) ===")
    result = agent_adaptive.run_sync(
        "Compare les avantages et inconvénients de REST vs GraphQL pour une API mobile."
    )
    logger.success(f"Réponse : {result.output[:200]}...")
