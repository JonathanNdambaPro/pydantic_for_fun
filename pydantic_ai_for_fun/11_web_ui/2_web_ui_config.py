import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WEB UI CONFIG — Configuration avancée de l'interface web

Options de configuration de agent.to_web() :

- models          → modèles supplémentaires (liste ou dict avec labels)
- builtin_tools   → outils intégrés (CodeExecution, WebSearch...)
- instructions    → instructions supplémentaires par run
- html_source     → source HTML custom (fichier local ou URL)

Modèles :
- Liste de noms/instances → ['openai:gpt-5.2', 'anthropic:claude-sonnet-4-6']
- Dict label→modèle     → {'GPT': 'openai:gpt-5.2', 'Claude': model_instance}

Builtin tools :
- CodeExecutionTool() → exécution de code
- WebSearchTool()     → recherche web
- ⚠ MemoryTool n'est PAS supporté via to_web()
  → configurer MemoryTool directement sur l'agent

HTML source (offline / enterprise) :
- Télécharger le HTML : curl -o ui.html <DEFAULT_HTML_URL>
- Passer le chemin : agent.to_web(html_source='~/ui.html')
- Ou une URL custom : agent.to_web(html_source='https://cdn.example.com/ui.html')

Routes réservées (ne pas écraser) :
- /            → UI de chat
- /{id}        → conversation par ID
- /api/chat    → endpoint chat (POST, OPTIONS)
- /api/configure → config frontend (GET)
- /api/health  → health check (GET)
"""


# =====================================================================
# PARTIE 1 : Plusieurs modèles disponibles dans l'UI
# =====================================================================

# On peut proposer plusieurs modèles à l'utilisateur.
# Via une liste :

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant. Réponds en français.',
)

app_multi_models_list = agent.to_web(
    models=[
        'gateway/anthropic:claude-sonnet-4-6',
        'gateway/anthropic:claude-haiku-4-5',
    ],
)

# Via un dict avec des labels personnalisés :

app_multi_models_dict = agent.to_web(
    models={
        'Claude Sonnet': 'gateway/anthropic:claude-sonnet-4-6',
        'Claude Haiku (rapide)': 'gateway/anthropic:claude-haiku-4-5',
    },
)


# =====================================================================
# PARTIE 2 : Builtin tools (CodeExecution, WebSearch)
# =====================================================================

# Les builtin tools sont proposés comme options dans l'UI
# si le modèle sélectionné les supporte.

agent_with_builtins = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant avancé avec des outils intégrés.',
)

app_builtins = agent_with_builtins.to_web(
    builtin_tools=[
        CodeExecutionTool(),
        WebSearchTool(),
    ],
)


# =====================================================================
# PARTIE 3 : Combiner modèles + tools + instructions
# =====================================================================

# Configuration complète avec tout ensemble.

agent_full = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant de développement.',
)


@agent_full.tool_plain
def run_linter(code: str) -> str:
    """Simule un linter sur du code Python."""
    issues = []
    if "import *" in code:
        issues.append("Wildcard import détecté")
    if len(code.split("\n")) > 100:
        issues.append("Fichier trop long (> 100 lignes)")
    if not issues:
        return "Aucun problème détecté !"
    return f"Problèmes : {', '.join(issues)}"


@agent_full.tool_plain
def search_docs(query: str) -> str:
    """Cherche dans la documentation."""
    return f"Documentation trouvée pour '{query}': voir https://docs.example.com/{query}"


app_full = agent_full.to_web(
    models={
        'Sonnet (défaut)': 'gateway/anthropic:claude-sonnet-4-6',
        'Haiku (rapide)': 'gateway/anthropic:claude-haiku-4-5',
    },
    builtin_tools=[CodeExecutionTool()],
    instructions='Réponds toujours en français. Sois technique et précis.',
)


# =====================================================================
# PARTIE 4 : Source HTML custom (offline / enterprise)
# =====================================================================

# Pour un usage offline ou en environnement enterprise,
# on peut pointer vers un fichier HTML local.

# Étape 1 : Télécharger le HTML (une seule fois, avec internet)
# from pydantic_ai.ui import DEFAULT_HTML_URL
# print(DEFAULT_HTML_URL)
# → curl -o ~/pydantic-ai-ui.html <url>

# Étape 2 : Utiliser le fichier local
# app_offline = agent.to_web(html_source='~/pydantic-ai-ui.html')

# Ou une URL custom (CDN interne d'entreprise)
# app_enterprise = agent.to_web(html_source='https://cdn.internal.com/ui.html')


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

# Pour lancer :
#   uvicorn pydantic_ai_for_fun.11_web_ui.2_web_ui_config:app_full --port 7932

if __name__ == "__main__":
    import uvicorn

    logger.info("=== Web UI configurée sur http://127.0.0.1:7932 ===")
    logger.info("Modèles disponibles : Sonnet, Haiku")
    logger.info("Builtin tools : CodeExecution")
    logger.info("Tools custom : run_linter, search_docs")
    uvicorn.run(app_full, host="127.0.0.1", port=7932)
