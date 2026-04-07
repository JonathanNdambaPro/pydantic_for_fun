import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
WEB CHAT UI BASICS — Interface de chat web intégrée

Pydantic AI inclut une interface web de chat pour interagir avec
un agent directement dans le navigateur.

Installation :
  uv add 'pydantic-ai-slim[web]'

Usage :
  1. Créer une app avec agent.to_web()
  2. Lancer avec uvicorn : uvicorn module:app --host 127.0.0.1 --port 7932

Points clés :
- Destiné au développement local et débogage
- En production → utiliser UI Event Stream pour un frontend custom
- L'UI est récupérée depuis un CDN et cachée localement
- Routes réservées : /, /{id}, /api/chat, /api/configure, /api/health
- L'app ne peut PAS être montée sur un sous-chemin (/chat etc.)

Pour la CLI : `clai web` (voir docs CLI)
"""


# =====================================================================
# PARTIE 1 : Web app basique
# =====================================================================

# Le plus simple : agent.to_web() retourne une app ASGI (Starlette).
# Lancer avec : uvicorn 11_web_ui.1_web_ui_basics:app_basic --port 7932

agent_basic = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant. Réponds en français de manière concise.',
)

app_basic = agent_basic.to_web()


# =====================================================================
# PARTIE 2 : Web app avec des tools
# =====================================================================

# Les outils de l'agent sont disponibles dans l'interface web.

agent_tools = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant météo. Utilise les outils disponibles.',
)


@agent_tools.tool_plain
def get_weather(city: str) -> str:
    """Retourne la météo d'une ville."""
    weather_data = {
        "paris": "Ensoleillé, 22°C",
        "tokyo": "Nuageux, 18°C",
        "new york": "Pluvieux, 15°C",
    }
    return weather_data.get(city.lower(), f"Météo inconnue pour {city}")


@agent_tools.tool_plain
def get_time(city: str) -> str:
    """Retourne l'heure dans une ville."""
    return f"Il est 14:30 à {city}"


app_tools = agent_tools.to_web()


# =====================================================================
# PARTIE 3 : Web app avec instructions supplémentaires
# =====================================================================

# On peut passer des instructions supplémentaires incluses à chaque run.

agent_friendly = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions='Tu es un assistant polyvalent.',
)

app_friendly = agent_friendly.to_web(
    instructions='Réponds toujours de manière amicale et en français.',
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

# Pour lancer l'un des apps :
#   uvicorn pydantic_ai_for_fun.11_web_ui.1_web_ui_basics:app_basic --port 7932
#   uvicorn pydantic_ai_for_fun.11_web_ui.1_web_ui_basics:app_tools --port 7932
#   uvicorn pydantic_ai_for_fun.11_web_ui.1_web_ui_basics:app_friendly --port 7932

if __name__ == "__main__":
    import uvicorn

    logger.info("=== Lancement Web UI basique sur http://127.0.0.1:7932 ===")
    logger.info("Ouvrez votre navigateur pour interagir avec l'agent.")
    uvicorn.run(app_tools, host="127.0.0.1", port=7932)
