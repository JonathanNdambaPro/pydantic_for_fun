import asyncio

import httpx
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from pydantic_ai_for_fun.settings import settings

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
logger.configure(handlers=[logfire.loguru_handler()])

"""
DEPENDENCIES (Injection de Dépendances)

Les dépendances permettent de passer des objets externes (clients HTTP,
clés API, connexions DB, config…) à ton agent. Elles sont accessibles
dans les tools et les system prompts dynamiques via `RunContext`.

Pourquoi ?
- Découplage : l'agent ne crée pas ses propres ressources
- Testabilité : tu peux injecter des mocks en test
- Partage : un même client HTTP pour tous les tools (connection pooling)

Flux :
1. Définir un type de deps (dataclass, BaseModel, ou même un simple str)
2. Déclarer `deps_type=MonType` sur l'Agent
3. Passer `deps=mon_instance` au moment du `run()`
4. Récupérer `ctx.deps` dans les tools / system prompts dynamiques
"""


# =====================================================================
# PARTIE 1 : Définir les dépendances avec un dataclass
# =====================================================================

class WeatherDeps(BaseModel):
    """Conteneur pour les dépendances de notre agent météo."""
    model_config = {"arbitrary_types_allowed": True}

    http_client: httpx.AsyncClient = Field(description="Client HTTP partagé pour les appels API")
    api_key: str = Field(default=settings.OPENWEATHERMAP_API_KEY, description="Clé API OpenWeatherMap")
    unite: str = Field(default="metric", description="Système d'unités : 'metric' (°C) ou 'imperial' (°F)")


# L'agent déclare le TYPE de dépendances qu'il attend.
# Il ne reçoit les dépendances concrètes qu'au moment du `run()`.
agent_meteo = Agent(
    'gateway/openai:gpt-4o',
    deps_type=WeatherDeps,
    instructions=(
        "Tu es un assistant météo. "
        "Utilise TOUJOURS le tool get_weather pour répondre. "
        "Donne la température et une courte description."
    ),
    output_type=str,
)


# =====================================================================
# PARTIE 2 : System prompt dynamique utilisant les deps
# =====================================================================

# Le system prompt dynamique peut lire les dépendances pour
# adapter le comportement de l'agent au runtime.
@agent_meteo.system_prompt
def adapter_unite(ctx: RunContext[WeatherDeps]) -> str:
    if ctx.deps.unite == "metric":
        return "Affiche les températures en °C."
    return "Affiche les températures en °F."


# =====================================================================
# PARTIE 3 : Tools qui accèdent aux dépendances via RunContext
# =====================================================================

# `ctx.deps` donne accès à l'instance WeatherDeps passée au `run()`.
# Le type générique RunContext[WeatherDeps] assure le type-safety.
@agent_meteo.tool
async def get_weather(ctx: RunContext[WeatherDeps], ville: str) -> str:
    """Récupère la météo actuelle pour une ville.

    Args:
        ville: Nom de la ville (ex: "Paris", "Tokyo")
    """
    logger.info(f"Appel API météo pour '{ville}' (unité: {ctx.deps.unite})")

    # On utilise le client HTTP partagé depuis les dépendances
    # (pas de `httpx.get()` en dur → testable, configurable)
    resp = await ctx.deps.http_client.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": ville,
            "appid": ctx.deps.api_key,
            "units": ctx.deps.unite,
            "lang": "fr",
        },
    )

    if resp.status_code != 200:
        logger.error(f"Erreur API : {resp.status_code} — {resp.text}")
        return f"Impossible de récupérer la météo pour '{ville}' (erreur {resp.status_code})."

    data = resp.json()
    temp = data["main"]["temp"]
    description = data["weather"][0]["description"]
    symbole = "°C" if ctx.deps.unite == "metric" else "°F"

    logger.success(f"{ville} : {temp}{symbole}, {description}")
    return f"{ville} : {temp}{symbole}, {description}"


# =====================================================================
# PARTIE 4 : ctx.agent — accéder aux propriétés de l'agent depuis un tool
# =====================================================================

# En plus de ctx.deps, RunContext expose ctx.agent qui donne accès
# à l'agent en cours d'exécution (nom, output_type, instructions…).
# Utile quand un même tool est partagé entre plusieurs agents
# et doit adapter son comportement selon l'agent qui l'appelle.
@agent_meteo.tool
async def debug_info(ctx: RunContext[WeatherDeps]) -> str:
    """Retourne des infos de debug sur l'agent et le contexte d'exécution."""
    return (
        f"Agent: {ctx.agent.name}, "
        f"Output type: {ctx.agent.output_type.__name__}, "
        f"Unité configurée: {ctx.deps.unite}, "
        f"Tentative: {ctx.retry}/{ctx.max_retries}, "
        f"Step: {ctx.run_step}"
    )


# =====================================================================
# PARTIE 5 : output_validator — valider la réponse finale avec les deps
# =====================================================================

# Le output_validator intercepte la réponse FINALE du LLM.
# Contrairement à un tool retry (qui valide les args d'un outil),
# ici on valide ce que le LLM renvoie à l'utilisateur.
# Si c'est invalide → ModelRetry → le LLM regénère sa réponse.
@agent_meteo.output_validator
async def valider_reponse_meteo(ctx: RunContext[WeatherDeps], output: str) -> str:
    """Vérifie que la réponse contient bien une température."""
    symbole = "°C" if ctx.deps.unite == "metric" else "°F"
    if symbole not in output:
        logger.warning(f"Réponse sans {symbole} détecté → retry")
        raise ModelRetry(
            f"Ta réponse doit contenir la température avec le symbole {symbole}. "
            f"Reformule en incluant la température."
        )
    logger.success(f"Réponse validée (contient {symbole})")
    return output


# =====================================================================
# PARTIE 6 : Exécution — on injecte les deps au moment du run
# =====================================================================

async def main():
    # Le client HTTP est créé UNE FOIS et partagé.
    # → Connection pooling, headers communs, timeout global…
    async with httpx.AsyncClient(timeout=10.0) as client:
        # On construit les dépendances concrètes
        deps = WeatherDeps(
            http_client=client,
            unite="metric",
        )

        # On passe les deps au `run()` — PAS à l'Agent()
        # C'est ça l'injection : l'agent est défini une fois,
        # mais les deps peuvent changer à chaque appel.
        logger.info("Demande météo Paris (metric)")
        result = await agent_meteo.run(
            "Quel temps fait-il à Paris ?",
            deps=deps,
        )
        logger.success(f"Réponse : {result.output}")

        # Même agent, dépendances différentes → comportement différent
        deps_us = WeatherDeps(
            http_client=client,
            unite="imperial",  # Maintenant en Fahrenheit
        )

        logger.info("Demande météo New York (imperial)")
        result2 = await agent_meteo.run(
            "What's the weather in New York?",
            deps=deps_us,
        )
        logger.success(f"Réponse : {result2.output}")


asyncio.run(main())
