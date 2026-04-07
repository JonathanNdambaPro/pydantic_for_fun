import asyncio

import httpx
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from pydantic_ai_for_fun.settings import settings

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx()
logger.configure(handlers=[logfire.loguru_handler()])

"""
OVERRIDING DEPENDENCIES (agent.override)

Quand ton agent est appelé au fin fond de ton code métier,
tu ne peux pas toujours passer `deps=` au `run()` directement.

`agent.override(deps=...)` permet de REMPLACER les dépendances
pour TOUS les appels à `run()` dans le scope du `with`, sans
modifier le code appelant.

Cas d'usage :
- Tests : remplacer le vrai client HTTP par un mock
- Staging : pointer vers une API de test
- Démo : retourner des données en dur sans consommer de crédits
- Multi-tenant : chaque client a sa propre clé API
- Debug : injecter un client HTTP avec logs verbeux

C'est du monkey-patching propre, scopé, et thread-safe.
"""


# =====================================================================
# PARTIE 1 : L'agent et ses dépendances (code "de production")
# =====================================================================

class WeatherDeps(BaseModel):
    """Dépendances pour l'agent météo."""
    model_config = {"arbitrary_types_allowed": True}

    http_client: httpx.AsyncClient = Field(description="Client HTTP partagé")
    api_key: str = Field(default=settings.OPENWEATHERMAP_API_KEY, description="Clé API OpenWeatherMap")
    unite: str = Field(default="metric", description="'metric' (°C) ou 'imperial' (°F)")


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


@agent_meteo.tool
async def get_weather(ctx: RunContext[WeatherDeps], ville: str) -> str:
    """Récupère la météo actuelle pour une ville.

    Args:
        ville: Nom de la ville (ex: "Paris", "Tokyo")
    """
    logger.info(f"Appel API météo pour '{ville}' (unité: {ctx.deps.unite})")

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
# PARTIE 2 : Le code métier — enfoui dans l'application
# =====================================================================

# Imagine que cette fonction est appelée par un endpoint FastAPI,
# un worker Celery, ou n'importe quel code profond.
# Elle crée ses propres deps en interne → on ne peut pas facilement
# passer des deps de test depuis l'extérieur.
async def get_meteo_for_user(ville: str) -> str:
    """Code métier qui appelle l'agent. Simule un appel depuis une API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        deps = WeatherDeps(http_client=client)
        result = await agent_meteo.run(
            f"Quel temps fait-il à {ville} ?",
            deps=deps,
        )
    return result.output


# =====================================================================
# PARTIE 3 : Override pour les tests — deps mockées
# =====================================================================

# On crée une sous-classe de WeatherDeps qui simule les réponses.
# Pas besoin de vrai client HTTP ni de vraie clé API.
class MockWeatherDeps(WeatherDeps):
    """Deps mockées : le client HTTP n'est jamais utilisé car
    le tool est aussi overridé via l'agent model."""

    # On garde un vrai AsyncClient pour satisfaire le type,
    # mais on va override le model de l'agent pour ne jamais
    # appeler la vraie API.
    pass


# =====================================================================
# PARTIE 4 : Override pour le multi-tenant — clé API par client
# =====================================================================

# Chaque tenant (client) a sa propre config.
# On stocke juste la config, le client HTTP sera injecté au runtime.
TENANTS_CONFIG = {
    "client_france": {"api_key": settings.OPENWEATHERMAP_API_KEY, "unite": "metric"},
    "client_usa": {"api_key": settings.OPENWEATHERMAP_API_KEY, "unite": "imperial"},
    # En prod, chaque client aurait sa propre clé API
}


# =====================================================================
# PARTIE 5 : Exécution — démonstration des overrides
# =====================================================================

async def main():
    # --- TEST 1 : Appel normal (production) ---
    logger.info("=== TEST 1 : Appel normal (production) ===")
    result = await get_meteo_for_user("Paris")
    logger.success(f"Résultat prod : {result}")

    # --- TEST 2 : Override avec agent.override() ---
    # On remplace les deps pour TOUS les run() dans le scope.
    # Le code de get_meteo_for_user() n'est PAS modifié !
    # Même s'il crée ses propres deps en interne, l'override gagne.
    logger.info("=== TEST 2 : Override des deps (simule un test) ===")
    async with httpx.AsyncClient(timeout=10.0) as client:
        test_deps = WeatherDeps(
            http_client=client,
            api_key=settings.OPENWEATHERMAP_API_KEY,
            unite="imperial",  # On force Fahrenheit pour le test
        )
        with agent_meteo.override(deps=test_deps):
            # get_meteo_for_user crée ses propres deps en metric,
            # mais l'override les remplace par test_deps en imperial !
            result_override = await get_meteo_for_user("London")
            logger.success(f"Résultat override : {result_override}")

    # --- TEST 3 : Hors du scope override → retour à la normale ---
    logger.info("=== TEST 3 : Hors du override → comportement normal ===")
    result_normal = await get_meteo_for_user("Tokyo")
    logger.success(f"Résultat normal : {result_normal}")

    # --- TEST 4 : Override multi-tenant ---
    logger.info("=== TEST 4 : Override multi-tenant ===")
    async with httpx.AsyncClient(timeout=10.0) as client:
        for tenant_name, config in TENANTS_CONFIG.items():
            # On construit les deps du tenant avec le client HTTP partagé
            tenant_deps = WeatherDeps(http_client=client, **config)
            with agent_meteo.override(deps=tenant_deps):
                ville = "Paris" if "france" in tenant_name else "New York"
                logger.info(f"Tenant '{tenant_name}' → {ville} ({config['unite']})")
                result_tenant = await get_meteo_for_user(ville)
                logger.success(f"  Résultat : {result_tenant}")


asyncio.run(main())
