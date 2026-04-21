import asyncio
import re
from dataclasses import dataclass
from datetime import date, timezone

import logfire
import pytest
from dirty_equals import IsNow, IsStr
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, capture_run_messages, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TESTING — Écrire des tests unitaires pour du code Pydantic AI

Les tests unitaires pour du code Pydantic AI suivent les mêmes principes que
n'importe quel test Python classique. La librairie fournit des outils dédiés
pour éviter d'appeler de vrais LLMs pendant les tests :

- TestModel : modèle de test qui appelle automatiquement tous les outils puis
  renvoie une réponse texte ou structurée basée sur les schémas JSON.
- FunctionModel : modèle personnalisable où on définit soi-même la logique
  de réponse (utile pour contrôler les arguments passés aux outils).
- Agent.override : remplace le modèle, les dépendances ou les outils d'un
  agent dans le code applicatif sans le modifier.
- capture_run_messages : context manager pour capturer tous les messages
  échangés pendant un run (requêtes, réponses, appels d'outils).
- ALLOW_MODEL_REQUESTS : flag global pour bloquer toute requête vers un
  vrai modèle (sécurité anti-oubli dans les tests).

Stratégie recommandée :
- pytest comme framework de test
- inline-snapshot pour les assertions longues
- dirty-equals pour comparer des structures complexes (timestamps, IDs)
- TestModel ou FunctionModel à la place du vrai modèle
- ALLOW_MODEL_REQUESTS=False en global dans les modules de test
"""


# =====================================================================
# PARTIE 1 : Code applicatif — Agent météo avec dépendances
# =====================================================================

# Simulons un service météo et une connexion base de données
# pour illustrer un cas réaliste de code à tester.


@dataclass
class WeatherService:
    """Service météo simulé avec historique et prévisions."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get_historic_weather(self, location: str, forecast_date: date) -> str:
        return "Sunny with a chance of rain"

    def get_forecast(self, location: str, forecast_date: date) -> str:
        return "Rainy with a chance of sun"


@dataclass
class DatabaseConn:
    """Connexion base de données simulée pour stocker les prévisions."""

    _forecasts: dict[int, str] = None

    def __post_init__(self):
        if self._forecasts is None:
            self._forecasts = {}

    async def store_forecast(self, user_id: int, forecast: str):
        self._forecasts[user_id] = forecast

    async def get_forecast(self, user_id: int) -> str | None:
        return self._forecasts.get(user_id)


# Définition de l'agent météo
weather_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=WeatherService,
    instructions='Fournir les prévisions météo pour les lieux demandés par l\'utilisateur.',
)


@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    """Obtient la météo pour un lieu et une date donnés.

    Args:
        location: Nom du lieu (ex: 'Paris', 'London').
        forecast_date: Date de la prévision (YYYY-MM-DD).
    """
    logger.info(f"[weather_forecast] {location} @ {forecast_date}")
    if forecast_date < date.today():
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Lance les prévisions météo pour une liste de prompts utilisateur et sauvegarde."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # Exécution parallèle de tous les prompts
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )


# =====================================================================
# PARTIE 2 : Tests avec TestModel — Le plus simple et rapide
# =====================================================================

# TestModel appelle automatiquement tous les outils enregistrés puis
# renvoie une réponse. Il génère des données valides basées sur les
# schémas JSON des outils (pas de ML, juste du code procédural).

# IMPORTANT : On bloque les requêtes vers les vrais modèles
# pour éviter toute fuite accidentelle dans les tests.
models.ALLOW_MODEL_REQUESTS = False

# pytest.mark.anyio permet d'exécuter des tests async
pytestmark = pytest.mark.anyio


async def test_forecast_with_test_model():
    """Test basique avec TestModel : vérifie que le pipeline complet fonctionne."""
    logger.info("--- Test avec TestModel ---")
    conn = DatabaseConn()
    user_id = 1

    # capture_run_messages permet d'inspecter tous les messages échangés
    with capture_run_messages() as messages:
        # Agent.override remplace le modèle sans modifier le code applicatif
        with weather_agent.override(model=TestModel()):
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)

    # Vérification du résultat stocké en base
    forecast = await conn.get_forecast(user_id)
    logger.success(f"Prévision stockée : {forecast}")
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'

    # Vérification détaillée des messages échangés avec dirty-equals
    # pour les timestamps et IDs dynamiques
    assert messages == [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            instructions="Fournir les prévisions météo pour les lieux demandés par l'utilisateur.",
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=RequestUsage(
                input_tokens=60,
                output_tokens=7,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            instructions="Fournir les prévisions météo pour les lieux demandés par l'utilisateur.",
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=RequestUsage(
                input_tokens=66,
                output_tokens=16,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
    ]
    logger.success("Test TestModel passé !")


# =====================================================================
# PARTIE 3 : Tests avec FunctionModel — Contrôle total des réponses
# =====================================================================

# TestModel génère des données "mécaniques" (ex: date='2024-01-01', location='a').
# Avec FunctionModel, on contrôle exactement les arguments passés aux outils,
# ce qui permet de tester des branches spécifiques (ex: date future → get_forecast).


def call_weather_forecast(
    messages: list, info: AgentInfo
) -> ModelResponse:
    """Fonction de modèle personnalisée qui simule un LLM intelligent.

    Premier appel : extrait la date du prompt et appelle l'outil weather_forecast.
    Deuxième appel : formate le résultat de l'outil en réponse texte.
    """
    if len(messages) == 1:
        # Premier appel — on extrait la date du prompt utilisateur
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # Deuxième appel — on formate le retour de l'outil
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])


async def test_forecast_future_date():
    """Test avec FunctionModel : vérifie que get_forecast est appelé pour une date future."""
    logger.info("--- Test avec FunctionModel (date future) ---")
    conn = DatabaseConn()
    user_id = 1

    with weather_agent.override(model=FunctionModel(call_weather_forecast)):
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    logger.success(f"Prévision stockée : {forecast}")
    # Cette fois, get_forecast est appelé (date future) → résultat différent
    assert forecast == 'The forecast is: Rainy with a chance of sun'
    logger.success("Test FunctionModel passé !")


# =====================================================================
# PARTIE 4 : Fixtures pytest — Override réutilisable
# =====================================================================

# Pour éviter de répéter `weather_agent.override(model=TestModel())`
# dans chaque test, on utilise une fixture pytest.


@pytest.fixture
def override_weather_agent():
    """Fixture qui remplace le modèle de l'agent par TestModel pour tous les tests."""
    with weather_agent.override(model=TestModel()):
        yield


async def test_with_fixture(override_weather_agent: None):
    """Exemple de test utilisant la fixture — le modèle est déjà overridé."""
    logger.info("--- Test avec fixture ---")
    conn = DatabaseConn()
    result = await weather_agent.run(
        'Weather in Paris tomorrow?',
        deps=WeatherService(),
    )
    logger.success(f"Résultat : {result.output}")
    assert isinstance(result.output, str)
    logger.success("Test avec fixture passé !")


# =====================================================================
# PARTIE 5 : Exécution — Démos des tests
# =====================================================================


async def main():
    """Exécute les démos de tests (hors pytest, pour illustration)."""

    # --- Démo 1 : Test avec TestModel ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Test avec TestModel")
    logger.info("=" * 60)
    await test_forecast_with_test_model()

    # --- Démo 2 : Test avec FunctionModel (date future) ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Test avec FunctionModel")
    logger.info("=" * 60)
    await test_forecast_future_date()

    # --- Démo 3 : Test avec fixture (simulation manuelle) ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Test avec fixture (simulation)")
    logger.info("=" * 60)
    with weather_agent.override(model=TestModel()):
        conn = DatabaseConn()
        result = await weather_agent.run(
            'Météo à Tokyo demain ?',
            deps=WeatherService(),
        )
        logger.success(f"Résultat fixture simulée : {result.output}")

    logger.success("Toutes les démos de testing sont passées !")


if __name__ == "__main__":
    asyncio.run(main())
