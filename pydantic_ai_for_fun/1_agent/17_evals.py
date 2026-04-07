import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance, LLMJudge

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
EVALS — TESTER SYSTÉMATIQUEMENT SON AGENT

Le problème : tu modifies ton system prompt, tu changes de modèle, tu ajoutes un tool...
Comment savoir si ton agent est toujours bon ? Tu ne vas pas tester à la main à chaque fois.

Les Evals c'est comme des tests unitaires, mais pour l'IA :
1. Tu définis des cas de test (input → output attendu)
2. Tu les exécutes contre ton agent
3. Tu obtiens un rapport avec des scores

3 types de vérifications :
- assertions (bool)  : "la réponse contient-elle X ?" → pass/fail
- scores (float)     : "qualité de la réponse sur 100" → 0-100
- labels (str)       : "catégorie de la réponse" → "correct", "partiel", etc.
"""


# =====================================================================
# PARTIE 1 : Un agent simple à évaluer
# =====================================================================

class CityInfo(BaseModel):
    city: str = Field(description="Le nom de la ville")
    country: str = Field(description="Le pays de la ville")
    population_millions: float = Field(description="Population approximative en millions")


agent = Agent(
    'gateway/openai:gpt-4o',
    output_type=CityInfo,
    instructions="Tu es un expert en géographie. Réponds avec des données précises.",
)


async def run_agent(question: str) -> CityInfo:
    """La fonction que les evals vont tester."""
    result = await agent.run(question)
    return result.output


# =====================================================================
# PARTIE 2 : Dataset — les cas de test
# =====================================================================

dataset = Dataset(
    name='geo_agent_eval',
    # Evaluateurs appliqués à TOUS les cas
    evaluators=[
        IsInstance(type_name='CityInfo'),  # Vérifie que l'output est bien un CityInfo
    ],
    cases=[
        # --- Cas faciles (le LLM doit cartonner) ---
        Case(
            name='capitale_france',
            inputs='Quelle est la capitale de la France ?',
            expected_output=CityInfo(city='Paris', country='France', population_millions=2.1),
            evaluators=[
                EqualsExpected(),  # Match exact avec expected_output
            ],
        ),
        Case(
            name='capitale_japon',
            inputs='Quelle est la capitale du Japon ?',
            expected_output=CityInfo(city='Tokyo', country='Japon', population_millions=13.9),
            evaluators=[
                EqualsExpected(),
            ],
        ),

        # --- Cas ambigu (test de robustesse) ---
        Case(
            name='plus_grande_ville_usa',
            inputs='Quelle est la plus grande ville des États-Unis ?',
            expected_output=CityInfo(city='New York', country='États-Unis', population_millions=8.3),
            evaluators=[
                # LLMJudge utilise un LLM pour évaluer la qualité de la réponse
                # Idéal pour les cas où la réponse exacte peut varier
                LLMJudge(
                    rubric=(
                        "La ville doit être New York (ou New York City). "
                        "Le pays doit être les États-Unis (ou USA, US). "
                        "La population doit être entre 8 et 9 millions."
                    ),
                    model='gateway/openai:gpt-4o',
                ),
            ],
        ),

        # --- Cas piège (question vague) ---
        Case(
            name='ville_lumiere',
            inputs='Quelle est la Ville Lumière ?',
            expected_output=CityInfo(city='Paris', country='France', population_millions=2.1),
            evaluators=[
                LLMJudge(
                    rubric="La réponse doit identifier Paris comme la Ville Lumière.",
                    model='gateway/openai:gpt-4o',
                ),
            ],
        ),
    ],
)


# =====================================================================
# PARTIE 3 : Exécution et rapport
# =====================================================================

logger.info("Lancement des evals...")

report = dataset.evaluate_sync(
    run_agent,
    max_concurrency=3,  # 3 cas en parallèle max
)

# Afficher le rapport
report.print(include_input=True, include_output=True)

# Résumé des moyennes
averages = report.averages()
logger.info(f"Assertions passées : {averages.assertions}")
logger.info(f"Durée moyenne : {averages.task_duration:.2f}s")
logger.info(f"Cas réussis : {len(report.cases)}/{len(report.cases) + len(report.failures)}")

if report.failures:
    for failure in report.failures:
        logger.error(f"Cas échoué : {failure.name} — {failure.error}")
