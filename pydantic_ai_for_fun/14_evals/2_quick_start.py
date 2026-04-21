import asyncio
import time

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    EqualsExpected,
    IsInstance,
    LLMJudge,
    MaxDuration,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
QUICK START EVALS — Démarrage rapide avec Pydantic Evals

Pydantic Evals fonctionne avec n'importe quelle fonction, pas seulement
des systèmes IA. Ce fichier illustre les cas d'usage les plus courants
à travers des exemples simples et déterministes.

Concepts clés :
- Dataset : ensemble de cases de test + evaluators (optionnels)
- Case : scénario de test avec inputs et expected_output optionnel
- Evaluator : fonction qui évalue/valide les résultats
- EvaluationReport : résultats de l'évaluation avec métriques

Evaluators built-in utilisés ici :
- EqualsExpected : vérifie le match exact avec expected_output
- Contains : vérifie qu'une valeur est présente dans l'output
- IsInstance : vérifie le type de l'output
- MaxDuration : vérifie que l'exécution ne dépasse pas un délai
- LLMJudge : utilise un LLM pour évaluer des qualités subjectives (coûte de l'argent !)
"""


# =====================================================================
# PARTIE 1 : Validation basique — Transformation de texte
# =====================================================================

# Exemple le plus simple : on évalue une fonction uppercase
# avec EqualsExpected (match exact) et Contains (sous-chaîne).

logger.info("Création du dataset uppercase_tests...")

uppercase_dataset = Dataset(
    name='uppercase_tests',
    cases=[
        Case(
            name='uppercase_basic',
            inputs='hello world',
            expected_output='HELLO WORLD',
        ),
        Case(
            name='uppercase_with_numbers',
            inputs='hello 123',
            expected_output='HELLO 123',
        ),
        Case(
            name='uppercase_mixed',
            inputs='Bonjour le Monde',
            expected_output='BONJOUR LE MONDE',
        ),
    ],
    evaluators=[
        EqualsExpected(),
        Contains(value='HELLO', case_sensitive=True),
    ],
)


def uppercase_text(text: str) -> str:
    """Fonction simple à évaluer : met le texte en majuscules."""
    return text.upper()


# =====================================================================
# PARTIE 2 : Validation de structure — Vérification de type
# =====================================================================

# On peut vérifier que l'output est du bon type et contient
# certaines valeurs avec IsInstance et Contains.

logger.info("Création du dataset dict_validation...")


def process_data(data: dict) -> dict:
    """Traite des données et renvoie un résultat structuré."""
    if 'required_key' in str(data.get('data', '')):
        return {'result': 'success', 'data': data['data']}
    return {'result': 'failure', 'error': 'missing required_key'}


dict_dataset = Dataset(
    name='dict_validation',
    cases=[
        Case(
            name='with_required_key',
            inputs={'data': 'required_key present'},
            expected_output={'result': 'success', 'data': 'required_key present'},
        ),
        Case(
            name='without_required_key',
            inputs={'data': 'some other data'},
            expected_output={'result': 'failure', 'error': 'missing required_key'},
        ),
    ],
    evaluators=[
        IsInstance(type_name='dict'),
        EqualsExpected(),
    ],
)


# =====================================================================
# PARTIE 3 : Tests de performance — MaxDuration
# =====================================================================

# MaxDuration vérifie que la fonction s'exécute dans un délai donné.
# Utile pour s'assurer que le système répond aux exigences de latence.

logger.info("Création du dataset performance_test...")


def fast_function(text: str) -> str:
    """Fonction rapide — devrait passer le test de durée."""
    return text.upper()


def slow_function(text: str) -> str:
    """Fonction lente — va échouer le test de durée."""
    time.sleep(0.5)
    return text.upper()


performance_dataset = Dataset(
    name='performance_test',
    cases=[
        Case(
            name='quick_transform',
            inputs='test input',
            expected_output='TEST INPUT',
        ),
    ],
    evaluators=[
        EqualsExpected(),
        MaxDuration(seconds=0.1),
    ],
)


# =====================================================================
# PARTIE 4 : LLM as a Judge — Évaluation subjective (coûte de l'argent)
# =====================================================================

# LLMJudge utilise un LLM pour évaluer des qualités subjectives :
# pertinence, précision, utilité, ton, etc.
# ATTENTION : chaque évaluation fait un appel LLM → ça coûte de l'argent !

logger.info("Création du dataset llm_judge_test...")

llm_judge_dataset = Dataset(
    name='llm_judge_test',
    cases=[
        Case(
            name='capital_question',
            inputs='What is the capital of France?',
            expected_output='Paris',
        ),
        Case(
            name='verbose_answer',
            inputs='What is 2+2?',
            expected_output='4',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is accurate and helpful',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 5 : Exécution — Démos des évaluations
# =====================================================================


async def main():
    """Exécute les démos d'évaluation du quick start."""

    # --- Démo 1 : Uppercase basique ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Validation basique (uppercase)")
    logger.info("=" * 60)
    report = await uppercase_dataset.evaluate(uppercase_text)
    report.print(include_input=True, include_output=True, include_durations=False)
    logger.success("Uppercase : tous les tests passent !")

    # --- Démo 2 : Validation de structure ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Validation de structure (dict)")
    logger.info("=" * 60)
    report = await dict_dataset.evaluate(process_data)
    report.print(include_input=True, include_output=True, include_durations=False)
    logger.success("Dict validation terminée !")

    # --- Démo 3 : Performance — fonction rapide ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Test de performance (fast_function)")
    logger.info("=" * 60)
    report_fast = await performance_dataset.evaluate(fast_function)
    report_fast.print(include_input=True, include_output=True)
    logger.success("Fast function : test de durée passé !")

    # --- Démo 4 : Performance — fonction lente ---
    logger.info("=" * 60)
    logger.info("DÉMO 4 : Test de performance (slow_function)")
    logger.info("=" * 60)
    report_slow = await performance_dataset.evaluate(slow_function)
    report_slow.print(include_input=True, include_output=True)
    logger.warning("Slow function : MaxDuration devrait échouer !")

    # --- Démo 5 : LLM Judge (ATTENTION : coûte de l'argent) ---
    # Chaque case fait un appel à Claude → coût réel.
    logger.info("=" * 60)
    logger.info("DÉMO 5 : LLM Judge (coûte de l'argent)")
    logger.info("=" * 60)

    async def simple_answer(question: str) -> str:
        """Simule une réponse simple."""
        answers = {
            'What is the capital of France?': 'The capital of France is Paris.',
            'What is 2+2?': 'The answer is 4.',
        }
        return answers.get(question, 'I do not know.')

    report_llm = await llm_judge_dataset.evaluate(simple_answer)
    report_llm.print(include_input=True, include_output=True)

    logger.success("Toutes les démos du quick start sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
