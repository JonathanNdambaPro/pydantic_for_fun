import asyncio
import time
from datetime import timedelta

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    Contains,
    Equals,
    EqualsExpected,
    HasMatchingSpan,
    IsInstance,
    LLMJudge,
    MaxDuration,
    PrecisionRecallEvaluator,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
BUILT-IN EVALUATORS — Référence complète des evaluators fournis par Pydantic Evals

Pydantic Evals fournit des evaluators prêts à l'emploi couvrant les cas
les plus courants. Ce fichier est une référence exhaustive avec exemples.

Case-level evaluators :
┌──────────────────┬──────────────────────────┬──────────┬───────┬─────────┐
│ Evaluator        │ Usage                    │ Retour   │ Coût  │ Vitesse │
├──────────────────┼──────────────────────────┼──────────┼───────┼─────────┤
│ EqualsExpected   │ Match exact avec attendu │ bool     │ Gratuit│ Instant│
│ Equals           │ Match exact avec valeur  │ bool     │ Gratuit│ Instant│
│ Contains         │ Sous-chaîne / élément    │ bool+why │ Gratuit│ Instant│
│ IsInstance       │ Validation de type       │ bool+why │ Gratuit│ Instant│
│ MaxDuration      │ Seuil de performance     │ bool     │ Gratuit│ Instant│
│ HasMatchingSpan  │ Check comportemental     │ bool     │ Gratuit│ Rapide │
│ LLMJudge        │ Qualité subjective       │ bool/flt │ $$$   │ Lent   │
└──────────────────┴──────────────────────────┴──────────┴───────┴─────────┘

Report-level evaluators :
┌───────────────────────────┬─────────────────────┬────────────────┐
│ Evaluator                 │ Usage               │ Output         │
├───────────────────────────┼─────────────────────┼────────────────┤
│ ConfusionMatrixEvaluator  │ Matrice de confusion│ ConfusionMatrix│
│ PrecisionRecallEvaluator  │ Courbe PR + AUC     │ PrecisionRecall│
└───────────────────────────┴─────────────────────┴────────────────┘
"""


# =====================================================================
# PARTIE 1 : EqualsExpected — Match exact avec expected_output
# =====================================================================

# Compare ctx.output == ctx.expected_output via l'opérateur ==.
# Fonctionne avec n'importe quel type comparable (str, int, dict, list).
# Skip automatiquement si expected_output est None.

equals_expected_dataset = Dataset(
    name='equals_expected_demo',
    cases=[
        Case(name='string_match', inputs='hello', expected_output='HELLO'),
        Case(name='int_match', inputs='2+2', expected_output=4),
        Case(name='dict_match', inputs='data', expected_output={'status': 'ok', 'code': 200}),
        Case(name='list_match', inputs='items', expected_output=['a', 'b', 'c']),
        Case(name='no_expected', inputs='skip'),  # Pas d'expected_output → skip
    ],
    evaluators=[EqualsExpected()],
)


# =====================================================================
# PARTIE 2 : Equals — Match exact avec une valeur spécifique
# =====================================================================

# Contrairement à EqualsExpected qui compare avec expected_output,
# Equals compare avec une valeur fixe passée en paramètre.
# Utile pour vérifier des valeurs sentinelles ou des constantes.

equals_dataset = Dataset(
    name='equals_demo',
    cases=[
        Case(name='success_case', inputs='good_input'),
        Case(name='failure_case', inputs='bad_input'),
    ],
    evaluators=[
        Equals(value='success', evaluation_name='is_success'),
        Equals(value='error', evaluation_name='is_error'),
    ],
)


# =====================================================================
# PARTIE 3 : Contains — Sous-chaîne, élément ou clé-valeur
# =====================================================================

# Contains adapte son comportement selon le type de l'output :
# - str   → vérifie la sous-chaîne
# - list  → vérifie la membership
# - dict  → vérifie les paires clé-valeur

# --- Strings : sous-chaîne avec option case_sensitive ---
string_contains_dataset = Dataset(
    name='string_contains',
    cases=[
        Case(name='exact_sub', inputs='test', expected_output='Hello World'),
        Case(name='case_insensitive', inputs='test', expected_output='HELLO WORLD'),
    ],
    evaluators=[
        Contains(value='Hello', case_sensitive=True, evaluation_name='exact_hello'),
        Contains(value='hello', case_sensitive=False, evaluation_name='any_case_hello'),
    ],
)

# --- Lists : membership ---
list_contains_dataset = Dataset(
    name='list_contains',
    cases=[
        Case(name='has_apple', inputs='fruits', expected_output=['apple', 'banana', 'cherry']),
        Case(name='no_grape', inputs='fruits', expected_output=['apple', 'banana']),
    ],
    evaluators=[
        Contains(value='apple', evaluation_name='has_apple'),
        Contains(value='grape', evaluation_name='has_grape'),
    ],
)

# --- Dicts : paires clé-valeur ---
dict_contains_dataset = Dataset(
    name='dict_contains',
    cases=[
        Case(
            name='has_name',
            inputs='user',
            expected_output={'name': 'Alice', 'age': 30, 'role': 'admin'},
        ),
    ],
    evaluators=[
        Contains(value={'name': 'Alice'}, evaluation_name='has_alice'),
        Contains(value={'role': 'user'}, evaluation_name='has_user_role'),
    ],
)

# --- as_strings : convertit en str avant de comparer ---
as_strings_dataset = Dataset(
    name='as_strings_contains',
    cases=[
        Case(name='number_in_output', inputs='test', expected_output=42),
    ],
    evaluators=[
        Contains(value='42', as_strings=True, evaluation_name='contains_42_str'),
    ],
)


# =====================================================================
# PARTIE 4 : IsInstance — Validation de type
# =====================================================================

# Vérifie que l'output est une instance d'un type donné.
# Fonctionne avec les types built-in, Pydantic models, et classes custom.
# Parcourt le MRO (Method Resolution Order) pour l'héritage.

isinstance_dataset = Dataset(
    name='isinstance_demo',
    cases=[
        Case(name='is_str', inputs='text', expected_output='hello'),
        Case(name='is_dict', inputs='data', expected_output={'key': 'value'}),
        Case(name='is_list', inputs='items', expected_output=[1, 2, 3]),
        Case(name='is_int', inputs='number', expected_output=42),
    ],
    evaluators=[
        IsInstance(type_name='str', evaluation_name='check_str'),
        IsInstance(type_name='dict', evaluation_name='check_dict'),
    ],
)


# =====================================================================
# PARTIE 5 : MaxDuration — Seuil de performance
# =====================================================================

# Vérifie que l'exécution de la task ne dépasse pas un seuil.
# Accepte un float (secondes) ou un timedelta.

duration_dataset = Dataset(
    name='duration_demo',
    cases=[
        Case(name='fast_task', inputs='fast'),
        Case(name='slow_task', inputs='slow'),
    ],
    evaluators=[
        MaxDuration(seconds=0.5),
        MaxDuration(seconds=timedelta(milliseconds=200)),
    ],
)


# =====================================================================
# PARTIE 6 : HasMatchingSpan — Vérification comportementale (spans)
# =====================================================================

# Vérifie que les spans OpenTelemetry correspondent à un query.
# Nécessite Logfire configuré. Utile pour vérifier quels outils
# ont été appelés, quels chemins de code ont été exécutés.

span_dataset = Dataset(
    name='span_check_demo',
    cases=[
        Case(name='tool_check', inputs='search for data'),
    ],
    evaluators=[
        # Vérifie qu'un outil spécifique a été appelé
        HasMatchingSpan(
            query={'name_contains': 'search_database'},
            evaluation_name='used_database',
        ),
        # Vérifie qu'il n'y a pas eu d'erreurs
        HasMatchingSpan(
            query={'has_attributes': {'error': True}},
            evaluation_name='had_errors',
        ),
    ],
)


# =====================================================================
# PARTIE 7 : LLMJudge — LLM-as-a-Judge (coûteux)
# =====================================================================

# Utilise un LLM pour évaluer des qualités subjectives.
# Configurable : assertion seule, score seul, ou les deux.

# --- Mode assertion (défaut) : retourne bool + reason ---
judge_assertion_dataset = Dataset(
    name='judge_assertion',
    cases=[
        Case(
            name='helpful_check',
            inputs='How do I reset my password?',
            expected_output='Go to Settings > Security > Reset Password',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is factually accurate and addresses the question directly',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)

# --- Mode score : retourne float (0.0 à 1.0) + reason ---
judge_score_dataset = Dataset(
    name='judge_score',
    cases=[
        Case(
            name='quality_score',
            inputs='Explain quantum computing',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Overall response quality: clarity, accuracy, completeness',
            model='gateway/anthropic:claude-sonnet-4-6',
            score={'evaluation_name': 'quality', 'include_reason': True},
            assertion=False,
        ),
    ],
)

# --- Mode combiné : assertion + score ---
judge_combined_dataset = Dataset(
    name='judge_combined',
    cases=[
        Case(
            name='full_eval',
            inputs='Write a professional email',
            expected_output='A formal, clear, and concise email',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is professional, clear, and follows email conventions',
            include_input=True,
            include_expected_output=True,
            model='gateway/anthropic:claude-sonnet-4-6',
            score={'evaluation_name': 'professionalism', 'include_reason': True},
            assertion={'evaluation_name': 'is_professional', 'include_reason': True},
        ),
    ],
)


# =====================================================================
# PARTIE 8 : Report evaluators — ConfusionMatrix et PrecisionRecall
# =====================================================================

# Les report evaluators tournent UNE FOIS après tous les cases.
# Ils analysent l'ensemble des résultats pour des métriques globales.

# --- ConfusionMatrixEvaluator : matrice de confusion ---
confusion_dataset = Dataset(
    name='sentiment_classifier',
    cases=[
        Case(name='pos_1', inputs='I love this!', expected_output='positive'),
        Case(name='pos_2', inputs='Amazing product', expected_output='positive'),
        Case(name='neg_1', inputs='Terrible service', expected_output='negative'),
        Case(name='neg_2', inputs='Waste of money', expected_output='negative'),
        Case(name='neu_1', inputs='It is okay', expected_output='neutral'),
        Case(name='neu_2', inputs='Nothing special', expected_output='neutral'),
    ],
    evaluators=[EqualsExpected()],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
        ),
    ],
)

# --- PrecisionRecallEvaluator : courbe PR avec AUC ---
pr_dataset = Dataset(
    name='spam_detector',
    cases=[
        Case(name='spam_1', inputs='Buy now!!!', expected_output='spam'),
        Case(name='spam_2', inputs='Free money', expected_output='spam'),
        Case(name='ham_1', inputs='Meeting at 3pm', expected_output='ham'),
        Case(name='ham_2', inputs='Project update', expected_output='ham'),
        Case(name='edge', inputs='Special offer from boss', expected_output='ham'),
    ],
    evaluators=[EqualsExpected()],
    report_evaluators=[
        PrecisionRecallEvaluator(
            predicted_from='output',
            expected_from='expected_output',
            positive_label='spam',
        ),
    ],
)


# =====================================================================
# PARTIE 9 : Combinaison d'evaluators — pattern layered
# =====================================================================

# Best practice : checks déterministes rapides d'abord,
# LLM judges coûteux en dernier. Fail fast.

combined_dataset = Dataset(
    name='combined_evaluators',
    cases=[
        Case(
            name='complete_check',
            inputs='Explain REST vs GraphQL',
            expected_output='REST uses endpoints, GraphQL uses a single query endpoint',
        ),
    ],
    evaluators=[
        # Couche 1 : gratuit et instantané
        IsInstance(type_name='str'),
        Contains(value='REST', case_sensitive=True),
        MaxDuration(seconds=2.0),

        # Couche 2 : LLM (coûteux, en dernier)
        LLMJudge(
            rubric='Response is accurate, balanced, and covers key differences',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 10 : Exécution
# =====================================================================


async def demo_equals_expected():
    """Démo 1 : EqualsExpected sur différents types."""
    logger.info("=== DÉMO 1 : EqualsExpected (str, int, dict, list) ===")

    async def typed_task(key: str) -> str | int | dict | list:
        outputs = {
            'hello': 'HELLO',
            '2+2': 4,
            'data': {'status': 'ok', 'code': 200},
            'items': ['a', 'b', 'c'],
            'skip': 'anything',
        }
        return outputs.get(key, 'unknown')

    report = await equals_expected_dataset.evaluate(typed_task)
    report.print(include_input=True, include_output=True, include_durations=False)


async def demo_equals():
    """Démo 2 : Equals avec valeur sentinelle."""
    logger.info("=== DÉMO 2 : Equals (valeur sentinelle) ===")

    async def status_task(text: str) -> str:
        return 'success' if text == 'good_input' else 'error'

    report = await equals_dataset.evaluate(status_task)
    report.print(include_input=True, include_output=True, include_durations=False)


async def demo_contains():
    """Démo 3 : Contains sur strings, lists, dicts."""
    logger.info("=== DÉMO 3 : Contains (strings) ===")

    async def echo_task(text: str) -> str:
        return 'Hello World'

    report = await string_contains_dataset.evaluate(echo_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    logger.info("=== DÉMO 3b : Contains (lists) ===")

    async def list_task(text: str) -> list:
        return ['apple', 'banana', 'cherry']

    report = await list_contains_dataset.evaluate(list_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    logger.info("=== DÉMO 3c : Contains (dicts) ===")

    async def dict_task(text: str) -> dict:
        return {'name': 'Alice', 'age': 30, 'role': 'admin'}

    report = await dict_contains_dataset.evaluate(dict_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)


async def demo_isinstance():
    """Démo 4 : IsInstance sur différents types."""
    logger.info("=== DÉMO 4 : IsInstance ===")

    async def mixed_task(key: str) -> str | dict | list | int:
        outputs = {
            'text': 'hello',
            'data': {'key': 'value'},
            'items': [1, 2, 3],
            'number': 42,
        }
        return outputs.get(key, 'unknown')

    report = await isinstance_dataset.evaluate(mixed_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)


async def demo_duration():
    """Démo 5 : MaxDuration — rapide vs lent."""
    logger.info("=== DÉMO 5 : MaxDuration ===")

    async def timed_task(speed: str) -> str:
        if speed == 'slow':
            time.sleep(0.3)
        return f'{speed} done'

    report = await duration_dataset.evaluate(timed_task)
    report.print(include_input=True, include_output=True)


async def demo_confusion_matrix():
    """Démo 6 : ConfusionMatrixEvaluator."""
    logger.info("=== DÉMO 6 : ConfusionMatrix (report evaluator) ===")

    async def sentiment_task(text: str) -> str:
        """Classifieur imparfait pour montrer la confusion matrix."""
        mapping = {
            'I love this!': 'positive',
            'Amazing product': 'positive',
            'Terrible service': 'negative',
            'Waste of money': 'neutral',     # Erreur : negative → neutral
            'It is okay': 'neutral',
            'Nothing special': 'positive',   # Erreur : neutral → positive
        }
        return mapping.get(text, 'neutral')

    report = await confusion_dataset.evaluate(sentiment_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    if report.analyses:
        for analysis in report.analyses:
            logger.info(f"  Analyse experiment-wide : {analysis}")


async def demo_precision_recall():
    """Démo 7 : PrecisionRecallEvaluator."""
    logger.info("=== DÉMO 7 : PrecisionRecall (report evaluator) ===")

    async def spam_task(text: str) -> str:
        """Détecteur de spam imparfait."""
        spam_keywords = ['buy', 'free', 'offer']
        if any(kw in text.lower() for kw in spam_keywords):
            return 'spam'
        return 'ham'

    report = await pr_dataset.evaluate(spam_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    if report.analyses:
        for analysis in report.analyses:
            logger.info(f"  Analyse PR : {analysis}")


async def demo_combined():
    """Démo 8 : Combinaison layered (ATTENTION : coûte de l'argent)."""
    logger.info("=== DÉMO 8 : Combinaison layered (déterministe + LLM) ===")

    async def explain_task(question: str) -> str:
        return (
            'REST uses resource-based endpoints with HTTP methods. '
            'GraphQL uses a single query endpoint where clients specify '
            'exactly what data they need.'
        )

    report = await combined_dataset.evaluate(explain_task)
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
        include_durations=False,
    )


async def main():
    """Exécute toutes les démos built-in evaluators."""

    # Démos gratuites (pas d'appel LLM)
    await demo_equals_expected()
    await demo_equals()
    await demo_contains()
    await demo_isinstance()
    await demo_duration()
    await demo_confusion_matrix()
    await demo_precision_recall()

    # Démos coûteuses (appels LLM)
    logger.warning("Les démos suivantes font des appels LLM (coût réel)")
    await demo_combined()

    logger.success("Toutes les démos Built-in Evaluators sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
