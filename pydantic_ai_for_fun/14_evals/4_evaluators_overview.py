import asyncio
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    Contains,
    EqualsExpected,
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    IsInstance,
    LLMJudge,
    MaxDuration,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
EVALUATORS OVERVIEW — Guide complet des evaluators Pydantic Evals

Les evaluators sont le coeur de Pydantic Evals. Ils analysent les
outputs des tasks et produisent des scores, labels ou assertions.

3 familles d'evaluators :
1. Déterministes (rapides, gratuits, fiables)
   → EqualsExpected, Contains, IsInstance, MaxDuration, HasMatchingSpan
2. LLM-as-a-Judge (flexibles, nuancés, coûteux)
   → LLMJudge avec rubric personnalisée
3. Custom (logique métier spécifique)
   → Hériter de Evaluator, sync ou async

3 types de retour :
- bool   → assertion (pass/fail, ✔/✗)
- float  → score (0.0 à 1.0, métrique de qualité)
- str    → label (catégorie : "good", "error", etc.)
- dict   → multiple résultats en un seul evaluator
- EvaluationReason → résultat + raison explicative

Concepts avancés couverts ici :
- Combinaison d'evaluators (layered evaluation)
- Golden datasets avec LLMJudge case-specific
- Evaluators sync vs async
- Gestion d'erreurs (EvaluatorFailure)
- ReportEvaluators (analyses experiment-wide : confusion matrix, etc.)
"""


# =====================================================================
# PARTIE 1 : Evaluators déterministes — quand utiliser lequel
# =====================================================================

# Les evaluators déterministes sont rapides, gratuits et reproductibles.
# Toujours les placer AVANT les LLM judges dans la liste.

# Tableau de décision :
# EqualsExpected  → match exact avec expected_output
# Contains        → sous-chaîne ou élément présent
# IsInstance      → validation de type
# MaxDuration     → seuil de performance (SLA)
# HasMatchingSpan → vérification comportementale (spans OpenTelemetry)

deterministic_dataset = Dataset(
    name='deterministic_checks',
    cases=[
        Case(
            name='format_check',
            inputs='hello world',
            expected_output='HELLO WORLD',
            metadata={'category': 'format'},
        ),
        Case(
            name='keyword_check',
            inputs='explain pydantic',
            expected_output='Pydantic is a data validation library',
            metadata={'category': 'content'},
        ),
    ],
    evaluators=[
        IsInstance(type_name='str'),
        EqualsExpected(),
        Contains(value='HELLO', case_sensitive=True),
        MaxDuration(seconds=1.0),
    ],
)


# =====================================================================
# PARTIE 2 : Custom evaluators domain-specific
# =====================================================================

# Les custom evaluators servent pour la logique métier spécifique
# que les built-in ne couvrent pas.


@dataclass
class HasKeyword(Evaluator):
    """Assertion : vérifie qu'un mot-clé est présent dans l'output."""

    keyword: str

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return self.keyword.lower() in str(ctx.output).lower()


@dataclass
class ConfidenceScore(Evaluator):
    """Score : note la confiance basée sur la longueur et la précision."""

    def evaluate(self, ctx: EvaluatorContext) -> float:
        output = str(ctx.output)
        expected = str(ctx.expected_output) if ctx.expected_output else ''

        if output == expected:
            return 1.0
        if expected.lower() in output.lower():
            # Pénalise la verbosité
            ratio = len(expected) / max(len(output), 1)
            return round(min(ratio + 0.3, 0.95), 2)
        return 0.0


@dataclass
class SentimentClassifier(Evaluator):
    """Label : classifie le sentiment de l'output."""

    def evaluate(self, ctx: EvaluatorContext) -> str:
        output = str(ctx.output).lower()
        if any(w in output for w in ('error', 'fail', 'wrong', 'bad')):
            return 'negative'
        if any(w in output for w in ('success', 'correct', 'good', 'great')):
            return 'positive'
        return 'neutral'


@dataclass
class ComprehensiveCheck(Evaluator):
    """Multiple résultats : assertion + score + label en un seul evaluator."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        output = str(ctx.output)
        expected = str(ctx.expected_output) if ctx.expected_output else ''

        is_correct = output == expected
        length_ratio = min(len(output) / max(len(expected), 1), 2.0)

        return {
            'correct': is_correct,
            'precision': 1.0 if is_correct else max(0, 1.0 - abs(length_ratio - 1.0)),
            'quality': 'exact' if is_correct else ('partial' if expected.lower() in output.lower() else 'wrong'),
        }


# =====================================================================
# PARTIE 3 : Gestion d'erreurs — EvaluatorFailure
# =====================================================================

# Si un evaluator lève une exception, elle est capturée comme
# EvaluatorFailure dans report.cases[i].evaluator_failures.
# Le reste des evaluators continue de tourner normalement.


@dataclass
class RiskyEvaluator(Evaluator):
    """Evaluator qui peut échouer — l'exception est capturée, pas propagée."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = str(ctx.output)
        if 'CRASH' in output:
            raise ValueError(f"Output invalide détecté : {output!r}")
        return True


@dataclass
class ValidJSON(Evaluator):
    """Evaluator domain-specific : valide que l'output est du JSON."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        import json

        try:
            json.loads(str(ctx.output))
            return EvaluationReason(value=True, reason='JSON valide')
        except json.JSONDecodeError as e:
            return EvaluationReason(value=False, reason=f'JSON invalide : {e}')


error_handling_dataset = Dataset(
    name='error_handling',
    cases=[
        Case(name='normal', inputs='test', expected_output='result'),
        Case(name='crash_case', inputs='crash', expected_output='CRASH'),
        Case(name='json_valid', inputs='json', expected_output='{"key": "value"}'),
        Case(name='json_invalid', inputs='bad_json', expected_output='not json {'),
    ],
    evaluators=[
        RiskyEvaluator(),
        ValidJSON(),
    ],
)


# =====================================================================
# PARTIE 4 : Async vs Sync evaluators
# =====================================================================

# Pydantic Evals gère les deux automatiquement.
# Utiliser async pour : appels API, I/O, requêtes DB, appels LLM.


@dataclass
class SyncEvaluator(Evaluator):
    """Evaluator synchrone — pour les checks rapides sans I/O."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) > 0


async def check_external_api(output: str) -> bool:
    """Simule un appel API externe."""
    await asyncio.sleep(0.01)
    return 'error' not in output.lower()


@dataclass
class AsyncEvaluator(Evaluator):
    """Evaluator asynchrone — pour les checks avec I/O."""

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        is_valid = await check_external_api(str(ctx.output))
        return EvaluationReason(
            value=is_valid,
            reason='API externe OK' if is_valid else 'API externe a rejeté l\'output',
        )


# =====================================================================
# PARTIE 5 : Layered evaluation — combiner les evaluators
# =====================================================================

# Pattern recommandé : déterministes rapides d'abord, LLM judges après.
# Les checks rapides filtrent les cas évidents avant de dépenser
# de l'argent sur un LLM judge.

layered_dataset = Dataset(
    name='layered_evaluation',
    cases=[
        Case(
            name='support_greeting',
            inputs='Say hello to the customer',
            expected_output='Hello! How can I help you today?',
        ),
        Case(
            name='support_technical',
            inputs='Explain how to reset a password',
            expected_output='To reset your password, go to Settings > Security > Reset Password.',
        ),
    ],
    evaluators=[
        # Couche 1 : checks rapides et gratuits
        IsInstance(type_name='str'),
        MaxDuration(seconds=2.0),

        # Couche 2 : checks custom déterministes
        HasKeyword(keyword='help'),
        ConfidenceScore(),
        SentimentClassifier(),
        ComprehensiveCheck(),

        # Couche 3 : LLM judge (coûteux, en dernier)
        LLMJudge(
            rubric='Response is helpful, professional, and addresses the user request directly',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 6 : Golden datasets — LLMJudge case-specific
# =====================================================================

# Pattern puissant : chaque case a sa propre rubric LLMJudge
# qui décrit ce que "bon" signifie pour CE scénario précis.
# Permet de capturer la nuance sans expected_output exact.

golden_dataset = Dataset(
    name='customer_support_golden',
    cases=[
        Case(
            name='refund_request',
            inputs={'query': 'I want my money back', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Acknowledge the refund request empathetically
                    2. Ask for the reason for the refund
                    3. Mention the 30-day refund policy
                    4. NOT process the refund immediately (needs manager approval)
                    """,
                    include_input=True,
                    model='gateway/anthropic:claude-sonnet-4-6',
                ),
            ],
        ),
        Case(
            name='shipping_question',
            inputs={'query': 'Where is my order?', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Confirm the order number
                    2. Provide tracking information
                    3. Give estimated delivery date
                    4. Be brief and factual (not overly apologetic)
                    """,
                    include_input=True,
                    model='gateway/anthropic:claude-sonnet-4-6',
                ),
            ],
        ),
        Case(
            name='angry_customer',
            inputs={'query': 'This is completely unacceptable!', 'order_id': '12345'},
            evaluators=[
                LLMJudge(
                    rubric="""
                    Response should:
                    1. Prioritize de-escalation with empathy
                    2. Avoid being defensive
                    3. Offer concrete next steps
                    4. Use phrases like "I understand" and "Let me help"
                    """,
                    include_input=True,
                    model='gateway/anthropic:claude-sonnet-4-6',
                ),
            ],
        ),
    ],
    evaluators=[
        # Dataset-level : tourne pour tous les cases
        IsInstance(type_name='str'),
    ],
)


# =====================================================================
# PARTIE 7 : ReportEvaluators — analyses experiment-wide
# =====================================================================

# Les ReportEvaluators tournent UNE FOIS après tous les cases,
# sur l'ensemble des résultats. Utiles pour :
# - Confusion matrices (classification)
# - Précision/rappel
# - Métriques scalaires globales
# - Tableaux de synthèse

classification_dataset = Dataset(
    name='animal_classifier',
    cases=[
        Case(name='cat_meow', inputs='meow', expected_output='cat'),
        Case(name='dog_woof', inputs='woof', expected_output='dog'),
        Case(name='cat_purr', inputs='purr', expected_output='cat'),
        Case(name='dog_bark', inputs='bark', expected_output='dog'),
        Case(name='bird_tweet', inputs='tweet', expected_output='bird'),
        Case(name='cat_hiss', inputs='hiss', expected_output='cat'),
    ],
    evaluators=[
        EqualsExpected(),
    ],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
        ),
    ],
)


# =====================================================================
# PARTIE 8 : Exécution
# =====================================================================


async def main():
    """Exécute les démos de l'overview evaluators."""

    # --- Démo 1 : Evaluators déterministes ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Evaluators déterministes")
    logger.info("=" * 60)

    async def format_task(text: str) -> str:
        return text.upper()

    report = await deterministic_dataset.evaluate(format_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    # --- Démo 2 : Gestion d'erreurs (EvaluatorFailure) ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Gestion d'erreurs — EvaluatorFailure")
    logger.info("=" * 60)

    async def echo_task(text: str) -> str:
        if text == 'crash':
            return 'CRASH'
        if text == 'json':
            return '{"key": "value"}'
        if text == 'bad_json':
            return 'not json {'
        return 'result'

    report = await error_handling_dataset.evaluate(echo_task)
    report.print(include_input=True, include_output=True, include_reasons=True)

    # Inspecte les failures
    for case_result in report.cases:
        if case_result.evaluator_failures:
            for failure in case_result.evaluator_failures:
                logger.warning(
                    f"  Evaluator failure sur '{case_result.name}' : {failure}"
                )

    # --- Démo 3 : Layered evaluation (déterministe + custom + LLM) ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Layered evaluation (ATTENTION : coûte de l'argent)")
    logger.info("=" * 60)

    async def support_task(prompt: str) -> str:
        responses = {
            'Say hello to the customer': 'Hello! How can I help you today?',
            'Explain how to reset a password': (
                'To reset your password, go to Settings > Security > Reset Password. '
                'Let me know if you need more help!'
            ),
        }
        return responses.get(prompt, 'I can help you with that.')

    report = await layered_dataset.evaluate(support_task)
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
        include_durations=False,
    )

    # --- Démo 4 : ReportEvaluator — Confusion Matrix ---
    logger.info("=" * 60)
    logger.info("DÉMO 4 : ReportEvaluator — Confusion Matrix")
    logger.info("=" * 60)

    async def classifier_task(sound: str) -> str:
        """Classifieur imparfait — se trompe parfois."""
        mapping = {
            'meow': 'cat',
            'woof': 'dog',
            'purr': 'cat',
            'bark': 'dog',
            'tweet': 'cat',   # Erreur volontaire : bird → cat
            'hiss': 'dog',    # Erreur volontaire : cat → dog
        }
        return mapping.get(sound, 'unknown')

    report = await classification_dataset.evaluate(classifier_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    # Affiche les analyses experiment-wide (confusion matrix)
    if report.analyses:
        for analysis in report.analyses:
            logger.info(f"  Analyse : {analysis}")

    # --- Démo 5 : Async vs Sync evaluators ---
    logger.info("=" * 60)
    logger.info("DÉMO 5 : Async vs Sync evaluators")
    logger.info("=" * 60)

    async_dataset = Dataset(
        name='async_vs_sync',
        cases=[
            Case(name='ok', inputs='test', expected_output='result'),
            Case(name='error_case', inputs='error', expected_output='error output'),
        ],
        evaluators=[
            SyncEvaluator(),
            AsyncEvaluator(),
        ],
    )

    async def simple_task(text: str) -> str:
        return f'{text} output'

    report = await async_dataset.evaluate(simple_task)
    report.print(include_input=True, include_output=True, include_reasons=True)

    logger.success("Toutes les démos Evaluators Overview sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
