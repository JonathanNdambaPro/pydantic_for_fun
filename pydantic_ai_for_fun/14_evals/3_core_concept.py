import asyncio
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EqualsExpected,
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    IsInstance,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CORE CONCEPTS — Concepts fondamentaux de Pydantic Evals

Pydantic Evals repose sur un modèle en 3 phases :
1. Définition (Dataset + Cases + Evaluators) — ce qu'on veut tester
2. Exécution (Experiment via dataset.evaluate) — lancer la task sur les cases
3. Résultats (EvaluationReport) — scores, assertions, labels, analyses

Analogie avec les tests unitaires :
- Case + Evaluator  ≈  test function
- Dataset           ≈  test suite
- Experiment        ≈  pytest run
- EvaluationReport  ≈  rapport de tests

Différence clé : les systèmes IA sont probabilistes, donc on a :
- Des scores quantitatifs (0.0 à 1.0)
- Des labels qualitatifs ("good", "acceptable", "poor")
- Des assertions pass/fail avec raisons explicatives

Ce fichier couvre les concepts non illustrés dans 1_evals.py et 2_quick_start.py :
- Case-specific evaluators (evaluators au niveau du case)
- Inputs typés complexes (BaseModel)
- Multiple evaluations (retourner un dict)
- EvaluationReason (raisons explicatives)
- evaluate_sync() vs evaluate()
- report.averages() et structure détaillée du rapport
- Comparaison de tasks avec averages
"""


# =====================================================================
# PARTIE 1 : Inputs typés — BaseModel et types complexes
# =====================================================================

# Les inputs peuvent être de n'importe quel type : str, int, dict,
# ou même un BaseModel Pydantic pour du type-safe.


class QuestionInput(BaseModel):
    """Input structuré pour une question avec contexte."""
    question: str
    max_words: int = 10
    language: str = 'fr'


# Dataset avec inputs typés BaseModel
typed_dataset = Dataset[QuestionInput, str, dict](
    name='typed_inputs',
    cases=[
        Case(
            name='capital_simple',
            inputs=QuestionInput(question='Capitale de la France ?', max_words=3),
            expected_output='Paris',
            metadata={'difficulty': 'easy'},
        ),
        Case(
            name='capital_complex',
            inputs=QuestionInput(
                question='Capitale du Myanmar ?',
                max_words=5,
                language='en',
            ),
            expected_output='Naypyidaw',
            metadata={'difficulty': 'hard'},
        ),
        Case(
            name='math_question',
            inputs=QuestionInput(question='Combien font 7 * 8 ?', max_words=3),
            expected_output='56',
            metadata={'difficulty': 'easy'},
        ),
    ],
    evaluators=[IsInstance(type_name='str')],
)


async def typed_task(inp: QuestionInput) -> str:
    """Task qui traite des inputs structurés."""
    answers = {
        'Capitale de la France ?': 'Paris',
        'Capitale du Myanmar ?': 'Naypyidaw',
        'Combien font 7 * 8 ?': '56',
    }
    answer = answers.get(inp.question, 'Unknown')
    # Respecte la contrainte max_words
    words = answer.split()[:inp.max_words]
    return ' '.join(words)


# =====================================================================
# PARTIE 2 : Case-specific evaluators — evaluators par case
# =====================================================================

# Les evaluators peuvent être définis au niveau du dataset (globaux)
# OU au niveau de chaque case (spécifiques). Les case-specific
# evaluators ne tournent que pour CE case.

case_specific_dataset = Dataset(
    name='case_specific_demo',
    cases=[
        Case(
            name='exact_match_needed',
            inputs='hello',
            expected_output='HELLO',
            evaluators=[
                # Cet evaluator ne tourne QUE pour ce case
                EqualsExpected(),
            ],
        ),
        Case(
            name='flexible_match',
            inputs='world',
            expected_output='WORLD',
            # Pas d'evaluator spécifique → seuls les dataset-level tournent
        ),
        Case(
            name='strict_check',
            inputs='test',
            expected_output='TEST',
            evaluators=[
                # Combine dataset-level + case-specific
                EqualsExpected(),
                IsInstance(type_name='str'),
            ],
        ),
    ],
    evaluators=[
        # Cet evaluator tourne pour TOUS les cases
        IsInstance(type_name='str'),
    ],
)


# =====================================================================
# PARTIE 3 : Types de retour des evaluators — bool, float, str
# =====================================================================

# Un evaluator peut retourner :
# - bool  → assertion (pass/fail)
# - float → score (0.0 à 1.0)
# - str   → label (catégorie)


@dataclass
class AssertionEvaluator(Evaluator[str, str]):
    """Retourne un bool — assertion pass/fail."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> bool:
        return ctx.output == ctx.expected_output


@dataclass
class ScoreEvaluator(Evaluator[str, str]):
    """Retourne un float — score de qualité (0.0 à 1.0)."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        if (
            isinstance(ctx.expected_output, str)
            and isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.7
        return 0.0


@dataclass
class LabelEvaluator(Evaluator[str, str]):
    """Retourne un str — label catégoriel."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> str:
        if ctx.output == ctx.expected_output:
            return 'exact'
        if (
            isinstance(ctx.expected_output, str)
            and isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 'partial'
        return 'wrong'


# =====================================================================
# PARTIE 4 : Multiple evaluations — retourner un dict
# =====================================================================

# Un evaluator peut retourner un dict pour produire plusieurs
# résultats en une seule évaluation. Les clés deviennent les noms
# des métriques dans le rapport.


@dataclass
class MultiCheckEvaluator(Evaluator[str, str]):
    """Retourne plusieurs résultats en un seul evaluator."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool | float | str]:
        output = ctx.output or ''
        return {
            'is_valid': isinstance(output, str) and len(output) > 0,  # assertion
            'length_score': min(len(output) / 20, 1.0),               # score
            'size_label': 'long' if len(output) > 50 else 'short',    # label
        }


# =====================================================================
# PARTIE 5 : EvaluationReason — raisons explicatives
# =====================================================================

# EvaluationReason ajoute un champ 'reason' à n'importe quel résultat.
# Les raisons apparaissent dans les rapports avec include_reasons=True.


@dataclass
class ReasonedEvaluator(Evaluator[str, str]):
    """Evaluator avec raisons explicatives."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> EvaluationReason:
        if ctx.output == ctx.expected_output:
            return EvaluationReason(
                value=True,
                reason='Match exact avec la sortie attendue',
            )
        if (
            isinstance(ctx.expected_output, str)
            and isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return EvaluationReason(
                value=0.7,
                reason=f'Match partiel : attendu {ctx.expected_output!r}, '
                       f'trouvé dans {ctx.output!r}',
            )
        return EvaluationReason(
            value=False,
            reason=f'Aucun match : attendu {ctx.expected_output!r}, '
                   f'obtenu {ctx.output!r}',
        )


# Dataset combinant tous les types d'evaluators
advanced_dataset = Dataset(
    name='advanced_evaluators',
    cases=[
        Case(
            name='perfect_answer',
            inputs='What is 2+2?',
            expected_output='4',
        ),
        Case(
            name='verbose_answer',
            inputs='Capital of France?',
            expected_output='Paris',
        ),
        Case(
            name='wrong_answer',
            inputs='Capital of Brazil?',
            expected_output='Brasília',
        ),
    ],
    evaluators=[
        AssertionEvaluator(),
        ScoreEvaluator(),
        LabelEvaluator(),
        MultiCheckEvaluator(),
        ReasonedEvaluator(),
    ],
)


# =====================================================================
# PARTIE 6 : Comparaison de tasks — averages et structure du rapport
# =====================================================================

# On peut exécuter le même dataset contre plusieurs tasks et
# comparer les résultats via report.averages().

comparison_dataset = Dataset(
    name='comparison_test',
    cases=[
        Case(inputs='hello', expected_output='HELLO'),
        Case(inputs='world', expected_output='WORLD'),
        Case(inputs='test', expected_output='TEST'),
    ],
    evaluators=[EqualsExpected()],
)


def task_v1(text: str) -> str:
    """V1 : transformation correcte."""
    return text.upper()


def task_v2(text: str) -> str:
    """V2 : ajoute un '!' → casse le EqualsExpected."""
    return text.upper() + '!'


def task_v3(text: str) -> str:
    """V3 : parfois correct, parfois non."""
    if text == 'hello':
        return 'HELLO'
    return text.lower()


# =====================================================================
# PARTIE 7 : EvaluatorContext — accès complet au contexte
# =====================================================================

# L'EvaluatorContext donne accès à tout : inputs, output,
# expected_output, metadata, duration, etc.


@dataclass
class ContextAwareEvaluator(Evaluator):
    """Evaluator qui utilise le contexte complet."""

    async def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        results = {}

        # Accès au nom du case
        results['has_name'] = ctx.name is not None

        # Accès aux metadata
        if ctx.metadata and isinstance(ctx.metadata, dict):
            difficulty = ctx.metadata.get('difficulty', 'unknown')
            results['difficulty'] = difficulty

        # Accès à la durée d'exécution
        results['fast_enough'] = ctx.duration < 1.0 if ctx.duration else True

        return results


# =====================================================================
# PARTIE 8 : Exécution
# =====================================================================


async def main():
    """Exécute les démos des core concepts."""

    # --- Démo 1 : Inputs typés BaseModel ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Inputs typés (BaseModel)")
    logger.info("=" * 60)
    report = await typed_dataset.evaluate(typed_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    # --- Démo 2 : Case-specific evaluators ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Case-specific evaluators")
    logger.info("=" * 60)

    async def upper_task(text: str) -> str:
        return text.upper()

    report = await case_specific_dataset.evaluate(upper_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    # --- Démo 3 : Evaluators avancés (bool, float, str, dict, reason) ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Evaluators avancés — tous les types de retour")
    logger.info("=" * 60)

    async def mixed_task(question: str) -> str:
        answers = {
            'What is 2+2?': '4',
            'Capital of France?': 'The capital is Paris, of course',
            'Capital of Brazil?': 'Rio de Janeiro',
        }
        return answers.get(question, 'Unknown')

    report = await advanced_dataset.evaluate(mixed_task)
    report.print(
        include_input=True,
        include_output=True,
        include_reasons=True,
        include_durations=False,
    )

    # --- Démo 4 : Comparaison de tasks avec averages ---
    logger.info("=" * 60)
    logger.info("DÉMO 4 : Comparaison de 3 tasks — averages()")
    logger.info("=" * 60)

    report_v1 = comparison_dataset.evaluate_sync(task_v1)
    report_v2 = comparison_dataset.evaluate_sync(task_v2)
    report_v3 = comparison_dataset.evaluate_sync(task_v3)

    for name, report in [('V1', report_v1), ('V2', report_v2), ('V3', report_v3)]:
        avg = report.averages()
        pass_rate = avg.assertions if avg and avg.assertions is not None else 0
        logger.info(f"  {name} pass rate : {pass_rate}")

    logger.success("V1 devrait être à 100%, V2 à 0%, V3 partiel")

    # --- Démo 5 : Structure détaillée du rapport ---
    logger.info("=" * 60)
    logger.info("DÉMO 5 : Structure détaillée du rapport")
    logger.info("=" * 60)

    report = await advanced_dataset.evaluate(mixed_task)

    for case_result in report.cases:
        logger.info(f"  Case '{case_result.name}' :")
        logger.info(f"    output     = {case_result.output!r}")
        logger.info(f"    scores     = {case_result.scores}")
        logger.info(f"    labels     = {case_result.labels}")
        logger.info(f"    assertions = {case_result.assertions}")
        logger.info(f"    duration   = {case_result.task_duration:.4f}s")

    # Moyennes globales
    avg = report.averages()
    if avg:
        logger.success(f"  Averages — assertions: {avg.assertions}, scores: {avg.scores}")

    # Échecs d'evaluators (s'il y en a)
    if report.failures:
        logger.warning(f"  {len(report.failures)} case(s) en échec")
    else:
        logger.success("  Aucun échec d'exécution")

    logger.success("Toutes les démos Core Concepts sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
