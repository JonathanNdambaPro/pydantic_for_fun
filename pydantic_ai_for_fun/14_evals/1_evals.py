import asyncio
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
EVALS — Framework d'évaluation systématique pour systèmes IA avec Pydantic Evals

Pydantic Evals est un framework d'évaluation puissant pour tester et évaluer
des systèmes IA, des simples appels LLM aux applications multi-agents complexes.

Philosophie : approche "code-first" où tous les composants d'évaluation sont
définis en Python. Contrairement aux plateformes web, on écrit et exécute
les evals en code, puis on visualise les résultats en terminal ou dans Logfire.

Modèle de données :
- Dataset : collection de Cases pour évaluer une tâche spécifique
- Case : scénario de test avec inputs, output attendu, metadata et evaluators
- Experiment : exécution d'un Dataset contre une Task, produit un rapport
- Evaluator : analyse et note les résultats (déterministe ou LLM-based)
- Task : la fonction IA à évaluer

Analogie avec les tests unitaires :
- Cases + Evaluators ≈ tests unitaires individuels
- Datasets ≈ suites de tests (test suites)
- Experiments ≈ exécution de pytest avec rapport de résultats

La différence clé : les systèmes IA sont probabilistes, donc les scores
peuvent être qualitatifs/catégoriels plutôt que simple pass/fail.
"""


# =====================================================================
# PARTIE 1 : Dataset et Cases — La base de toute évaluation
# =====================================================================

# Un Case définit un scénario de test : inputs, output attendu,
# metadata optionnelle et evaluators spécifiques au cas.

logger.info("Création du dataset de quiz géographie...")

case_france = Case(
    name='capital_france',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy', 'region': 'europe'},
)

case_japan = Case(
    name='capital_japan',
    inputs='What is the capital of Japan?',
    expected_output='Tokyo',
    metadata={'difficulty': 'easy', 'region': 'asia'},
)

case_brazil = Case(
    name='capital_brazil',
    inputs='What is the capital of Brazil?',
    expected_output='Brasília',
    metadata={'difficulty': 'medium', 'region': 'south_america'},
)

case_myanmar = Case(
    name='capital_myanmar',
    inputs='What is the capital of Myanmar?',
    expected_output='Naypyidaw',
    metadata={'difficulty': 'hard', 'region': 'asia'},
)

# Le Dataset regroupe les cases et peut définir des evaluators globaux
simple_dataset = Dataset(
    name='capital_quiz',
    cases=[case_france, case_japan, case_brazil, case_myanmar],
)

logger.success(f"Dataset '{simple_dataset.name}' créé avec {len(simple_dataset.cases)} cases")


# =====================================================================
# PARTIE 2 : Evaluators — Analyser et noter les résultats
# =====================================================================

# Les evaluators peuvent être déterministes (regex, type check)
# ou basés sur un LLM (accuracy, hallucinations, etc.).
# Ici on utilise des evaluators déterministes, moins chers et plus simples.

# Evaluator built-in : vérifie que l'output est du bon type
simple_dataset.add_evaluator(IsInstance(type_name='str'))


# Evaluator custom : compare l'output avec la réponse attendue
@dataclass
class ExactMatchEvaluator(Evaluator[str, str]):
    """Évalue si la réponse correspond exactement à l'output attendu."""

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and isinstance(ctx.expected_output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


# Evaluator custom : pénalise les réponses trop longues
@dataclass
class ConcisenessEvaluator(Evaluator[str, str]):
    """Évalue la concision de la réponse — pénalise les réponses verbeuses."""

    max_words: int = 5

    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if not isinstance(ctx.output, str):
            return 0.0
        word_count = len(ctx.output.split())
        if word_count <= self.max_words:
            return 1.0
        elif word_count <= self.max_words * 2:
            return 0.5
        else:
            return 0.2


simple_dataset.add_evaluator(ExactMatchEvaluator())
simple_dataset.add_evaluator(ConcisenessEvaluator(max_words=3))

logger.info("Evaluators ajoutés : IsInstance, ExactMatch, Conciseness")


# =====================================================================
# PARTIE 3 : Task — La fonction à évaluer
# =====================================================================

# La task est la fonction IA qu'on veut évaluer.
# Ici on simule différentes qualités de réponses pour illustrer les scores.


async def perfect_task(question: str) -> str:
    """Task parfaite : renvoie toujours la bonne réponse."""
    answers = {
        'What is the capital of France?': 'Paris',
        'What is the capital of Japan?': 'Tokyo',
        'What is the capital of Brazil?': 'Brasília',
        'What is the capital of Myanmar?': 'Naypyidaw',
    }
    return answers.get(question, 'Unknown')


async def verbose_task(question: str) -> str:
    """Task verbeuse : bonne réponse mais trop de mots."""
    answers = {
        'What is the capital of France?': 'The capital of France is Paris, a beautiful city',
        'What is the capital of Japan?': 'Tokyo is the capital city of Japan',
        'What is the capital of Brazil?': 'Brasília is the capital of Brazil since 1960',
        'What is the capital of Myanmar?': 'The capital of Myanmar is Naypyidaw',
    }
    return answers.get(question, 'I do not know the answer to that question')


async def bad_task(question: str) -> str:
    """Task médiocre : se trompe sur les questions difficiles."""
    answers = {
        'What is the capital of France?': 'Paris',
        'What is the capital of Japan?': 'Tokyo',
        'What is the capital of Brazil?': 'Rio de Janeiro',
        'What is the capital of Myanmar?': 'Yangon',
    }
    return answers.get(question, 'Unknown')


# =====================================================================
# PARTIE 4 : Experiment — Exécuter et comparer les résultats
# =====================================================================

# Un Experiment exécute toutes les cases d'un dataset contre une task
# et produit un rapport avec scores et assertions.


async def run_experiments():
    """Exécute les 3 tasks contre le dataset et affiche les rapports."""

    # --- Expérience 1 : Task parfaite ---
    logger.info("Exécution de l'expérience 1 : perfect_task")
    report_perfect = await simple_dataset.evaluate(perfect_task)
    report_perfect.print(
        include_input=True,
        include_output=True,
        include_durations=False,
    )

    # --- Expérience 2 : Task verbeuse ---
    logger.info("Exécution de l'expérience 2 : verbose_task")
    report_verbose = await simple_dataset.evaluate(verbose_task)
    report_verbose.print(
        include_input=True,
        include_output=True,
        include_durations=False,
    )

    # --- Expérience 3 : Task médiocre ---
    logger.info("Exécution de l'expérience 3 : bad_task")
    report_bad = await simple_dataset.evaluate(bad_task)
    report_bad.print(
        include_input=True,
        include_output=True,
        include_durations=False,
    )

    return report_perfect, report_verbose, report_bad


# =====================================================================
# PARTIE 5 : Dataset déclaratif — Evaluators dans le constructeur
# =====================================================================

# On peut aussi passer les evaluators directement au Dataset
# pour une approche plus déclarative et concise.

declarative_dataset = Dataset(
    name='math_quiz',
    cases=[
        Case(
            name='addition',
            inputs='What is 2 + 2?',
            expected_output='4',
            metadata={'operation': 'addition'},
        ),
        Case(
            name='multiplication',
            inputs='What is 7 * 8?',
            expected_output='56',
            metadata={'operation': 'multiplication'},
        ),
        Case(
            name='square_root',
            inputs='What is the square root of 144?',
            expected_output='12',
            metadata={'operation': 'square_root'},
        ),
    ],
    evaluators=[
        IsInstance(type_name='str'),
        ExactMatchEvaluator(),
        ConcisenessEvaluator(max_words=3),
    ],
)


async def math_task(question: str) -> str:
    """Task mathématique simple."""
    answers = {
        'What is 2 + 2?': '4',
        'What is 7 * 8?': '56',
        'What is the square root of 144?': '12',
    }
    return answers.get(question, 'Unknown')


# =====================================================================
# PARTIE 6 : Exécution — Démos des évaluations
# =====================================================================


async def main():
    """Exécute les démos d'évaluation."""

    # --- Démo 1 : Comparaison de 3 tasks sur le même dataset ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Comparaison de 3 tasks (perfect, verbose, bad)")
    logger.info("=" * 60)
    report_perfect, report_verbose, report_bad = await run_experiments()

    logger.success("Comparaison terminée — observez les scores :")
    logger.info("  perfect_task : scores élevés partout")
    logger.info("  verbose_task : ExactMatch partiel, Conciseness faible")
    logger.info("  bad_task     : ExactMatch 0 sur les questions difficiles")

    # --- Démo 2 : Dataset déclaratif avec evaluators inline ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Dataset déclaratif (math_quiz)")
    logger.info("=" * 60)
    report_math = await declarative_dataset.evaluate(math_task)
    report_math.print(
        include_input=True,
        include_output=True,
        include_durations=False,
    )
    logger.success("Dataset déclaratif évalué avec succès !")

    # --- Démo 3 : Accès programmatique aux résultats ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Accès programmatique aux résultats")
    logger.info("=" * 60)
    for case_result in report_math.cases:
        logger.info(
            f"  Case '{case_result.name}' : "
            f"output='{case_result.output}' | "
            f"scores={case_result.scores}"
        )

    logger.success("Toutes les démos d'évaluation sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
