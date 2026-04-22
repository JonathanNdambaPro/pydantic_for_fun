import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    EqualsExpected,
    Evaluator,
    EvaluatorContext,
    KolmogorovSmirnovEvaluator,
    PrecisionRecallEvaluator,
    ReportEvaluator,
    ReportEvaluatorContext,
    ROCAUCEvaluator,
)
from pydantic_evals.reporting.analyses import (
    ReportAnalysis,
    ScalarResult,
    TableResult,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
REPORT EVALUATORS — Analyses experiment-wide sur l'ensemble des résultats

Les report evaluators sont différents des case evaluators :
- Case evaluators → tournent 1 fois PAR CASE
- Report evaluators → tournent 1 fois PAR EXPERIMENT, après tous les cases

Pipeline d'exécution :
  Cases exécutés → Case evaluators → Report evaluators → Rapport final

Report evaluators built-in :
- ConfusionMatrixEvaluator → matrice de confusion (classification)
- PrecisionRecallEvaluator → courbe PR + AUC
- ROCAUCEvaluator → courbe ROC + AUC
- KolmogorovSmirnovEvaluator → plot KS + statistique KS

Types de résultats (ReportAnalysis) :
- ScalarResult → métrique unique (accuracy, F1, AUC)
- TableResult → tableau de données (per-class metrics)
- ConfusionMatrix → matrice de confusion
- PrecisionRecall → courbe précision/rappel
- LinePlot → graphique XY générique (ROC, KS, calibration)

Les analyses sont automatiquement visualisées dans Logfire :
- Confusion matrices → heatmaps
- Courbes PR/ROC/KS → line charts
- Scalars → valeurs labellisées
- Tables → tableaux formatés
"""


# =====================================================================
# PARTIE 1 : ConfusionMatrixEvaluator — matrice de confusion
# =====================================================================

# Compare predicted vs expected sur toutes les cases.
# Sources possibles : output, expected_output, metadata, labels.

confusion_dataset = Dataset(
    name='animal_classifier',
    cases=[
        Case(name='cat_meow', inputs='The cat goes meow', expected_output='cat'),
        Case(name='dog_bark', inputs='The dog barks loudly', expected_output='dog'),
        Case(name='cat_purr', inputs='The cat purrs softly', expected_output='cat'),
        Case(name='dog_woof', inputs='Woof woof says the dog', expected_output='dog'),
        Case(name='bird_chirp', inputs='The bird chirps', expected_output='bird'),
        Case(name='bird_tweet', inputs='Tweet tweet in the tree', expected_output='bird'),
        Case(name='cat_hiss', inputs='The cat hisses', expected_output='cat'),
        Case(name='dog_howl', inputs='The dog howls at night', expected_output='dog'),
    ],
    evaluators=[EqualsExpected()],
    report_evaluators=[
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
            title='Animal Classification',
        ),
    ],
)


def animal_classifier(text: str) -> str:
    """Classifieur imparfait — se trompe sur certains sons."""
    text = text.lower()
    if 'meow' in text or 'purr' in text:
        return 'cat'
    if 'bark' in text or 'woof' in text:
        return 'dog'
    if 'chirp' in text:
        return 'bird'
    # Erreurs volontaires :
    if 'hiss' in text:
        return 'dog'    # cat → dog
    if 'tweet' in text:
        return 'cat'    # bird → cat
    if 'howl' in text:
        return 'cat'    # dog → cat
    return 'unknown'


# =====================================================================
# PARTIE 2 : Confusion matrix avec labels d'evaluator
# =====================================================================

# On peut utiliser les labels produits par un case evaluator
# comme source pour la confusion matrix.


@dataclass
class SentimentLabeler(Evaluator):
    """Produit un label 'predicted_sentiment' basé sur l'output."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, str]:
        output = str(ctx.output).lower()
        if any(w in output for w in ('great', 'love', 'excellent', 'happy')):
            sentiment = 'positive'
        elif any(w in output for w in ('bad', 'hate', 'terrible', 'awful')):
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {'predicted_sentiment': sentiment}


labels_confusion_dataset = Dataset(
    name='sentiment_from_labels',
    cases=[
        Case(name='pos_1', inputs='review_1', expected_output='positive'),
        Case(name='pos_2', inputs='review_2', expected_output='positive'),
        Case(name='neg_1', inputs='review_3', expected_output='negative'),
        Case(name='neg_2', inputs='review_4', expected_output='negative'),
        Case(name='neu_1', inputs='review_5', expected_output='neutral'),
    ],
    evaluators=[SentimentLabeler()],
    report_evaluators=[
        # Utilise le label 'predicted_sentiment' au lieu de l'output direct
        ConfusionMatrixEvaluator(
            predicted_from='labels',
            predicted_key='predicted_sentiment',
            expected_from='expected_output',
            title='Sentiment (from labels)',
        ),
    ],
)


# =====================================================================
# PARTIE 3 : PrecisionRecallEvaluator — courbe PR + AUC
# =====================================================================

# Calcule une courbe précision/rappel à partir de scores numériques
# et de labels binaires ground-truth.
# L'AUC est calculée à pleine résolution puis downsampleé pour l'affichage.


@dataclass
class ConfidenceEvaluator(Evaluator):
    """Produit un score de confiance et une assertion is_correct."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, Any]:
        is_correct = ctx.output == ctx.expected_output
        # Simule un score de confiance variable
        confidence = 0.95 if is_correct else 0.3
        return {
            'confidence': confidence,
            'is_correct': is_correct,
        }


pr_dataset = Dataset(
    name='precision_recall_demo',
    cases=[
        Case(name='cat_1', inputs='meow', expected_output='cat'),
        Case(name='cat_2', inputs='purr', expected_output='cat'),
        Case(name='dog_1', inputs='bark', expected_output='dog'),
        Case(name='dog_2', inputs='woof', expected_output='dog'),
        Case(name='bird_1', inputs='chirp', expected_output='bird'),
        Case(name='tricky', inputs='hiss', expected_output='cat'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        PrecisionRecallEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
            title='Classifier PR Curve',
            n_thresholds=50,
        ),
    ],
)


# =====================================================================
# PARTIE 4 : ROCAUCEvaluator — courbe ROC + AUC
# =====================================================================

# Courbe ROC : True Positive Rate vs False Positive Rate.
# Inclut une diagonale "Random" en pointillé comme baseline.

roc_dataset = Dataset(
    name='roc_auc_demo',
    cases=[
        Case(name='spam_1', inputs='Buy now!!!', expected_output='spam'),
        Case(name='spam_2', inputs='Free money click here', expected_output='spam'),
        Case(name='spam_3', inputs='Win a prize', expected_output='spam'),
        Case(name='ham_1', inputs='Meeting at 3pm tomorrow', expected_output='ham'),
        Case(name='ham_2', inputs='Project update attached', expected_output='ham'),
        Case(name='ham_3', inputs='Lunch plans?', expected_output='ham'),
        Case(name='edge', inputs='Special offer from your boss', expected_output='ham'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        ROCAUCEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
            title='Spam Detector ROC',
        ),
    ],
)


# =====================================================================
# PARTIE 5 : KolmogorovSmirnovEvaluator — plot KS + statistique
# =====================================================================

# Le KS plot montre les CDFs empiriques des scores pour les
# positifs et négatifs. La statistique KS est la distance maximale
# entre les deux CDFs — plus élevée = meilleure séparation.

ks_dataset = Dataset(
    name='ks_demo',
    cases=[
        Case(name='correct_1', inputs='easy_1', expected_output='A'),
        Case(name='correct_2', inputs='easy_2', expected_output='B'),
        Case(name='correct_3', inputs='easy_3', expected_output='C'),
        Case(name='wrong_1', inputs='hard_1', expected_output='X'),
        Case(name='wrong_2', inputs='hard_2', expected_output='Y'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        KolmogorovSmirnovEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
            title='Score Distribution KS',
        ),
    ],
)


# =====================================================================
# PARTIE 6 : Custom ReportEvaluator — ScalarResult
# =====================================================================

# Hériter de ReportEvaluator et implémenter evaluate().
# Accès au rapport complet via ctx.report.cases.


@dataclass
class AccuracyEvaluator(ReportEvaluator):
    """Calcule l'accuracy globale comme métrique scalaire."""

    def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
        cases = ctx.report.cases
        if not cases:
            return ScalarResult(title='Accuracy', value=0.0, unit='%')
        correct = sum(1 for c in cases if c.output == c.expected_output)
        accuracy = correct / len(cases) * 100
        return ScalarResult(
            title='Accuracy',
            value=round(accuracy, 1),
            unit='%',
            description='Percentage of correctly classified cases.',
        )


# =====================================================================
# PARTIE 7 : Custom ReportEvaluator — multiple analyses (list)
# =====================================================================

# Un seul report evaluator peut retourner plusieurs analyses
# en retournant une list[ReportAnalysis].


@dataclass
class ClassificationSummary(ReportEvaluator):
    """Produit accuracy (scalar) + per-class metrics (table)."""

    def evaluate(self, ctx: ReportEvaluatorContext) -> list[ReportAnalysis]:
        cases = ctx.report.cases
        if not cases:
            return []

        labels = sorted({str(c.expected_output) for c in cases if c.expected_output})

        # Scalar : accuracy globale
        correct = sum(1 for c in cases if c.output == c.expected_output)
        accuracy = ScalarResult(
            title='Accuracy',
            value=round(correct / len(cases) * 100, 1),
            unit='%',
        )

        # Table : precision, recall, F1 par classe
        rows: list[list[str | int | float | bool | None]] = []
        for label in labels:
            tp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) == label)
            fp = sum(1 for c in cases if str(c.output) == label and str(c.expected_output) != label)
            fn = sum(1 for c in cases if str(c.output) != label and str(c.expected_output) == label)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append([label, round(p, 3), round(r, 3), round(f1, 3)])

        table = TableResult(
            title='Per-Class Metrics',
            columns=['Class', 'Precision', 'Recall', 'F1'],
            rows=rows,
            description='Precision, recall, and F1 per class.',
        )

        return [accuracy, table]


# =====================================================================
# PARTIE 8 : Exemple complet — tous les report evaluators combinés
# =====================================================================

# Combine case evaluators + tous les report evaluators built-in
# + custom report evaluators.

full_dataset = Dataset(
    name='full_classification',
    cases=[
        Case(inputs='The cat meows', expected_output='cat'),
        Case(inputs='The dog barks', expected_output='dog'),
        Case(inputs='A bird chirps', expected_output='bird'),
        Case(inputs='The cat purrs', expected_output='cat'),
        Case(inputs='Woof woof', expected_output='dog'),
        Case(inputs='Tweet tweet', expected_output='bird'),
        Case(inputs='The cat hisses', expected_output='cat'),
        Case(inputs='The dog howls', expected_output='dog'),
    ],
    evaluators=[ConfidenceEvaluator()],
    report_evaluators=[
        # Built-in
        ConfusionMatrixEvaluator(
            predicted_from='output',
            expected_from='expected_output',
            title='Animal Classification',
        ),
        PrecisionRecallEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        ROCAUCEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        KolmogorovSmirnovEvaluator(
            score_from='scores',
            score_key='confidence',
            positive_from='assertions',
            positive_key='is_correct',
        ),
        # Custom
        AccuracyEvaluator(),
        ClassificationSummary(),
    ],
)


# =====================================================================
# PARTIE 9 : Exécution
# =====================================================================


async def demo_confusion_matrix():
    """Démo 1 : ConfusionMatrixEvaluator."""
    logger.info("=== DÉMO 1 : Confusion Matrix ===")
    report = confusion_dataset.evaluate_sync(animal_classifier)
    report.print(include_input=True, include_output=True, include_durations=False)

    for analysis in report.analyses:
        logger.info(f"  {analysis.type}: {analysis.title}")


async def demo_labels_confusion():
    """Démo 2 : Confusion matrix depuis les labels d'evaluator."""
    logger.info("=== DÉMO 2 : Confusion Matrix (from labels) ===")

    async def review_task(review_id: str) -> str:
        reviews = {
            'review_1': 'I love this product, it is great!',
            'review_2': 'Excellent quality, very happy',
            'review_3': 'Terrible experience, awful service',
            'review_4': 'Bad product, I hate it',
            'review_5': 'It is okay, nothing special',
        }
        return reviews.get(review_id, 'No review found')

    report = await labels_confusion_dataset.evaluate(review_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    for analysis in report.analyses:
        logger.info(f"  {analysis.type}: {analysis.title}")


async def demo_precision_recall():
    """Démo 3 : PrecisionRecallEvaluator."""
    logger.info("=== DÉMO 3 : Precision-Recall Curve + AUC ===")
    report = pr_dataset.evaluate_sync(animal_classifier)
    report.print(include_input=True, include_output=True, include_durations=False)

    for analysis in report.analyses:
        logger.info(f"  {analysis.type}: {analysis.title}")


async def demo_roc_auc():
    """Démo 4 : ROCAUCEvaluator."""
    logger.info("=== DÉMO 4 : ROC Curve + AUC ===")

    def spam_detector(text: str) -> str:
        spam_words = ['buy', 'free', 'win', 'prize', 'offer']
        if any(w in text.lower() for w in spam_words):
            return 'spam'
        return 'ham'

    report = roc_dataset.evaluate_sync(spam_detector)
    report.print(include_input=True, include_output=True, include_durations=False)

    for analysis in report.analyses:
        logger.info(f"  {analysis.type}: {analysis.title}")


async def demo_ks():
    """Démo 5 : KolmogorovSmirnovEvaluator."""
    logger.info("=== DÉMO 5 : KS Plot + Statistique ===")

    def simple_classifier(text: str) -> str:
        mapping = {
            'easy_1': 'A', 'easy_2': 'B', 'easy_3': 'C',
            'hard_1': 'Z', 'hard_2': 'W',  # Erreurs volontaires
        }
        return mapping.get(text, 'unknown')

    report = ks_dataset.evaluate_sync(simple_classifier)
    report.print(include_input=True, include_output=True, include_durations=False)

    for analysis in report.analyses:
        logger.info(f"  {analysis.type}: {analysis.title}")


async def demo_full_example():
    """Démo 6 : Exemple complet — tous les report evaluators."""
    logger.info("=== DÉMO 6 : Exemple complet (tous report evaluators) ===")
    report = full_dataset.evaluate_sync(animal_classifier)
    report.print(include_input=True, include_output=True, include_durations=False)

    # Accès programmatique aux analyses
    logger.info("  Analyses produites :")
    for analysis in report.analyses:
        logger.info(f"    {analysis.type}: {analysis.title}")

    logger.success("  Toutes les analyses sont dans report.analyses")


async def main():
    """Exécute toutes les démos report evaluators."""
    await demo_confusion_matrix()
    await demo_labels_confusion()
    await demo_precision_recall()
    await demo_roc_auc()
    await demo_ks()
    await demo_full_example()

    logger.success("Toutes les démos Report Evaluators sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
