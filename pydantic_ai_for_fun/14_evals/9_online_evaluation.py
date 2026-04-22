import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent
from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
    LLMJudge,
)
from pydantic_evals.online import (
    OnlineEvalConfig,
    OnlineEvaluator,
    SamplingContext,
    disable_evaluation,
    evaluate,
    run_evaluators,
    wait_for_evaluations,
)
from pydantic_evals.online_capability import OnlineEvaluation
from pydantic_evals.otel.span_tree import SpanTree

load_dotenv()
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
ONLINE EVALUATION — Évaluation en production, en continu et en background

L'évaluation offline (Dataset.evaluate) teste AVANT le déploiement.
L'évaluation online évalue CHAQUE APPEL en production (ou un échantillon).

Pipeline :
  Appel de fonction → résultat retourné au caller immédiatement
                    → evaluators lancés en background (non bloquants)
                    → résultats émis comme événements OpenTelemetry

Concepts clés :
- @evaluate() : décorateur qui attache des evaluators à une fonction
- OnlineEvaluator : wrapper avec sample_rate, max_concurrency, sink
- OnlineEvalConfig : config globale (default_sample_rate, sinks, metadata)
- OnlineEvaluation : capability PydanticAI pour les agents
- Sampling : statique (float), dynamique (callable), corrélé
- Sinks : handlers pour les résultats (alerting, DB, logging)
- run_evaluators() : ré-évaluer des données historiques sans ré-exécuter

Quand l'utiliser :
- Monitoring de qualité en prod
- Détection de régressions entre déploiements
- Collecte de données d'évaluation depuis du vrai trafic
- Contrôle des coûts (LLMJudge sur 1%, checks basiques sur 100%)
"""


# =====================================================================
# PARTIE 1 : @evaluate — décorateur basique
# =====================================================================

# Le décorateur @evaluate attache des evaluators à une fonction.
# Les evaluators tournent en background sans bloquer le caller.
# Les résultats sont émis comme événements OTel (gen_ai.evaluation.result).


@dataclass
class OutputNotEmpty(Evaluator):
    """Vérifie que l'output n'est pas vide."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


@dataclass
class OutputLength(Evaluator):
    """Score basé sur la longueur de l'output."""

    min_length: int = 10

    def evaluate(self, ctx: EvaluatorContext) -> float:
        length = len(str(ctx.output))
        return min(length / self.min_length, 1.0)


# Décorateur simple : les evaluators tournent sur chaque appel
@evaluate(OutputNotEmpty(), OutputLength(min_length=20))
async def summarize(text: str) -> str:
    """Fonction décorée — évaluée en background à chaque appel."""
    return f'Summary of: {text}'


# Avec target custom (nom affiché dans les dashboards)
@evaluate(OutputNotEmpty(), target='customer_support')
async def support_agent(query: str) -> str:
    """Target override — résultats groupés sous 'customer_support'."""
    return f'I can help you with: {query}'


# =====================================================================
# PARTIE 2 : OnlineEvaluator — sampling et concurrency
# =====================================================================

# OnlineEvaluator wraps un Evaluator avec :
# - sample_rate : probabilité d'évaluer (0.0 à 1.0)
# - max_concurrency : limite de parallélisme
# - sink : handler custom pour les résultats


@dataclass
class QuickCheck(Evaluator):
    """Check rapide et gratuit."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) > 0


# Check gratuit : tourne sur 100% du trafic
always_check = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=1.0)

# LLM Judge coûteux : tourne sur 1%, max 5 en parallèle
rare_check = OnlineEvaluator(
    evaluator=LLMJudge(
        rubric='Response is helpful and accurate',
        model='gateway/anthropic:claude-sonnet-4-6',
    ),
    sample_rate=0.01,
    max_concurrency=5,
)

# Désactivé (mais reste déclaré dans le code)
disabled_check = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=0.0)


# =====================================================================
# PARTIE 3 : Sampling dynamique et corrélé
# =====================================================================

# --- Sampling dynamique : callable au lieu d'un float ---
# Utile pour feature flags, remote config, ou logique input-dependent.


def rate_from_config(ctx: SamplingContext) -> float:
    """Taux de sampling configurable à runtime."""
    # En prod : lire depuis un service de config (Logfire managed variables, etc.)
    return 0.5


def sample_long_inputs(ctx: SamplingContext) -> bool:
    """Évalue seulement les inputs longs."""
    return len(str(ctx.inputs.get('text', ''))) > 100


dynamic_evaluator = OnlineEvaluator(
    evaluator=QuickCheck(),
    sample_rate=rate_from_config,
)

input_dependent = OnlineEvaluator(
    evaluator=OutputLength(min_length=50),
    sample_rate=sample_long_inputs,
)

# --- Sampling corrélé ---
# Par défaut, chaque evaluator sample indépendamment.
# Avec sampling_mode='correlated', les mêmes N% de requêtes
# passent par TOUS les evaluators (un seul random par appel).

correlated_config = OnlineEvalConfig(
    sampling_mode='correlated',
)


# =====================================================================
# PARTIE 4 : OnlineEvalConfig — configuration globale
# =====================================================================

# OnlineEvalConfig centralise : default_sample_rate, sinks,
# metadata, OTel toggle, error handlers.

results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    """Sink custom : log les résultats dans une liste."""
    for r in results:
        results_log.append(f'{r.name}={r.value}')
    for f in failures:
        results_log.append(f'FAILED:{f.name}={f.error_message}')


my_config = OnlineEvalConfig(
    default_sink=log_sink,
    default_sample_rate=1.0,
    metadata={'service': 'my-app', 'environment': 'staging'},
)


@my_config.evaluate(OutputNotEmpty(), OutputLength(min_length=10))
async def configured_function(query: str) -> str:
    """Fonction avec config custom — résultats vont dans log_sink."""
    return f'Answer to: {query}'


# =====================================================================
# PARTIE 5 : Per-evaluator sink overrides
# =====================================================================

# Différents evaluators peuvent envoyer leurs résultats à
# différentes destinations.

default_log: list[str] = []
alert_log: list[str] = []


async def default_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        default_log.append(r.name)


async def alert_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    """Sink d'alerte — pour les checks critiques."""
    for r in results:
        alert_log.append(f'ALERT:{r.name}={r.value}')


sink_config = OnlineEvalConfig(default_sink=default_sink)


@dataclass
class CriticalCheck(Evaluator):
    """Check critique qui doit alerter."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        output = str(ctx.output).lower()
        return 'error' not in output and 'fail' not in output


@sink_config.evaluate(
    QuickCheck(),  # → default_sink
    OnlineEvaluator(evaluator=CriticalCheck(), sink=alert_sink),  # → alert_sink
)
async def monitored_function(text: str) -> str:
    return f'Processed: {text}'


# =====================================================================
# PARTIE 6 : Evaluator versioning
# =====================================================================

# evaluator_version permet de filtrer les résultats historiques
# quand on change la logique d'un evaluator (ex: nouveau rubric).


@dataclass
class ToneCheck(Evaluator):
    """Check de ton — versionné pour tracking historique."""

    evaluator_version = 'v2'  # Bumpé après réécriture du prompt

    def evaluate(self, ctx: EvaluatorContext) -> str:
        output = str(ctx.output).lower()
        if any(w in output for w in ('sorry', 'apologize')):
            return 'apologetic'
        if any(w in output for w in ('great', 'happy')):
            return 'positive'
        return 'neutral'


# =====================================================================
# PARTIE 7 : Agent integration — OnlineEvaluation capability
# =====================================================================

# Pour les agents PydanticAI, utiliser OnlineEvaluation comme capability.
# Le target est automatiquement le nom de l'agent.

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    name='assistant',
    capabilities=[
        OnlineEvaluation(
            evaluators=[
                OutputNotEmpty(),
                OnlineEvaluator(
                    evaluator=LLMJudge(
                        rubric='Response is helpful and on-topic',
                        model='gateway/anthropic:claude-sonnet-4-6',
                    ),
                    sample_rate=0.1,  # 10% du trafic
                    max_concurrency=3,
                ),
            ],
            # config optionnelle pour override les defaults
            # config=OnlineEvalConfig(default_sample_rate=0.5),
        ),
    ],
    instructions='Tu es un assistant utile. Réponds de manière concise.',
)


# =====================================================================
# PARTIE 8 : disable_evaluation et conditional evaluation
# =====================================================================

# disable_evaluation() supprime toute évaluation dans un scope.
# Utile dans les tests pour ne pas polluer les métriques.


@dataclass
class ConditionalAnalysis(Evaluator):
    """Check conditionnel : cheap toujours, expensive seulement si output long."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        output = str(ctx.output)
        results: dict[str, float | bool] = {
            'has_content': len(output) > 0,
        }
        # Expensive check seulement sur les outputs longs
        if len(output) > 50:
            results['detail_score'] = min(len(output) / 200.0, 1.0)
        return results


# =====================================================================
# PARTIE 9 : run_evaluators — ré-évaluer des données historiques
# =====================================================================

# run_evaluators() permet d'exécuter des evaluators sur un
# EvaluatorContext construit manuellement (depuis des logs, DB, etc.)
# sans ré-exécuter la fonction originale.


@dataclass
class HistoricalCheck(Evaluator):
    """Evaluator pour ré-évaluation de données historiques."""

    keyword: str = 'hello'

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return self.keyword in str(ctx.output).lower()


# =====================================================================
# PARTIE 10 : Error handling
# =====================================================================

# on_error : appelé quand un sink ou callback échoue
# on_sampling_error : appelé quand un sample_rate callable raise

from pydantic_evals.online import OnErrorLocation  # noqa: E402


def handle_error(
    exc: Exception,
    ctx: EvaluatorContext,
    evaluator: Evaluator,
    location: OnErrorLocation,
) -> None:
    """Handler d'erreur global — log au lieu de silently suppress."""
    logger.error(f"[{location}] {type(exc).__name__}: {exc}")


error_config = OnlineEvalConfig(
    default_sink=log_sink,
    on_error=handle_error,
)


# =====================================================================
# PARTIE 11 : Exécution
# =====================================================================


async def main():
    """Exécute les démos online evaluation."""

    # --- Démo 1 : @evaluate basique ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : @evaluate basique")
    logger.info("=" * 60)

    result = await summarize("The quick brown fox jumps over the lazy dog")
    logger.info(f"  Output : {result}")
    await wait_for_evaluations()
    logger.success("  Evaluators ont tourné en background (voir OTel events)")

    # --- Démo 2 : Config custom avec sink ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : OnlineEvalConfig + log_sink")
    logger.info("=" * 60)

    results_log.clear()
    result = await configured_function("What is PydanticAI?")
    logger.info(f"  Output : {result}")
    await wait_for_evaluations()
    logger.info(f"  Résultats capturés : {results_log}")

    # --- Démo 3 : Per-evaluator sink overrides ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Sink overrides (default + alert)")
    logger.info("=" * 60)

    default_log.clear()
    alert_log.clear()
    result = await monitored_function("All systems operational")
    await wait_for_evaluations()
    logger.info(f"  Default log : {default_log}")
    logger.info(f"  Alert log   : {alert_log}")

    # --- Démo 4 : disable_evaluation ---
    logger.info("=" * 60)
    logger.info("DÉMO 4 : disable_evaluation (pour les tests)")
    logger.info("=" * 60)

    results_log.clear()
    with disable_evaluation():
        result = await configured_function("This should not be evaluated")
        logger.info(f"  Output : {result}")
    await wait_for_evaluations()
    logger.info(f"  Résultats (devrait être vide) : {results_log}")

    # Hors du scope, les evals reprennent
    result = await configured_function("This SHOULD be evaluated")
    await wait_for_evaluations()
    logger.info(f"  Résultats (devrait avoir des entrées) : {results_log}")

    # --- Démo 5 : run_evaluators — ré-évaluation historique ---
    logger.info("=" * 60)
    logger.info("DÉMO 5 : run_evaluators (données historiques)")
    logger.info("=" * 60)

    historical_ctx = EvaluatorContext(
        name='historical_call_42',
        inputs={'query': 'greet the user'},
        output='Hello! How can I help you today?',
        expected_output=None,
        metadata=None,
        duration=0.5,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    results, failures = await run_evaluators(
        [
            HistoricalCheck(keyword='hello'),
            OutputNotEmpty(),
            OutputLength(min_length=20),
        ],
        historical_ctx,
    )

    for r in results:
        logger.info(f"  {r.name}: {r.value}")
    if failures:
        for f in failures:
            logger.warning(f"  FAILED {f.name}: {f.error_message}")
    logger.success(f"  {len(results)} résultats, {len(failures)} échecs")

    # --- Démo 6 : Agent avec OnlineEvaluation (ATTENTION : coûte de l'argent) ---
    logger.info("=" * 60)
    logger.info("DÉMO 6 : Agent + OnlineEvaluation capability")
    logger.info("=" * 60)

    result = await agent.run("Qu'est-ce que PydanticAI ?")
    logger.info(f"  Output : {result.output[:200]}")
    await wait_for_evaluations()
    logger.success("  Agent évalué en background (target='assistant')")

    logger.success("Toutes les démos Online Evaluation sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
