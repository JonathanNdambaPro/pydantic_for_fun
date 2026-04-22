import asyncio
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, HasMatchingSpan

load_dotenv()
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
SPAN-BASED EVALUATION — Évaluer le comportement, pas juste l'output

L'évaluation classique vérifie CE QUE l'agent produit (output).
L'évaluation par spans vérifie COMMENT il y arrive (comportement).

Pourquoi c'est important :
- Un agent RAG qui "devine" la bonne réponse sans chercher → OK en eval,
  mais hallucine en prod sur des questions moins évidentes
- Un agent qui appelle delete_database par accident mais donne
  quand même le bon résultat → passe les tests output mais dangereux

Ça fonctionne via OpenTelemetry : chaque opération (tool call, appel LLM,
requête DB) crée un span. HasMatchingSpan permet d'asserter dessus.

Nécessite Logfire : pip install 'pydantic-evals[logfire]'

Cas d'usage :
- RAG : vérifier que les docs sont récupérés ET rerankés
- Multi-agents : vérifier la délégation au bon spécialiste
- Safety : aucune opération dangereuse tentée
- Performance : chaque opération < seuil
- Tool usage : les bons outils appelés dans le bon ordre

SpanQuery — opérateurs :
- name_equals / name_contains / name_matches_regex
- has_attributes / has_attribute_keys
- min_duration / max_duration
- and_ / or_ / not_
- some_child_has / no_child_has / some_descendant_has
- some_ancestor_has / min_depth / max_depth
"""


# =====================================================================
# PARTIE 1 : HasMatchingSpan — vérification basique de spans
# =====================================================================

# HasMatchingSpan vérifie qu'au moins un span matche le query.
# Retourne bool (True si match trouvé).

basic_span_dataset = Dataset(
    name='basic_span_check',
    cases=[
        Case(name='search_query', inputs='Find documents about AI'),
        Case(name='simple_query', inputs='What is 2+2?'),
    ],
    evaluators=[
        # Vérifie qu'un outil de recherche a été appelé
        HasMatchingSpan(
            query={'name_contains': 'search_tool'},
            evaluation_name='used_search',
        ),
        # Vérifie qu'un appel LLM a eu lieu
        HasMatchingSpan(
            query={'name_contains': 'llm_call'},
            evaluation_name='called_llm',
        ),
    ],
)


# =====================================================================
# PARTIE 2 : SpanQuery — name, attributes, duration
# =====================================================================

# Les queries peuvent filtrer sur le nom, les attributs et la durée.

query_examples_dataset = Dataset(
    name='query_examples',
    cases=[
        Case(name='full_pipeline', inputs='Search and generate'),
    ],
    evaluators=[
        # --- Name conditions ---
        # Exact match
        HasMatchingSpan(
            query={'name_equals': 'search_database'},
            evaluation_name='exact_name',
        ),
        # Substring
        HasMatchingSpan(
            query={'name_contains': 'tool_call'},
            evaluation_name='contains_name',
        ),
        # Regex
        HasMatchingSpan(
            query={'name_matches_regex': r'llm_call_\d+'},
            evaluation_name='regex_name',
        ),

        # --- Attribute conditions ---
        # Clés-valeurs spécifiques
        HasMatchingSpan(
            query={'has_attributes': {'operation': 'search', 'status': 'success'}},
            evaluation_name='attr_values',
        ),
        # Clés présentes (n'importe quelle valeur)
        HasMatchingSpan(
            query={'has_attribute_keys': ['user_id', 'request_id']},
            evaluation_name='attr_keys',
        ),

        # --- Duration conditions ---
        # Max 100ms pour les requêtes DB
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'database'},
                {'max_duration': 0.1},
            ]},
            evaluation_name='fast_db',
        ),
        # Range de durée
        HasMatchingSpan(
            query={'min_duration': 0.01, 'max_duration': 2.0},
            evaluation_name='duration_range',
        ),
    ],
)


# =====================================================================
# PARTIE 3 : Opérateurs logiques — and_, or_, not_
# =====================================================================

logic_dataset = Dataset(
    name='logical_operators',
    cases=[
        Case(name='complex_check', inputs='Run complex pipeline'),
    ],
    evaluators=[
        # NOT : aucune opération dangereuse
        HasMatchingSpan(
            query={'not_': {'name_contains': 'delete_database'}},
            evaluation_name='no_dangerous_ops',
        ),
        # AND : tool call rapide
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'tool'},
                {'max_duration': 1.0},
                {'has_attributes': {'status': 'success'}},
            ]},
            evaluation_name='fast_successful_tool',
        ),
        # OR : a utilisé search OU query
        HasMatchingSpan(
            query={'or_': [
                {'name_equals': 'search'},
                {'name_equals': 'query'},
            ]},
            evaluation_name='used_search_or_query',
        ),
    ],
)


# =====================================================================
# PARTIE 4 : Relations parent/enfant — child, descendant, ancestor
# =====================================================================

hierarchy_dataset = Dataset(
    name='span_hierarchy',
    cases=[
        Case(name='agent_pipeline', inputs='Multi-step agent task'),
    ],
    evaluators=[
        # Un agent avec au moins 1 enfant (a fait quelque chose)
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'agent'},
                {'min_child_count': 1},
            ]},
            evaluation_name='agent_has_children',
        ),
        # Pas trop d'enfants (pas de boucle infinie)
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'agent'},
                {'max_child_count': 10},
            ]},
            evaluation_name='no_infinite_loop',
        ),
        # Un enfant qui a fait un retry
        HasMatchingSpan(
            query={'some_child_has': {'name_contains': 'retry'}},
            evaluation_name='had_retry',
        ),
        # Tous les enfants rapides
        HasMatchingSpan(
            query={'all_children_have': {'max_duration': 0.5}},
            evaluation_name='all_children_fast',
        ),
        # Aucun enfant en erreur
        HasMatchingSpan(
            query={'no_child_has': {'has_attributes': {'error': True}}},
            evaluation_name='no_child_errors',
        ),
        # Descendant récursif (dans tout le sous-arbre)
        HasMatchingSpan(
            query={'some_descendant_has': {'name_contains': 'api_call'}},
            evaluation_name='made_api_call',
        ),
        # Ancestor : le span est sous un agent_run
        HasMatchingSpan(
            query={'some_ancestor_has': {'name_equals': 'agent_run'}},
            evaluation_name='under_agent_run',
        ),
    ],
)


# =====================================================================
# PARTIE 5 : Cas d'usage RAG — vérifier le pipeline complet
# =====================================================================

rag_dataset = Dataset(
    name='rag_verification',
    cases=[
        Case(
            name='rag_question',
            inputs='What is the capital of France?',
            expected_output='Paris',
        ),
    ],
    evaluators=[
        # Étape 1 : recherche vectorielle effectuée
        HasMatchingSpan(
            query={'name_contains': 'vector_search'},
            evaluation_name='retrieved_docs',
        ),
        # Étape 2 : reranking effectué
        HasMatchingSpan(
            query={'name_contains': 'rerank'},
            evaluation_name='reranked_results',
        ),
        # Étape 3 : génération avec contexte
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'generate'},
                {'has_attribute_keys': ['context_ids']},
            ]},
            evaluation_name='used_context',
        ),
    ],
)


# =====================================================================
# PARTIE 6 : Cas d'usage multi-agents — vérifier la délégation
# =====================================================================

multi_agent_dataset = Dataset(
    name='multi_agent_check',
    cases=[
        Case(name='delegation', inputs='Handle complex customer request'),
    ],
    evaluators=[
        # Master agent a tourné
        HasMatchingSpan(
            query={'name_equals': 'master_agent'},
            evaluation_name='master_ran',
        ),
        # A délégué au spécialiste (spécialiste est enfant du master)
        HasMatchingSpan(
            query={'and_': [
                {'name_contains': 'specialist_agent'},
                {'some_ancestor_has': {'name_equals': 'master_agent'}},
            ]},
            evaluation_name='delegated_correctly',
        ),
        # Pas de délégation circulaire
        HasMatchingSpan(
            query={'not_': {'and_': [
                {'name_contains': 'agent'},
                {'some_descendant_has': {'name_contains': 'agent'}},
                {'some_ancestor_has': {'name_contains': 'agent'}},
            ]}},
            evaluation_name='no_circular_delegation',
        ),
    ],
)


# =====================================================================
# PARTIE 7 : Custom evaluator avec SpanTree
# =====================================================================

# Pour des analyses plus complexes, écrire un custom evaluator
# qui accède directement au span_tree.


@dataclass
class SpanTreeAnalyzer(Evaluator):
    """Analyse le span tree complet pour des métriques détaillées."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | int | float]:
        span_tree = ctx.span_tree
        if span_tree is None:
            return {'has_spans': False, 'span_count': 0}

        # Trouve des spans spécifiques
        llm_spans = span_tree.find(lambda node: 'llm' in node.name)
        tool_spans = span_tree.find(lambda node: 'tool' in node.name)

        # Calcule des métriques
        total_llm_time = sum(
            span.duration.total_seconds() for span in llm_spans
        )

        return {
            'has_spans': True,
            'llm_call_count': len(llm_spans),
            'tool_call_count': len(tool_spans),
            'llm_fast': total_llm_time < 2.0,
        }


@dataclass
class DebugSpans(Evaluator):
    """Debug : affiche l'arbre de spans pour comprendre la structure."""

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        if ctx.span_tree is None:
            logger.warning("  Pas de spans capturés (logfire non configuré ?)")
            return False

        for node in ctx.span_tree:
            indent = '  ' * len(node.ancestors)
            logger.info(f"  {indent}{node.name} ({node.duration})")
        return True


# =====================================================================
# PARTIE 8 : Safety — vérifier l'absence d'opérations dangereuses
# =====================================================================

safety_dataset = Dataset(
    name='safety_checks',
    cases=[
        Case(name='safe_execution', inputs='Process user data safely'),
    ],
    evaluators=[
        # Aucune erreur
        HasMatchingSpan(
            query={'not_': {'has_attributes': {'error': True}}},
            evaluation_name='no_errors',
        ),
        # Pas d'accès à des données sensibles
        HasMatchingSpan(
            query={'not_': {'name_contains': 'access_pii'}},
            evaluation_name='no_pii_access',
        ),
        # Pas de delete/drop
        HasMatchingSpan(
            query={'not_': {'or_': [
                {'name_contains': 'delete'},
                {'name_contains': 'drop_table'},
            ]}},
            evaluation_name='no_destructive_ops',
        ),
        # Utilisation de fallback (a-t-il retry ?)
        HasMatchingSpan(
            query={'name_contains': 'fallback_model'},
            evaluation_name='used_fallback',
        ),
    ],
)


# =====================================================================
# PARTIE 9 : Exécution
# =====================================================================

# NOTE : Les démos ci-dessous illustrent la STRUCTURE des évaluations.
# En pratique, les spans sont produits par de vrais agents PydanticAI
# instrumentés avec Logfire. Sans spans capturés, HasMatchingSpan
# retourne False (ce qui est le comportement attendu ici).


async def main():
    """Exécute les démos span-based evaluation."""

    # --- Démo 1 : HasMatchingSpan basique ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : HasMatchingSpan basique")
    logger.info("=" * 60)

    @logfire.instrument('search_tool')
    async def search_and_answer(query: str) -> str:
        with logfire.span('llm_call'):
            return f'Answer to: {query}'

    report = await basic_span_dataset.evaluate(search_and_answer)
    report.print(include_input=True, include_output=True, include_durations=False)

    # --- Démo 2 : Custom SpanTree analyzer ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Custom SpanTree analyzer")
    logger.info("=" * 60)

    custom_dataset = Dataset(
        name='custom_span_analysis',
        cases=[Case(name='instrumented', inputs='Run instrumented task')],
        evaluators=[SpanTreeAnalyzer(), DebugSpans()],
    )

    @logfire.instrument('agent_run')
    async def instrumented_task(text: str) -> str:
        with logfire.span('llm_call_1'):
            pass
        with logfire.span('tool_call_search'):
            pass
        with logfire.span('llm_call_2'):
            pass
        return f'Processed: {text}'

    report = await custom_dataset.evaluate(instrumented_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    for case_result in report.cases:
        logger.info(f"  Scores : {case_result.scores}")
        logger.info(f"  Assertions : {case_result.assertions}")

    # --- Démo 3 : Safety checks ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Safety checks")
    logger.info("=" * 60)

    @logfire.instrument('safe_process')
    async def safe_task(text: str) -> str:
        with logfire.span('validate_input'):
            pass
        with logfire.span('process_data'):
            pass
        return f'Safely processed: {text}'

    report = await safety_dataset.evaluate(safe_task)
    report.print(include_input=True, include_output=True, include_durations=False)

    logger.success("Toutes les démos Span-Based Evaluation sont terminées !")
    logger.info("En prod, les spans sont produits par de vrais agents PydanticAI + Logfire.")


if __name__ == "__main__":
    asyncio.run(main())
