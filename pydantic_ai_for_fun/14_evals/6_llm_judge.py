import asyncio
from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelSettings, format_as_xml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
LLM JUDGE — Évaluation subjective via LLM-as-a-Judge

LLMJudge utilise un LLM pour évaluer des qualités qui demandent
compréhension et jugement. Chaque évaluation = un appel LLM (coût réel).

Bons cas d'usage :
- Exactitude factuelle
- Pertinence et utilité
- Ton et style
- Complétude
- Suivi d'instructions complexes
- RAG : groundedness, citations

Mauvais cas d'usage (préférer les déterministes) :
- Validation de format → IsInstance
- Match exact → EqualsExpected
- Performance → MaxDuration
- Logique déterministe → custom Evaluator

Configuration clé :
- rubric (str) : critère d'évaluation — être SPÉCIFIQUE
- model : modèle juge (défaut : openai:gpt-5.2)
- include_input / include_expected_output : contexte visible par le juge
- score / assertion : modes de sortie (bool, float, ou les deux)
- model_settings : temperature=0.0 pour la reproductibilité

Bonnes pratiques :
1. Rubrics spécifiques > rubrics vagues
2. Plusieurs juges spécialisés > un juge générique
3. Checks déterministes d'abord, LLM en dernier
4. temperature=0.0 pour la cohérence
"""


# =====================================================================
# PARTIE 1 : Configuration de base — rubric et contexte
# =====================================================================

# Le rubric est le critère d'évaluation. Plus il est spécifique,
# plus le jugement est fiable.

# --- Mauvais rubric (vague) ---
# LLMJudge(rubric='Good response')              # Trop vague
# LLMJudge(rubric='Check quality')              # Quel aspect ?

# --- Bon rubric (spécifique) ---
basic_dataset = Dataset(
    name='rubric_quality',
    cases=[
        Case(
            name='factual_question',
            inputs='What is the capital of France?',
            expected_output='Paris',
        ),
        Case(
            name='explanation_question',
            inputs='Why is the sky blue?',
            expected_output='Rayleigh scattering of sunlight by the atmosphere',
        ),
    ],
    evaluators=[
        # Output seul (défaut)
        LLMJudge(
            rubric='Response is factually accurate and free of hallucinations',
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Output + Input → le juge voit la question
        LLMJudge(
            rubric='Response directly answers the input question without digression',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Output + Input + Expected → le juge compare
        LLMJudge(
            rubric='Response is semantically equivalent to the expected output',
            include_input=True,
            include_expected_output=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 2 : Modes de sortie — assertion, score, combiné
# =====================================================================

# --- Mode assertion (défaut) : bool + reason ---
assertion_dataset = Dataset(
    name='assertion_mode',
    cases=[Case(inputs='What is 2+2?', expected_output='4')],
    evaluators=[
        LLMJudge(
            rubric='Response is mathematically correct',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
            # Retourne : {'LLMJudge_pass': EvaluationReason(value=True, reason='...')}
        ),
    ],
)

# --- Mode score : float (0.0 à 1.0) + reason ---
score_dataset = Dataset(
    name='score_mode',
    cases=[Case(inputs='Explain quantum computing in simple terms')],
    evaluators=[
        LLMJudge(
            rubric='Overall quality: clarity, accuracy, and accessibility',
            model='gateway/anthropic:claude-sonnet-4-6',
            score={'evaluation_name': 'quality', 'include_reason': True},
            assertion=False,
            # Retourne : {'quality': EvaluationReason(value=0.85, reason='...')}
        ),
    ],
)

# --- Mode combiné : assertion + score ---
combined_dataset = Dataset(
    name='combined_mode',
    cases=[
        Case(
            inputs='Write a professional email to reschedule a meeting',
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
            # Retourne les deux :
            # {'professionalism': EvaluationReason(value=0.9, reason='...'),
            #  'is_professional': EvaluationReason(value=True, reason='...')}
        ),
    ],
)


# =====================================================================
# PARTIE 3 : Multi-aspect — plusieurs juges spécialisés
# =====================================================================

# Plutôt qu'un seul juge générique, utiliser plusieurs juges
# spécialisés par dimension de qualité.

multi_aspect_dataset = Dataset(
    name='multi_aspect',
    cases=[
        Case(
            name='support_response',
            inputs='My order arrived damaged, what can I do?',
        ),
    ],
    evaluators=[
        # Dimension 1 : exactitude
        LLMJudge(
            rubric='Response provides accurate information about return/refund process',
            include_input=True,
            assertion={'evaluation_name': 'accurate'},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Dimension 2 : utilité
        LLMJudge(
            rubric='Response is helpful and provides actionable next steps',
            include_input=True,
            score={'evaluation_name': 'helpfulness'},
            assertion=False,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Dimension 3 : ton
        LLMJudge(
            rubric='Response uses empathetic, professional language appropriate for customer support',
            assertion={'evaluation_name': 'professional_tone'},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Dimension 4 : sécurité
        LLMJudge(
            rubric='Response contains no harmful, biased, or inappropriate content',
            assertion={'evaluation_name': 'safe'},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 4 : RAG evaluation — groundedness et citations
# =====================================================================

# Évaluer si un système RAG utilise bien le contexte fourni
# et ne hallucine pas d'informations.


@dataclass
class RAGInput:
    """Input structuré pour une question RAG avec contexte."""
    question: str
    context: str


rag_dataset = Dataset(
    name='rag_evaluation',
    cases=[
        Case(
            name='grounded_answer',
            inputs=RAGInput(
                question='What is the capital of France?',
                context='France is a country in Western Europe. Its capital is Paris, '
                        'known for the Eiffel Tower and the Louvre Museum.',
            ),
        ),
        Case(
            name='insufficient_context',
            inputs=RAGInput(
                question='What is the population of Paris?',
                context='Paris is the capital of France. It is known for its cuisine.',
            ),
        ),
    ],
    evaluators=[
        # Le juge vérifie que la réponse est basée sur le contexte
        LLMJudge(
            rubric='Response answers the question using ONLY information from the provided context. '
                   'Does not add facts not present in the context.',
            include_input=True,
            assertion={'evaluation_name': 'grounded', 'include_reason': True},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
        # Le juge vérifie les citations
        LLMJudge(
            rubric='Response cites or references specific facts from the context. '
                   'If context is insufficient, acknowledges the limitation.',
            include_input=True,
            assertion={'evaluation_name': 'uses_citations', 'include_reason': True},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 5 : Recipe evaluation — case-specific + dataset-level
# =====================================================================

# Exemple concret : évaluation de recettes avec contraintes
# diététiques différentes par case.


class CustomerOrder(BaseModel):
    """Commande client avec restriction diététique."""
    dish_name: str
    dietary_restriction: str | None = None


class Recipe(BaseModel):
    """Recette générée."""
    ingredients: list[str]
    steps: list[str]


recipe_agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Recipe,
    instructions='Generate a recipe to cook the dish that meets the dietary restrictions.',
)


async def generate_recipe(order: CustomerOrder) -> Recipe:
    """Task qui génère une recette via l'agent."""
    result = await recipe_agent.run(format_as_xml(order))
    return result.output


recipe_dataset = Dataset[CustomerOrder, Recipe, None](
    name='recipe_evaluation',
    cases=[
        Case(
            name='vegetarian_bolognese',
            inputs=CustomerOrder(
                dish_name='Spaghetti Bolognese',
                dietary_restriction='vegetarian',
            ),
            evaluators=[
                # Case-specific : vérifie la contrainte végétarienne
                LLMJudge(
                    rubric='Recipe should not contain meat or animal products',
                    model='gateway/anthropic:claude-sonnet-4-6',
                ),
            ],
        ),
        Case(
            name='gluten_free_cake',
            inputs=CustomerOrder(
                dish_name='Chocolate Cake',
                dietary_restriction='gluten-free',
            ),
            evaluators=[
                # Case-specific : vérifie la contrainte sans gluten
                LLMJudge(
                    rubric='Recipe should not contain gluten or wheat products',
                    model='gateway/anthropic:claude-sonnet-4-6',
                ),
            ],
        ),
    ],
    evaluators=[
        # Dataset-level : tourne pour TOUS les cases
        IsInstance(type_name='Recipe'),
        LLMJudge(
            rubric='Recipe should have clear, numbered steps and relevant ingredients '
                   'for the requested dish',
            include_input=True,
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 6 : Comparative evaluation — output vs expected
# =====================================================================

# Comparer sémantiquement l'output avec l'expected_output.
# Utile pour les traductions, paraphrases, résumés.

comparative_dataset = Dataset(
    name='comparative_eval',
    cases=[
        Case(
            name='translation_fr',
            inputs='Hello world',
            expected_output='Bonjour le monde',
        ),
        Case(
            name='translation_es',
            inputs='Good morning',
            expected_output='Buenos días',
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is semantically equivalent to the expected output, '
                   'regardless of phrasing differences',
            include_input=True,
            include_expected_output=True,
            score={'evaluation_name': 'semantic_similarity'},
            assertion={'evaluation_name': 'correct_meaning'},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 7 : Model settings et default judge model
# =====================================================================

# temperature=0.0 pour la reproductibilité.
# set_default_judge_model() pour changer le modèle par défaut.

# Change le modèle par défaut pour TOUS les LLMJudge sans model explicite
set_default_judge_model('gateway/anthropic:claude-sonnet-4-6')

deterministic_judge_dataset = Dataset(
    name='deterministic_judge',
    cases=[
        Case(inputs='What is Python?'),
    ],
    evaluators=[
        # Utilise le default model (Claude) grâce à set_default_judge_model
        LLMJudge(
            rubric='Response accurately describes Python as a programming language',
            include_input=True,
            model_settings=ModelSettings(
                temperature=0.0,  # Plus déterministe
                max_tokens=200,   # Limite la verbosité du juge
            ),
        ),
    ],
)


# =====================================================================
# PARTIE 8 : Debugging — raisons et accès programmatique
# =====================================================================

# Toujours utiliser include_reasons=True dans report.print()
# pour comprendre les décisions du juge.

debug_dataset = Dataset(
    name='debug_reasons',
    cases=[
        Case(name='clear_case', inputs='What is 1+1?'),
        Case(name='ambiguous_case', inputs='Is water wet?'),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is clear, concise, and factually correct',
            include_input=True,
            assertion={'evaluation_name': 'clarity', 'include_reason': True},
            score={'evaluation_name': 'quality', 'include_reason': True},
            model='gateway/anthropic:claude-sonnet-4-6',
        ),
    ],
)


# =====================================================================
# PARTIE 9 : Exécution
# =====================================================================


async def main():
    """Exécute les démos LLM Judge. ATTENTION : chaque démo coûte de l'argent."""

    logger.warning("Toutes les démos font des appels LLM — coût réel !")

    # --- Démo 1 : Rubric et contexte ---
    logger.info("=" * 60)
    logger.info("DÉMO 1 : Rubric et niveaux de contexte")
    logger.info("=" * 60)

    async def answer_task(question: str) -> str:
        answers = {
            'What is the capital of France?': 'Paris is the capital of France.',
            'Why is the sky blue?': 'The sky appears blue due to Rayleigh scattering.',
        }
        return answers.get(question, 'I am not sure.')

    report = await basic_dataset.evaluate(answer_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    # --- Démo 2 : Modes de sortie ---
    logger.info("=" * 60)
    logger.info("DÉMO 2 : Assertion / Score / Combiné")
    logger.info("=" * 60)

    async def math_task(q: str) -> str:
        return 'The answer is 4.'

    report = await assertion_dataset.evaluate(math_task)
    report.print(include_reasons=True, include_durations=False)
    logger.info("  ^ Mode assertion : bool + reason")

    async def explain_task(q: str) -> str:
        return 'Quantum computing uses qubits that can be 0, 1, or both at once.'

    report = await score_dataset.evaluate(explain_task)
    report.print(include_reasons=True, include_durations=False)
    logger.info("  ^ Mode score : float + reason")

    async def email_task(q: str) -> str:
        return (
            'Dear Team,\n\n'
            'I hope this message finds you well. I would like to request '
            'rescheduling our meeting from Tuesday to Thursday at 2 PM.\n\n'
            'Best regards'
        )

    report = await combined_dataset.evaluate(email_task)
    report.print(include_reasons=True, include_durations=False)
    logger.info("  ^ Mode combiné : assertion + score")

    # --- Démo 3 : Multi-aspect ---
    logger.info("=" * 60)
    logger.info("DÉMO 3 : Multi-aspect (4 dimensions)")
    logger.info("=" * 60)

    async def support_task(query: str) -> str:
        return (
            'I am sorry to hear about the damage to your order. '
            'Here is what you can do:\n'
            '1. Take photos of the damaged item\n'
            '2. Contact us within 30 days for a full refund\n'
            '3. We will arrange a free return pickup\n'
            'Please let me know if you need any further assistance.'
        )

    report = await multi_aspect_dataset.evaluate(support_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    # --- Démo 4 : RAG evaluation ---
    logger.info("=" * 60)
    logger.info("DÉMO 4 : RAG — groundedness et citations")
    logger.info("=" * 60)

    async def rag_task(inp: RAGInput) -> str:
        if 'capital' in inp.question.lower() and 'Paris' in inp.context:
            return 'According to the context, Paris is the capital of France.'
        return 'Based on the provided context, I cannot find specific population data for Paris.'

    report = await rag_dataset.evaluate(rag_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    # --- Démo 5 : Comparative (traduction) ---
    logger.info("=" * 60)
    logger.info("DÉMO 5 : Comparative — similarité sémantique")
    logger.info("=" * 60)

    async def translate_task(text: str) -> str:
        translations = {
            'Hello world': 'Bonjour le monde',
            'Good morning': 'Bonjour',  # Pas exactement 'Buenos días'
        }
        return translations.get(text, text)

    report = await comparative_dataset.evaluate(translate_task)
    report.print(include_input=True, include_output=True, include_reasons=True, include_durations=False)

    # --- Démo 6 : Debugging — accès programmatique aux raisons ---
    logger.info("=" * 60)
    logger.info("DÉMO 6 : Debugging — raisons programmatiques")
    logger.info("=" * 60)

    async def simple_task(q: str) -> str:
        if '1+1' in q:
            return '2'
        return 'That is a philosophical question with no definitive answer.'

    report = await debug_dataset.evaluate(simple_task)
    report.print(include_reasons=True, include_durations=False)

    # Accès programmatique
    for case_result in report.cases:
        logger.info(f"  Case '{case_result.name}' :")
        for name, result in case_result.assertions.items():
            logger.info(f"    {name}: {result.value}")
            if hasattr(result, 'reason') and result.reason:
                logger.info(f"      Reason: {result.reason[:100]}...")
        for name, result in case_result.scores.items():
            logger.info(f"    {name}: {result.value}")

    logger.success("Toutes les démos LLM Judge sont terminées !")


if __name__ == "__main__":
    asyncio.run(main())
