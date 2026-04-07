import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, CodeExecutionTool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CODE EXECUTION TOOL — Exécution de code dans un environnement sécurisé

CodeExecutionTool permet au LLM d'écrire et exécuter du code
directement dans un sandbox sécurisé côté provider.
Parfait pour : calculs, analyse de données, opérations mathématiques.

Le code est exécuté par le provider, PAS par Pydantic AI.
→ Pas de risque de sécurité côté application.

Providers supportés :
- Anthropic        ✅
- OpenAI Responses ✅ (peut générer des images, ex: graphiques)
- Google           ✅ (limitations avec function tools)
- xAI              ✅ (support complet)
- Bedrock          ✅ (Nova 2.0 uniquement)
- Groq             ❌
- Mistral          ❌

Fonctionnalités spécifiques OpenAI :
- Peut générer des images (graphiques, charts…)
- Images accessibles via result.response.images
- Nécessite openai_include_code_execution_outputs=True

On peut inspecter le code exécuté et son résultat via
result.response.builtin_tool_calls qui retourne des paires
(BuiltinToolCallPart, BuiltinToolReturnPart).
"""


# =====================================================================
# PARTIE 1 : CodeExecutionTool basique
# =====================================================================

# Le LLM écrit du code Python, le provider l'exécute dans un sandbox,
# et retourne stdout/stderr + code de retour.

agent_code = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[CodeExecutionTool()],
    instructions=(
        "Tu es un assistant mathématique. "
        "Utilise l'exécution de code pour les calculs complexes. "
        "Réponds en français."
    ),
)


# =====================================================================
# PARTIE 2 : Inspecter le code exécuté
# =====================================================================

# Après un run, on peut voir exactement quel code le LLM a écrit
# et ce que le sandbox a retourné. Utile pour le debug ou l'audit.


def inspect_code_execution(result):
    """Affiche le code exécuté et son résultat."""
    for call, return_part in result.response.builtin_tool_calls:
        logger.info(f"Code exécuté :\n{call.args.get('code', 'N/A')}")
        if hasattr(return_part, 'content') and isinstance(return_part.content, dict):
            logger.info(f"stdout : {return_part.content.get('stdout', '')}")
            logger.info(f"stderr : {return_part.content.get('stderr', '')}")
            logger.info(f"Code retour : {return_part.content.get('return_code', '')}")


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Calcul mathématique ---
    logger.info("=== Calcul : factorielle de 15 ===")
    result = agent_code.run_sync("Calcule la factorielle de 15")
    logger.success(f"Réponse : {result.output}")
    inspect_code_execution(result)

    # --- Démo 2 : Analyse de données ---
    logger.info("=== Analyse de données ===")
    result = agent_code.run_sync(
        "Génère une liste de 100 nombres aléatoires entre 1 et 1000, "
        "puis donne-moi la moyenne, la médiane et l'écart-type."
    )
    logger.success(f"Réponse : {result.output}")
    inspect_code_execution(result)
