import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
CUSTOM ARGS VALIDATOR — Validation métier avant exécution du tool

Le pipeline de validation d'un tool se déroule en 3 étapes :

1. Validation Pydantic (schéma)   → types, champs requis, contraintes
2. args_validator (custom)        → logique métier, cross-field, limites
3. Exécution du tool              → seulement si 1 et 2 ont passé

args_validator s'intercale entre la validation de schéma et l'exécution.
C'est utile quand :
- La validation dépend de plusieurs champs à la fois (cross-field)
- Les règles dépendent du contexte (deps) → ex: limites par utilisateur
- On veut valider AVANT de demander une approbation humaine (deferred tools)

Signature du validator :
  def validator(ctx: RunContext[T], param1, param2, ...) -> None
  - Mêmes paramètres que le tool
  - Retourne None si OK
  - Lève ModelRetry si KO → le message est renvoyé au LLM

Disponible sur : @agent.tool, @agent.tool_plain, Tool, Tool.from_schema,
FunctionToolset. Peut être sync ou async.

Le résultat est exposé via args_valid sur FunctionToolCallEvent :
- True  → toute la validation a passé (schéma + custom)
- False → la validation a échoué
- None  → pas de validation effectuée (tool skippé ou deferred sans exécution)
"""


# =====================================================================
# PARTIE 1 : Validation cross-field avec args_validator
# =====================================================================

# Cas d'usage : une calculatrice dont la somme ne doit pas dépasser
# une limite définie dans les dépendances (deps).
# Le validator vérifie la combinaison de x et y AVANT l'exécution.

agent_calculator = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=int,  # La limite maximale pour la somme
    instructions=(
        "Tu es une calculatrice. "
        "Utilise l'outil pour additionner deux nombres. "
        "Si la validation échoue, essaie avec des nombres plus petits."
    ),
    retries=3,
)


def validate_sum_limit(ctx: RunContext[int], x: int, y: int) -> None:
    """Vérifie que la somme ne dépasse pas la limite définie dans deps."""
    if x + y > ctx.deps:
        logger.warning(f"Validation KO : {x} + {y} = {x + y} > limite {ctx.deps}")
        raise ModelRetry(f"La somme de x et y ne doit pas dépasser {ctx.deps}")


@agent_calculator.tool(args_validator=validate_sum_limit)
def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
    """Additionne deux nombres (la somme ne doit pas dépasser la limite)."""
    result = x + y
    logger.info(f"Addition : {x} + {y} = {result} (limite : {ctx.deps})")
    return result


# =====================================================================
# PARTIE 2 : Validation avec approbation humaine (deferred tools)
# =====================================================================

# Quand un tool a requires_approval=True, la validation custom
# s'exécute AVANT de demander l'approbation à l'utilisateur.
# → Le LLM corrige ses erreurs sans déranger l'humain.
# → Seuls les appels valides arrivent jusqu'à l'approbation.
#
# output_type=[str, DeferredToolRequests] signifie que l'agent peut
# soit retourner un str (réponse normale), soit un DeferredToolRequests
# (le tool est en attente d'approbation humaine).

agent_approval = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=int,  # La limite maximale pour la somme
    output_type=[str, DeferredToolRequests],
    instructions=(
        "Tu es une calculatrice sécurisée. "
        "Utilise l'outil pour additionner deux nombres."
    ),
    retries=3,
)


@agent_approval.tool(requires_approval=True, args_validator=validate_sum_limit)
def add_with_approval(ctx: RunContext[int], x: int, y: int) -> int:
    """Additionne deux nombres avec approbation humaine requise."""
    result = x + y
    logger.info(f"Addition approuvée : {x} + {y} = {result}")
    return result


# =====================================================================
# PARTIE 3 : Validation de plage de dates
# =====================================================================

# Un cas plus réaliste : un outil de réservation où la date de fin
# doit être après la date de début.

agent_booking = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant de réservation. "
        "Utilise l'outil pour créer des réservations."
    ),
    retries=3,
)


def validate_date_range(start_day: int, end_day: int) -> None:
    """Vérifie que end_day est strictement après start_day."""
    if end_day <= start_day:
        logger.warning(f"Validation KO : jour {end_day} <= jour {start_day}")
        raise ModelRetry(
            f"Le jour de fin ({end_day}) doit être après le jour de début ({start_day})."
        )


@agent_booking.tool_plain(args_validator=validate_date_range)
def book_room(room: str, start_day: int, end_day: int) -> str:
    """Réserve une salle du jour start_day au jour end_day (1-31)."""
    logger.info(f"Réservation : salle {room}, jours {start_day}-{end_day}")
    return f"Salle '{room}' réservée du jour {start_day} au jour {end_day}."


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Somme dans la limite ---
    logger.info("=== Addition avec limite à 50 ===")
    result = agent_calculator.run_sync("Additionne 20 et 15", deps=50)
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Validation + approbation humaine (deferred tool) ---
    logger.info("=== Addition avec approbation humaine (limite 100) ===")
    result = agent_approval.run_sync("Additionne 5 et 3", deps=100)
    if isinstance(result.output, DeferredToolRequests):
        # La validation custom a passé → les args sont prêts pour approbation
        for approval in result.output.approvals:
            logger.info(f"Tool en attente d'approbation : {approval.tool_name}")
            logger.info(f"Args validés : {approval.args}")
    else:
        logger.success(f"Réponse directe : {result.output}")

    # --- Démo 3 : Réservation valide ---
    logger.info("=== Réservation d'une salle ===")
    result = agent_booking.run_sync(
        "Réserve la salle 'Everest' du jour 10 au jour 15"
    )
    logger.success(f"Réponse : {result.output}")
