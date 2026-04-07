import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
OUTPUT VALIDATORS — Validation async/custom de la sortie

@agent.output_validator permet d'ajouter une validation APRÈS que
le modèle a retourné sa sortie. Contrairement aux validators Pydantic
(qui sont sync et sur les champs), les output validators peuvent :
- Être async (appels réseau, BDD…)
- Faire de l'IO (vérifier en BDD, appeler une API…)
- Lever ModelRetry pour demander au modèle de réessayer

Le flow :
1. Le modèle retourne une sortie
2. Pydantic valide le schéma (types, champs…)
3. @agent.output_validator valide la logique métier
4. Si ModelRetry → le modèle réessaie
5. Si OK → la sortie est retournée

Quand utiliser quoi :
- Validation simple sur un champ   → @field_validator (Pydantic)
- Validation avec IO/async          → @agent.output_validator
- Validation différente par type    → output functions (pas de isinstance)
- Transformation de texte brut      → TextOutput

partial_output (streaming) :
- En streaming, le validator est appelé plusieurs fois
- ctx.partial_output == True → sortie partielle (skip la validation)
- ctx.partial_output == False → sortie finale (valider)
"""


# =====================================================================
# PARTIE 1 : Output validator basique — vérification de longueur
# =====================================================================

agent_text = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un conteur. Raconte une histoire en français.",
)


@agent_text.output_validator
def validate_length(ctx: RunContext, output: str) -> str:
    """Vérifie que la réponse fait au moins 100 caractères."""
    if ctx.partial_output:
        return output  # Pas de validation sur les sorties partielles

    if len(output) < 100:
        logger.warning(f"Réponse trop courte ({len(output)} chars), retry...")
        raise ModelRetry("Ta réponse est trop courte. Développe davantage.")

    logger.info(f"Validation OK : {len(output)} caractères")
    return output


# =====================================================================
# PARTIE 2 : Output validator async — simulation de vérification BDD
# =====================================================================

# Cas d'usage : le modèle génère une requête SQL, on vérifie
# qu'elle est valide en l'exécutant avec EXPLAIN.


class Success(BaseModel):
    sql_query: str


class InvalidRequest(BaseModel):
    error_message: str


Output = Success | InvalidRequest

agent_sql = Agent[None, Output](
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Output,  # type: ignore
    instructions="Génère des requêtes SQL PostgreSQL basées sur la demande.",
)


async def fake_db_explain(query: str) -> bool:
    """Simule un EXPLAIN sur une BDD. Refuse les DROP et DELETE."""
    logger.info(f"EXPLAIN : {query}")
    if "DROP" in query.upper() or "DELETE" in query.upper():
        raise ValueError(f"Requête dangereuse détectée : {query}")
    return True


@agent_sql.output_validator
async def validate_sql(ctx: RunContext, output: Output) -> Output:
    """Vérifie la requête SQL en la passant à EXPLAIN."""
    if isinstance(output, InvalidRequest):
        return output

    try:
        await fake_db_explain(output.sql_query)
        logger.success(f"SQL valide : {output.sql_query}")
    except ValueError as e:
        raise ModelRetry(f"Requête invalide : {e}") from e

    return output


# =====================================================================
# PARTIE 3 : Output validator sur données structurées
# =====================================================================

# Validation métier qui ne peut pas se faire dans un field_validator
# (ex: vérifier la cohérence entre plusieurs champs).


class Event(BaseModel):
    name: str
    start_hour: int
    end_hour: int


agent_event = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Event,
    instructions="Crée un événement avec les horaires demandés (format 24h).",
)


@agent_event.output_validator
def validate_event(ctx: RunContext, output: Event) -> Event:
    """Vérifie que l'heure de fin est après l'heure de début."""
    if output.end_hour <= output.start_hour:
        raise ModelRetry(
            f"L'heure de fin ({output.end_hour}h) doit être après "
            f"l'heure de début ({output.start_hour}h)."
        )

    logger.info(f"Événement valide : {output.name} {output.start_hour}h-{output.end_hour}h")
    return output


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Validation de longueur ---
    logger.info("=== Validation longueur (min 100 chars) ===")
    result = agent_text.run_sync("Raconte-moi une histoire sur un chat")
    logger.success(f"Output ({len(result.output)} chars) : {result.output[:100]}...")

    # --- Démo 2 : Validation SQL async ---
    logger.info("=== Validation SQL ===")

    async def run_sql_demo():
        result = await agent_sql.run("Donne-moi tous les utilisateurs actifs hier")
        logger.success(f"Output : {result.output}")

    asyncio.run(run_sql_demo())

    # --- Démo 3 : Validation cohérence horaire ---
    logger.info("=== Validation événement ===")
    result = agent_event.run_sync("Crée une réunion 'Standup' de 9h à 10h")
    logger.success(f"Output : {result.output}")
