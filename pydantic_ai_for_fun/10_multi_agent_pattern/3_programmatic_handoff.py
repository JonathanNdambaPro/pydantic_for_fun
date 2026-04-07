from typing import Literal

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelMessage, RunContext, RunUsage, UsageLimits

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
PROGRAMMATIC HAND-OFF — Enchaîner les agents via le code applicatif

Le hand-off programmatique = le code applicatif (pas un agent)
décide quel agent appeler ensuite. Un humain peut être dans la boucle.

Différences avec la délégation :
- Délégation        → l'agent A appelle l'agent B via un tool
- Hand-off          → le CODE appelle A, puis B, puis C...
- Les agents n'ont pas besoin du même deps_type
- Le contrôle reste dans le code applicatif

Pattern :
1. Définir des agents spécialisés (chacun fait une chose)
2. Le code applicatif orchestre l'enchaînement
3. Utiliser message_history pour maintenir le contexte
4. Utiliser un RunUsage partagé pour le suivi global
5. Utiliser UsageLimits pour la protection
"""


# =====================================================================
# PARTIE 1 : Modèles de sortie
# =====================================================================

# Chaque agent a son propre type de sortie structuré.


class FlightDetails(BaseModel):
    """Détails d'un vol trouvé."""

    flight_number: str
    origin: str
    destination: str
    price: float


class Failed(BaseModel):
    """Impossible de trouver un choix satisfaisant."""

    reason: str = "Aucun résultat trouvé"


class SeatPreference(BaseModel):
    """Préférence de siège de l'utilisateur."""

    row: int = Field(ge=1, le=30)
    seat: Literal["A", "B", "C", "D", "E", "F"]


# =====================================================================
# PARTIE 2 : Agents spécialisés
# =====================================================================

# Agent 1 : Recherche de vol
flight_search_agent = Agent[None, FlightDetails | Failed](
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=FlightDetails | Failed,  # type: ignore
    instructions=(
        "Utilise l'outil 'flight_search' pour trouver un vol "
        "entre l'origine et la destination demandées."
    ),
)


@flight_search_agent.tool
async def flight_search(
    ctx: RunContext[None], origin: str, destination: str
) -> FlightDetails | None:
    """Simule une recherche de vol."""
    logger.info(f"[flight_search] Recherche {origin} → {destination}")
    # En réalité : appel API ou scraping
    return FlightDetails(
        flight_number="AF456",
        origin=origin,
        destination=destination,
        price=299.99,
    )


# Agent 2 : Préférence de siège
seat_preference_agent = Agent[None, SeatPreference | Failed](
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=SeatPreference | Failed,  # type: ignore
    instructions=(
        "Extrais la préférence de siège de l'utilisateur. "
        "Sièges A et F = hublot. "
        "Rang 1 = première rangée avec plus de place. "
        "Rangs 14 et 20 ont aussi plus de place pour les jambes."
    ),
)

# Limites globales partagées par tous les agents
usage_limits = UsageLimits(request_limit=15)


# =====================================================================
# PARTIE 3 : Fonctions d'orchestration
# =====================================================================

# Chaque fonction gère un agent avec sa boucle de retry/interaction.


async def find_flight(usage: RunUsage) -> FlightDetails | None:
    """Orchestre la recherche de vol avec retry."""
    prompts = [
        "Je veux un vol de Paris à Tokyo",
        "Et un vol de Lyon à New York ?",
    ]

    message_history: list[ModelMessage] | None = None

    for prompt in prompts:
        logger.info(f"[handoff] User : {prompt}")
        result = await flight_search_agent.run(
            prompt,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )

        if isinstance(result.output, FlightDetails):
            return result.output
        else:
            logger.warning(f"[handoff] Échec : {result.output.reason}")
            # On garde l'historique pour le prochain essai
            message_history = result.all_messages(
                output_tool_return_content="Essaie encore s'il te plaît."
            )

    return None


async def find_seat(usage: RunUsage) -> SeatPreference | None:
    """Orchestre la sélection de siège."""
    seat_requests = [
        "Je voudrais un siège hublot en première rangée",
        "Le siège 1A s'il vous plaît",
    ]

    message_history: list[ModelMessage] | None = None

    for request in seat_requests:
        logger.info(f"[handoff] User : {request}")
        result = await seat_preference_agent.run(
            request,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )

        if isinstance(result.output, SeatPreference):
            return result.output
        else:
            logger.warning("[handoff] Préférence non comprise, nouvel essai")
            message_history = result.all_messages()

    return None


# =====================================================================
# PARTIE 4 : Orchestration principale
# =====================================================================

# Le code applicatif enchaîne les agents dans l'ordre.
# Un RunUsage partagé suit la consommation globale.


async def book_flight():
    """Flux complet de réservation : vol → siège."""
    # Usage partagé entre tous les agents
    usage = RunUsage()

    # Étape 1 : Trouver un vol
    logger.info("--- Étape 1 : Recherche de vol ---")
    flight = await find_flight(usage)

    if flight is None:
        logger.error("Aucun vol trouvé. Abandon.")
        return

    logger.success(
        f"Vol trouvé : {flight.flight_number} "
        f"({flight.origin} → {flight.destination}) — {flight.price}€"
    )

    # Étape 2 : Choisir un siège
    logger.info("--- Étape 2 : Choix du siège ---")
    seat = await find_seat(usage)

    if seat is None:
        logger.error("Préférence de siège non comprise. Abandon.")
        return

    logger.success(f"Siège choisi : rangée {seat.row}, siège {seat.seat}")

    # Résumé final
    logger.info("--- Résumé de la réservation ---")
    logger.success(
        f"Vol {flight.flight_number} | "
        f"{flight.origin} → {flight.destination} | "
        f"Siège {seat.row}{seat.seat} | "
        f"Prix : {flight.price}€"
    )
    logger.info(f"Usage total : {usage}")


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    import asyncio

    logger.info("=== Programmatic Hand-off : Réservation de vol ===")
    asyncio.run(book_flight())
