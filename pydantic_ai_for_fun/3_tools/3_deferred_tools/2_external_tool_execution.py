import asyncio
from dataclasses import dataclass
from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
EXTERNAL TOOL EXECUTION — Tools exécutés en dehors du process agent

Contrairement aux tools avec approbation (fichier 1), ici le tool
ne s'exécute PAS dans le process Python de l'agent. Le résultat
vient d'ailleurs :
- Un frontend web ou mobile
- Un worker background (Celery, asyncio task…)
- Un service externe / API tierce

Le mécanisme :
1. Le LLM appelle le tool
2. La fonction du tool lance une tâche externe puis lève CallDeferred
3. L'agent s'arrête → retourne DeferredToolRequests avec `calls`
4. On attend que la tâche externe finisse
5. On relance l'agent avec DeferredToolResults contenant les résultats

Différence avec l'approbation humaine (fichier 1) :
- Approbation → DeferredToolRequests.approvals (True/False/ToolDenied)
- Externe     → DeferredToolRequests.calls (valeur de retour ou ModelRetry)

Deux façons de définir un tool externe :
- CallDeferred dans une fonction tool → quand le tool est parfois
  local, parfois externe (selon les args ou le contexte)
- ExternalToolset → quand le tool est TOUJOURS externe et que sa
  définition (nom + schéma JSON) est fournie par un service tiers
"""


# =====================================================================
# PARTIE 1 : CallDeferred — déléguer à une tâche background
# =====================================================================

# Cas d'usage : un calcul long qu'on lance en background.
# Le tool démarre la tâche, lève CallDeferred, et l'agent s'arrête.
# On récupère le résultat plus tard.


@dataclass
class TaskResult:
    task_id: str
    result: Any


async def heavy_computation_task(task_id: str, question: str) -> TaskResult:
    """Simule un calcul lourd en background (ex: ML, data processing…)."""
    logger.info(f"[background] Tâche {task_id} démarrée pour : {question}")
    await asyncio.sleep(2)  # Simule un traitement long
    logger.info(f"[background] Tâche {task_id} terminée")
    return TaskResult(task_id=task_id, result=42)


agent_external = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[str, DeferredToolRequests],
    instructions=(
        "Tu es un assistant de calcul. "
        "Utilise l'outil pour lancer des calculs complexes."
    ),
)

# Stockage des tâches en cours
tasks: list[asyncio.Task[TaskResult]] = []


@agent_external.tool
async def calculate_answer(ctx: RunContext, question: str) -> str:
    """Lance un calcul complexe en background."""
    task_id = f"task_{len(tasks)}"
    logger.info(f"Lancement tâche background : {task_id}")

    # On lance la tâche en background
    task = asyncio.create_task(heavy_computation_task(task_id, question))
    tasks.append(task)

    # On signale que le tool est deferred → l'agent s'arrête
    # metadata permet de retrouver la tâche plus tard
    raise CallDeferred(metadata={"task_id": task_id})


# =====================================================================
# PARTIE 2 : Récupérer les résultats et relancer l'agent
# =====================================================================


async def collect_results(requests: DeferredToolRequests) -> DeferredToolResults:
    """Attend que toutes les tâches background finissent et construit les résultats."""
    # On attend que toutes les tâches soient terminées
    done, _ = await asyncio.wait(tasks)
    task_results = {r.result().task_id: r.result().result for r in done}

    results = DeferredToolResults()
    for call in requests.calls:
        try:
            # On retrouve le résultat via le task_id stocké dans metadata
            task_id = requests.metadata[call.tool_call_id]["task_id"]
            result = task_results[task_id]
            results.calls[call.tool_call_id] = result
            logger.success(f"Résultat récupéré pour {task_id} : {result}")
        except KeyError:
            # Pas de résultat → on demande au LLM de réessayer
            results.calls[call.tool_call_id] = ModelRetry(
                "Aucun résultat trouvé pour cette tâche."
            )
            logger.error(f"Pas de résultat pour {call.tool_call_id}")

    return results


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================


async def main():
    # --- Run 1 : le LLM appelle le tool → tâche lancée en background ---
    logger.info("=== Run 1 : lancement du calcul ===")
    result = await agent_external.run(
        "Calcule la réponse à la grande question sur la vie, "
        "l'univers et tout le reste"
    )
    messages = result.all_messages()

    if isinstance(result.output, DeferredToolRequests):
        logger.info(f"Tools deferred : {len(result.output.calls)} tâche(s) en cours")

        # --- On attend les résultats des tâches background ---
        deferred_results = await collect_results(result.output)

        # --- Run 2 : on relance l'agent avec les résultats ---
        logger.info("=== Run 2 : reprise avec les résultats ===")
        result = await agent_external.run(
            message_history=messages,
            deferred_tool_results=deferred_results,
        )
        logger.success(f"Réponse finale : {result.output}")
    else:
        logger.success(f"Réponse : {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
