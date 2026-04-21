import asyncio
import logfire
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from typing import Any
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
EXTERNAL LOOP (DEFERRED TOOLS) — Exécution externe et asynchrone

Pydantic AI permet de déporter l'exécution de certains outils, utile quand :
- Le résultat dépend d'un système externe (frontend, API lente, un autre pod...).
- L'outil est particulièrement chronophage et ne devrait pas bloquer l'agent de façon synchrone.

Comment ça marche ?
1. L'outil soulève une exception `CallDeferred` en lui paramétrant éventuellement des metadata (ex: un ID de tâche).
2. L'agent détecte cette exception, s'arrête prématurément et retourne un objet `DeferredToolRequests`.
3. Notre application attend que la tâche en arrière-plan se termine (ou s'arrête pour attendre un POST de webhook).
4. On relance l'agent avec l'historique des messages ET un `DeferredToolResults` contenant les réponses associées aux `tool_call_id`.
"""

# =====================================================================
# Démo : Exécution d'une tâche lourde déportée en arrière-plan
# =====================================================================

@dataclass
class TaskResult:
    task_id: str
    result: Any

# Simulation d'une tâche asynchrone longue (comme un processus Celery / Temporal ou un appel API très lent)
async def calculate_answer_task(task_id: str, question: str) -> TaskResult:
    logger.info(f"[Worker interne] Début du traitement lourd {task_id} pour : '{question}'...")
    await asyncio.sleep(2)  # Simule une attente de 2 secondes
    logger.success(f"[Worker interne] Traitement {task_id} terminé !")
    return TaskResult(task_id=task_id, result=42)

# L'agent peut renvoyer soit une chaîne (str), soit une demande de defer (DeferredToolRequests)
agent = Agent(
    'gateway/openai:gpt-4o',  # Ou utiliser 'gateway/anthropic:claude-3-5-sonnet-latest'
    output_type=[str, DeferredToolRequests],
    instructions="Tu es un assistant scientifique qui répond aux grandes questions de l'univers de manière concise."
)

# Liste globale pour stocker nos tâches asynchrones en cours (simulation d'une file d'attente / d'un état global)
tasks: list[asyncio.Task[TaskResult]] = []

@agent.tool
async def calculate_answer(ctx: RunContext, question: str) -> str:
    """Outil complexe qui calcule la réponse à une question difficile."""
    task_id = f'task_{len(tasks)}'
    
    # 1. On démarre la tâche en arrière-plan
    task = asyncio.create_task(calculate_answer_task(task_id, question))
    tasks.append(task)
    
    logger.info(f"⚙️  Outil appelé par l'agent. Délégation de la tâche sous l'id '{task_id}'.")
    
    # 2. On interrompt l'outil (et donc l'agent) au moyen de CallDeferred.
    # Les metadata servent de passe-plat : elles permettent de conserver l'information métier (task_id)
    # pour, plus tard, relier notre réponse métier au 'tool_call_id' réclamé par le LLM.
    raise CallDeferred(metadata={'task_id': task_id})


async def main():
    logger.info("=== Étape 1 : Run initial de l'agent ===")
    prompt = "Calcule la réponse à l'ultime question sur la vie, l'univers et le reste."
    
    # Lancement initial : l'agent fera appel à l'outil puis se mettra en pause (exception CallDeferred interceptée)
    result = await agent.run(prompt)
    messages = result.all_messages()
    
    # L'output doit être de type DeferredToolRequests car CallDeferred a été levé.
    assert isinstance(result.output, DeferredToolRequests)
    requests = result.output
    
    logger.warning("⏸️  L'agent s'est mis en pause en attente de la résolution d'outils externes.")
    for call in requests.calls: # Note: contrairement à approvals, on lit requests.calls !
        # On inspecte les metadata stockées depuis CallDeferred, en utilisant le 'tool_call_id' natif
        metadata = requests.metadata.get(call.tool_call_id, {})
        task_id = metadata.get('task_id')
        logger.warning(f"  - Attente résultat pour l'outil : {call.tool_name} (Task ID = {task_id})")

    # --- On simule l'attente ou la pause du processus ---
    logger.info("=== Étape 2 : Le système attend que la tâche en arrière-plan se termine ===")
    done, _ = await asyncio.wait(tasks)
    
    # Récupération des résultats une fois la tâche complétée
    task_results = [task.result() for task in done]
    task_results_by_task_id = {res.task_id: res.result for res in task_results}
    
    logger.info("=== Étape 3 : Empaquetage des résultats et relance de l'agent ===")
    results = DeferredToolResults()
    
    for call in requests.calls:
        try:
            # On retrouve notre `task_id` qu'on avait stocké
            task_id = requests.metadata[call.tool_call_id]['task_id']
            # Puis on récupère son vrai résultat métier
            tool_result = task_results_by_task_id[task_id]
            logger.success(f"✔️  Résultat exfiltré pour la tâche {task_id} : {tool_result}")
        except KeyError:
            # S'il y a eu un problème avec le worker ou si la tâche a expiré, on peut renvoyer un ModelRetry
            logger.error("❌  Aucun résultat trouvé pour cette tâche !")
            tool_result = ModelRetry('Aucun résultat trouvé pour cette tâche.')
            
        # On associe le résultat identifié au 'tool_call_id' que l'agent attend
        results.calls[call.tool_call_id] = tool_result
        
    # Deuxième run complet : on restitue le contexte et on fournit l'objet résultats différé (deferred_tool_results)
    final_result = await agent.run(
        message_history=messages, 
        deferred_tool_results=results
    )
    
    logger.success("=== Réponse finale de l'agent ===")
    print(final_result.output)

if __name__ == "__main__":
    asyncio.run(main())
