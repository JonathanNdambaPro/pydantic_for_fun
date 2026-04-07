import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, ConcurrencyLimit
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

"""
CONCURRENCY LIMITING & BACKPRESSURE

1. max_running : Le nombre maximum de requêtes simultanées (comme la taille d'un ThreadPool/Semaphore).
2. max_queued : Le mécanisme de "Backpressure". Fixe une limite à la file d'attente.
   Sans ça, si tu envoies 100 000 tâches asynchrones, la RAM va exploser (OOM) 
   ou les tâches vont attendre des heures. Ici, on "fail fast" si la file est pleine.
"""
load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

# Agent configuré pour la production (gestion de la charge)
agent_batch = Agent(
    'openai:gpt-4o',
    # Autorise max 10 appels LLM en parallèle.
    # Si > 10, met en attente. 
    # Si la file d'attente dépasse 100 requêtes en attente, lève une exception.
    max_concurrency=ConcurrencyLimit(max_running=10, max_queued=100),
)

async def main():
    logger.info("Lancement du batch asynchrone massif...")

    tasks = [agent_batch.run(f'Analyse le document {i}') for i in range(150)]

    try:
        # On tente de lancer 150 tâches simultanément.
        # Le système va en lancer 10, en mettre 100 en file d'attente... 
        # Et les 40 restantes vont faire exploser la limite !
        _ = await asyncio.gather(*tasks)

    except ConcurrencyLimitExceeded as e:
        logger.warning(f"Backpressure activée — le système s'est protégé : {e}")
        # Résultat attendu : Fail fast car on a dépassé le max_queued (100).

if __name__ == '__main__':
    asyncio.run(main())