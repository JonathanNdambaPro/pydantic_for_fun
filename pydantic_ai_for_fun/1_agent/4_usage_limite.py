import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

"""
LE CONTRÔLE DU BUDGET (TOKEN LIMITS) : ÉVITER QUE L'IA SOIT TROP BAVARDE

Pourquoi c'est indispensable ?
Chaque "token" (grosso modo, un bout de mot) généré par un modèle comme GPT-4 ou Claude 
coûte de l'argent et du temps de calcul. Si tu t'attends à une réponse "Oui/Non" 
mais que l'IA décide de t'écrire un essai philosophique de 3 pages, c'est de la 
ressource gaspillée.

Le paramètre `response_tokens_limit` agit comme un couperet. 
Si l'IA dépasse la limite de mots imposée, le système coupe la connexion et 
lève une erreur pour protéger tes finances.
"""

# 1. Configuration de l'observabilité (comme vu précédemment)
load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

# 2. Création de l'agent
agent = Agent('openai:gpt-4o')

async def main():
    logger.info("=== TEST 1 : Une réponse courte autorisée ===")
    
    # Scénario 1 : L'IA doit donner juste un mot. Elle restera sous la limite des 10 tokens.
    resultat_court = await agent.run(
        'Quelle est la capitale de l\'Italie ? Réponds uniquement par le nom de la ville.',
        # On limite strictement la réponse à 10 tokens maximum.
        usage_limits=UsageLimits(response_tokens_limit=10),
    )

    # Ça passe ! On affiche la réponse ("Rome") et les statistiques de consommation.
    logger.success(f"Réponse : {resultat_court.data}")
    logger.info(f"Consommation : {resultat_court.usage()}")

    logger.info("\n=== TEST 2 : Une réponse trop longue qui va planter ===")
    
    try:
        # Scénario 2 : On demande à l'IA d'écrire un paragraphe entier.
        # Elle va inévitablement dépasser la limite de 10 tokens.
        resultat_long = await agent.run(
            'Quelle est la capitale de l\'Italie ? Réponds avec un paragraphe complet de 5 lignes.',
            usage_limits=UsageLimits(response_tokens_limit=10),
        )
    except UsageLimitExceeded as e:
        # Le couperet tombe ! L'erreur est interceptée proprement.
        logger.error(f"[BLOQUÉ] L'IA a été coupée car elle est devenue trop bavarde : {e}")

if __name__ == "__main__":
    asyncio.run(main())