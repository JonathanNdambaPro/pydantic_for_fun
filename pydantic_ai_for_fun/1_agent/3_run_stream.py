import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext

"""
L'OBSERVABILITÉ (LOGFIRE + LOGURU) : LA TOUR DE CONTRÔLE DE TON IA

Pourquoi c'est indispensable ?
Quand tu lances ton application sur un vrai serveur, tu n'es plus derrière ton écran 
pour lire les "print". Si un utilisateur se plaint que "l'IA a répondu n'importe quoi" 
ou que "ça a pris 10 secondes", tu dois pouvoir comprendre pourquoi.

1. `logfire.instrument_pydantic_ai()` : C'est la magie de pydantic-ai. Cette ligne 
   va automatiquement chronométrer l'IA, enregistrer le prompt exact de l'utilisateur, 
   les outils utilisés, le nombre de tokens consommés, et envoyer tout ça sur un 
   tableau de bord visuel.
   
2. `logger.configure(...)` : Cela permet de prendre tes propres messages de journalisation 
   (les `logger.info` que tu écris dans ton code) et de les attacher aux graphiques 
   de Logfire. Tout est centralisé au même endroit.
"""

# 1. Chargement des variables d'environnement (ex: clés API OpenAI et Logfire)
load_dotenv()

# 2. Configuration de la surveillance de l'application
logfire.configure()
logfire.instrument_pydantic_ai() # Active l'espionnage détaillé de l'agent

# 3. Connexion de notre journal (Loguru) à la plateforme de surveillance (Logfire)
logger.configure(handlers=[logfire.loguru_handler()])

# 4. Création de l'agent
agent = Agent('openai:gpt-4o')

@agent.tool
async def obtenir_meteo(ctx: RunContext, ville: str) -> str:
    """Outil basique de météo."""
    # Au lieu d'un print, on utilise le logger. 
    # Cette information sera visible dans notre tableau de bord !
    logger.info(f"L'IA est en train de consulter la météo pour la ville : {ville}")
    return "25°C et grand soleil."

async def main():
    question = "Quel temps fait-il à Tokyo ?"
    logger.info(f"Utilisateur : {question}")

    # On log le début de l'action pour garder une trace
    logger.info("Début de la génération de la réponse par l'agent.")

    logger.info("IA : ", end='')

    # On lance le stream pour l'utilisateur
    async with agent.run_stream(question) as run:
        async for morceau_de_texte in run.stream_text():
            # On utilise le print classique pour l'affichage fluide dans la console
            logger.info(morceau_de_texte, end='', flush=True)

    logger.info("\n")
    # On log la fin de l'action
    logger.success("La réponse a été générée et envoyée avec succès à l'utilisateur.")

if __name__ == "__main__":
    asyncio.run(main())
