import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, UsageLimitExceeded, UsageLimits

"""
LE FREIN D'URGENCE (USAGE LIMITS) : POURQUOI C'EST UTILE ?

Dans la vraie vie, un agent IA peut se retrouver coincé dans une boucle infinie pour plusieurs raisons :
1. Une API externe (comme une base de données ou un service météo) est en panne et renvoie des erreurs en boucle.
2. Le modèle de langage (LLM) "hallucine" et n'arrive pas à comprendre comment formater les arguments d'un outil, se trompant encore et encore.

Si vous laissez un agent tourner en roue libre sans limite :
- Il va consommer tout votre quota d'API (et potentiellement vous coûter beaucoup d'argent).
- Il va bloquer votre serveur qui attendra indéfiniment une réponse.

Ce script démontre comment `UsageLimits(request_limit=...)` agit comme un "disjoncteur". 
Ici, on force volontairement l'IA à faire une erreur en boucle, mais le programme s'arrête 
proprement après 3 tentatives au lieu de tourner à l'infini.
"""

# 1. Chargement des variables d'environnement (ex: clés API OpenAI et Logfire)
load_dotenv()

# 2. Configuration de la surveillance de l'application
logfire.configure()
logfire.instrument_pydantic_ai() # Active l'espionnage détaillé de l'agent

# 3. Connexion de notre journal (Loguru) à la plateforme de surveillance (Logfire)
logger.configure(handlers=[logfire.loguru_handler()])

# 1. On crée un format de réponse final absurde pour empêcher l'IA de terminer normalement
class FormatImpossible(BaseModel):
    ne_jamais_utiliser_ce_champ: str

# 2. On configure l'agent pour qu'il soit forcé de toujours utiliser un outil
agent = Agent(
    'openai:gpt-4o',
    retries=3,  # Nombre de tentatives globales de l'agent en cas d'erreur de format
    output_type=FormatImpossible,
    instructions=(
        "Tu ne dois jamais donner de réponse finale. "
        "À chaque fois que tu réfléchis, tu dois utiliser l'outil `outil_qui_plante`."
    ),
)

# 3. On crée un outil très simple (tool_plain) qui renvoie systématiquement une erreur
@agent.tool_plain(retries=5)
def outil_qui_plante() -> int:
    """Outil qui fait exprès de planter pour forcer l'IA à réessayer."""
    logger.info("-> L'outil a été appelé, mais il renvoie une erreur à l'IA...")
    # ModelRetry dit à l'IA : "Ça n'a pas marché, réessaie de l'utiliser."
    raise ModelRetry('Erreur volontaire de l\'outil. Veuillez réessayer.')


async def main():
    logger.info("Démarrage de l'agent. Tentative de boucle infinie en cours...\n")

    try:
        # 4. LE SECRET EST ICI : usage_limits=UsageLimits(request_limit=3)
        # Quoi qu'il arrive, l'agent ne pourra pas faire plus de 3 allers-retours avec le serveur.
        await agent.run(
            'Commence ton travail et appelle ton outil !',
            usage_limits=UsageLimits(request_limit=3)
        )

    except UsageLimitExceeded as e:
        # 5. Le disjoncteur saute : on attrape l'erreur proprement
        logger.success("\n[SUCCÈS] Le disjoncteur a fonctionné !")
        logger.error(f"Détail de l'arrêt : {e}")

if __name__ == '__main__':
    asyncio.run(main())