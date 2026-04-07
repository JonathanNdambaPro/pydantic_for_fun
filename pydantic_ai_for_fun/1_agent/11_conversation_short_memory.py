
import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
LA GESTION DE L'HISTORIQUE (MEMORY)

Comment faire pour qu'un Agent IA se souvienne du contexte 
entre deux requêtes distinctes.
"""

# agent = Agent('openai:gpt-4o')

# # --- RUN 1 : L'agent est vierge ---
# resultat_1 = agent.run_sync('Qui était Albert Einstein ?')
# print(f"Agent : {resultat_1.data}")
# # Réponse : "C'était un physicien théoricien..."


# # --- RUN 2 : On fait appel à la mémoire ---
# # On pose une question qui n'a aucun sens sans le contexte ("sa").
# # On injecte l'historique du Run 1 via `resultat_1.new_messages()`.
# resultat_2 = agent.run_sync(
#     'Quelle était sa formule la plus célèbre ?',    
#     message_history=resultat_1.new_messages(),  
# )
# print(f"Agent : {resultat_2.data}")
# Réponse : "Sa formule la plus célèbre est $E=mc^2$."


agent = Agent('gateway/openai:gpt-4o', system_prompt="Tu es un assistant sarcastique.")

async def main():
    logger.info("Démarrage du Chatbot (Tapez 'exit' pour quitter)")

    # 1. On initialise une mémoire vide en dehors de la boucle
    conversation_history = []

    while True:
        # 2. On attend l'input de l'utilisateur
        user_input = input("\nToi : ")
        if user_input.lower() == 'exit':
            break

        # 3. On lance le Run en lui injectant l'historique complet
        result = await agent.run(
            user_input, 
            message_history=conversation_history
        )
        logger.info(f"Bot : {result.output}")

        # 4. LE SECRET : On écrase l'ancienne mémoire avec la nouvelle mémoire complète
        # Au prochain tour de boucle, il aura tout le contexte !
        conversation_history = result.all_messages()

if __name__ == '__main__':
    asyncio.run(main())
