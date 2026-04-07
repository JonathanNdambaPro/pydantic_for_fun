
import asyncio
from datetime import datetime

import lancedb
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext
from sentence_transformers import SentenceTransformer

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
LA GESTION DE LA MÉMOIRE LONGUE DURÉE (LONG-TERM MEMORY) AVEC LANCEDB

Approche "à la MemGPT" : l'agent décide LUI-MÊME quand sauvegarder
ou chercher dans sa mémoire via des tools.

Avant : on sauvegardait et cherchait systématiquement à chaque message.
Maintenant : l'agent a 2 tools et décide si c'est pertinent :
  - recall_memories : "est-ce que j'ai besoin de contexte passé ?"
  - save_memory : "est-ce que cette info vaut le coup d'être retenue ?"
"""

# --- Embeddings locaux ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- LanceDB : base vectorielle sur disque ---
db = lancedb.connect(".lancedb_data")
TABLE_NAME = "conversations"


def _get_table():
    if TABLE_NAME in db.list_tables().tables:
        return db.open_table(TABLE_NAME)

    return None


# --- L'agent avec ses tools de mémoire ---
agent = Agent(
    'gateway/openai:gpt-4o',
    system_prompt="""\
Tu es un assistant sarcastique avec une mémoire d'éléphant.

Tu as accès à une mémoire long-terme via tes tools :
- Utilise `recall_memories` quand l'utilisateur fait référence à quelque chose
  de passé ou quand tu as besoin de contexte ("tu te souviens ?", "comme la dernière fois", etc.)
- Utilise `save_memory` quand l'utilisateur partage une info personnelle importante
  (son nom, ses préférences, un fait marquant). Ne sauvegarde PAS les banalités.

Tu n'es PAS obligé d'utiliser ces tools à chaque message. Utilise-les uniquement quand c'est pertinent.
""",
)


@agent.tool
async def recall_memories(ctx: RunContext, query: str) -> str:
    """Cherche dans la mémoire long-terme les souvenirs pertinents.
∏
    Args:π
        query: Ce que tu cherches (ex: "le prénom de l'utilisateur", "son langage préféré")
    """
    table = _get_table()
    if table is None:
        return "Aucun souvenir en mémoire."
    vector = embedder.encode(query).tolist()
    results = table.search(vector).limit(3).to_list()
    if not results:
        return "Aucun souvenir trouvé pour cette recherche."
    memories = "\n".join(f"- {r['text']}" for r in results)
    return f"Souvenirs trouvés :\n{memories}"


@agent.tool
async def save_memory(ctx: RunContext, info: str) -> str:
    """Sauvegarde une information importante dans la mémoire long-terme.

    Utilise cet outil UNIQUEMENT pour les infos qui méritent d'être retenues
    entre les sessions (préférences, faits personnels, décisions importantes).

    Args:
        info: L'information à retenir (ex: "L'utilisateur s'appelle Jojo et adore Python")
    """
    vector = embedder.encode(info).tolist()
    row = {
        "vector": vector,
        "text": info,
        "user_msg": "",
        "bot_msg": "",
        "timestamp": datetime.now().isoformat(),
    }
    table = _get_table()
    if table is None:
        db.create_table(TABLE_NAME, [row])
    else:
        table.add([row])
    return f"Souvenir sauvegardé : {info}"


async def main():
    logger.info("Chatbot avec mémoire agentique (Tapez 'exit' pour quitter)")
    logger.info("L'agent décide lui-même quand mémoriser ou se souvenir")

    conversation_history = []

    while True:
        user_input = input("Toi : ")
        if user_input.lower() == 'exit':
            break

        result = await agent.run(
            user_input,
            message_history=conversation_history,
        )
        logger.info(f"Bot : {result.output}")

        conversation_history = result.all_messages()


if __name__ == '__main__':
    asyncio.run(main())
