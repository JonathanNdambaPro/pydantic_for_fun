import asyncio

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
FILE SEARCH TOOL — Recherche vectorielle dans des fichiers (RAG managé)

FileSearchTool permet au LLM de chercher dans des fichiers uploadés
via une recherche vectorielle. C'est un système RAG complet et managé
par le provider : stockage, chunking, embeddings, injection de contexte.

Tu n'as PAS besoin de gérer les embeddings toi-même. Le provider
fait tout : découpe les fichiers, crée les vecteurs, et injecte
les passages pertinents dans le contexte du LLM.

Providers supportés :
- OpenAI Responses ✅ (support complet, fichiers dans vector stores)
- Google Gemini    ✅ (via Files API, auto-suppression après 48h,
                       max 2 Go/fichier, 20 Go/projet)
- Google Vertex AI ❌
- Anthropic        ❌
- Groq             ❌

Flow avec OpenAI :
1. Uploader un fichier via l'API Files (purpose='assistants')
2. Créer un vector store
3. Ajouter le fichier au vector store
4. Passer le vector_store_id à FileSearchTool

Flow avec Google Gemini :
1. Créer un file search store via l'API Files
2. Uploader le fichier dans le store
3. Passer le store name à FileSearchTool

Cas d'usage :
- Questions sur des documents internes (contrats, specs, docs…)
- Support client basé sur une knowledge base
- Analyse de rapports / articles uploadés
"""


# =====================================================================
# PARTIE 1 : FileSearchTool avec OpenAI (vector store)
# =====================================================================

# Le flow complet : upload du fichier → vector store → agent.
# Le vector store gère le chunking et les embeddings automatiquement.


async def demo_openai_file_search():
    """Démo complète avec OpenAI : upload + vector store + recherche."""
    model = OpenAIResponsesModel('gpt-5.2')

    # 1. Upload du fichier
    logger.info("Upload du fichier...")
    with open("exemple_document.txt", "rb") as f:
        file = await model.client.files.create(file=f, purpose="assistants")
    logger.info(f"Fichier uploadé : {file.id}")

    # 2. Créer un vector store
    vector_store = await model.client.vector_stores.create(name="mes-docs")
    logger.info(f"Vector store créé : {vector_store.id}")

    # 3. Ajouter le fichier au vector store
    await model.client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=file.id,
    )
    logger.info("Fichier ajouté au vector store")

    # 4. Créer l'agent avec FileSearchTool
    agent = Agent(
        model,
        builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        instructions="Tu es un assistant documentaire. Réponds en français.",
    )

    result = await agent.run("Que contient ce document sur Pydantic ?")
    logger.success(f"Réponse : {result.output}")


# =====================================================================
# PARTIE 2 : Utilisation simple (vector store existant)
# =====================================================================

# Si le vector store existe déjà, on passe juste l'ID.
# C'est le cas le plus courant en prod.

# agent_file_search = Agent(
#     'gateway/openai-responses:gpt-5.2',
#     builtin_tools=[
#         FileSearchTool(file_store_ids=["vs_abc123"])
#     ],
#     instructions="Cherche dans les documents pour répondre.",
# )


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # Note : cette démo nécessite un fichier "exemple_document.txt"
    # et une clé API OpenAI configurée.
    logger.info("=== File Search avec OpenAI ===")
    logger.info(
        "Pour exécuter cette démo, créez un fichier 'exemple_document.txt' "
        "et configurez votre clé API OpenAI."
    )

    # Décommenter pour lancer :
    # asyncio.run(demo_openai_file_search())
