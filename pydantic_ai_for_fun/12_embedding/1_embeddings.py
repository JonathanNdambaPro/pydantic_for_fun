import asyncio

import logfire
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
EMBEDDINGS — Représentations vectorielles de texte

Les embeddings transforment du texte en vecteurs numériques qui
capturent le sens sémantique. Pydantic AI fournit une interface
unifiée via la classe Embedder pour générer des embeddings
avec différents providers.

Cas d'usage :
- Recherche sémantique → trouver des documents par sens, pas par mots-clés
- RAG (Retrieval-Augmented Generation) → récupérer du contexte pertinent
- Détection de similarité → trouver des doublons, clusteriser du contenu
- Classification → utiliser les embeddings comme features ML

Points clés :
- embed_query() → pour les requêtes de recherche
- embed_documents() → pour le contenu à indexer
- EmbeddingResult → accès par index ou par texte original
- Contrôle des dimensions → réduire la taille des vecteurs
- Providers : OpenAI, Google, Cohere, VoyageAI, Bedrock, Sentence Transformers
"""


# =====================================================================
# PARTIE 1 : Quick Start — Embeddings avec OpenAI
# =====================================================================

# L'interface principale est la classe Embedder.
# On passe un identifiant de modèle au format 'provider:model'.

embedder = Embedder('openai:text-embedding-3-small')


async def demo_quickstart():
    # Embed une requête de recherche
    result = await embedder.embed_query('Qu\'est-ce que le machine learning ?')
    logger.success(f"Dimensions du vecteur : {len(result.embeddings[0])}")

    # Embed plusieurs documents d'un coup
    docs = [
        'Le machine learning est un sous-domaine de l\'IA.',
        'Le deep learning utilise des réseaux de neurones.',
        'Python est un langage de programmation.',
    ]
    result = await embedder.embed_documents(docs)
    logger.success(f"Documents embeddés : {len(result.embeddings)}")


# =====================================================================
# PARTIE 2 : EmbeddingResult — Accéder aux données
# =====================================================================

# Toutes les méthodes embed retournent un EmbeddingResult.
# On peut accéder aux embeddings par index ou par texte original.


async def demo_embedding_result():
    result = await embedder.embed_query('Bonjour le monde')

    # Accès par index via .embeddings
    embedding = result.embeddings[0]
    logger.info(f"Via .embeddings[0] → {len(embedding)} dimensions")

    # Accès direct via __getitem__ (index)
    embedding = result[0]
    logger.info(f"Via result[0] → {len(embedding)} dimensions")

    # Accès par texte original
    embedding = result['Bonjour le monde']
    logger.info(f"Via result['Bonjour le monde'] → {len(embedding)} dimensions")

    # Métadonnées d'usage
    logger.success(f"Tokens utilisés : {result.usage.input_tokens}")

    # Coût (nécessite genai-prices)
    try:
        cost = result.cost()
        logger.info(f"Coût : ${cost.total_price:.6f}")
    except Exception as e:
        logger.warning(f"Calcul du coût indisponible : {e}")


# =====================================================================
# PARTIE 3 : Contrôle des dimensions
# =====================================================================

# Les modèles text-embedding-3-* d'OpenAI supportent la réduction
# de dimensions via EmbeddingSettings. Moins de dimensions = vecteurs
# plus légers, stockage moins coûteux, recherche plus rapide.

embedder_small = Embedder(
    'openai:text-embedding-3-small',
    settings=EmbeddingSettings(dimensions=256),
)


async def demo_dimensions():
    # Dimensions par défaut (1536)
    result_full = await embedder.embed_query('Test de dimensions')
    logger.info(f"Dimensions par défaut : {len(result_full.embeddings[0])}")

    # Dimensions réduites (256)
    result_reduced = await embedder_small.embed_query('Test de dimensions')
    logger.info(f"Dimensions réduites : {len(result_reduced.embeddings[0])}")

    logger.success(
        f"Ratio de compression : {len(result_full.embeddings[0]) / len(result_reduced.embeddings[0])}x"
    )


# =====================================================================
# PARTIE 4 : Embeddings locaux avec Sentence Transformers
# =====================================================================

# Sentence Transformers tourne en local — pas d'API, pas de coût,
# données qui restent sur la machine. Idéal pour le dev et la privacy.
# Le modèle est téléchargé depuis Hugging Face au premier usage.

# Note : nécessite pydantic-ai-slim[sentence-transformers]
# uv add 'pydantic-ai-slim[sentence-transformers]'

embedder_local = Embedder('sentence-transformers:all-MiniLM-L6-v2')


async def demo_local_embeddings():
    result = await embedder_local.embed_query('Bonjour le monde')
    logger.success(
        f"Embedding local → {len(result.embeddings[0])} dimensions (all-MiniLM-L6-v2)"
    )

    # Embed de documents en batch
    docs = [
        'Les chats sont des animaux domestiques.',
        'Python est populaire en data science.',
        'La Tour Eiffel est à Paris.',
    ]
    result = await embedder_local.embed_documents(docs)
    logger.success(f"Documents embeddés localement : {len(result.embeddings)}")


# =====================================================================
# PARTIE 5 : Cas d'usage — Similarité sémantique
# =====================================================================

# L'application la plus courante : comparer le sens de textes.
# On calcule la similarité cosinus entre les vecteurs.


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    a_arr, b_arr = np.array(a), np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


async def demo_similarity():
    query = 'Comment fonctionne l\'intelligence artificielle ?'
    documents = [
        'L\'IA utilise des algorithmes pour apprendre à partir de données.',
        'La recette du gâteau au chocolat nécessite 200g de farine.',
        'Le machine learning est une branche de l\'intelligence artificielle.',
        'Les réseaux de neurones imitent le fonctionnement du cerveau.',
        'Le football est le sport le plus populaire au monde.',
    ]

    # Embed la requête et les documents
    query_result = await embedder.embed_query(query)
    docs_result = await embedder.embed_documents(documents)

    query_embedding = query_result.embeddings[0]

    # Calcule la similarité de chaque document avec la requête
    logger.info(f"Requête : '{query}'")
    logger.info("---")

    scores = []
    for i, doc in enumerate(documents):
        score = cosine_similarity(query_embedding, docs_result.embeddings[i])
        scores.append((score, doc))

    # Trie par score décroissant
    scores.sort(reverse=True)
    for score, doc in scores:
        emoji = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
        logger.success(f"{emoji} {score:.4f} → {doc}")


# =====================================================================
# PARTIE 6 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Quick start ---
    logger.info("=== Quick Start ===")
    asyncio.run(demo_quickstart())

    # --- Démo 2 : EmbeddingResult ---
    logger.info("=== EmbeddingResult ===")
    asyncio.run(demo_embedding_result())

    # --- Démo 3 : Contrôle des dimensions ---
    logger.info("=== Dimensions ===")
    asyncio.run(demo_dimensions())

    # --- Démo 4 : Embeddings locaux ---
    logger.info("=== Embeddings locaux (Sentence Transformers) ===")
    asyncio.run(demo_local_embeddings())

    # --- Démo 5 : Similarité sémantique ---
    logger.info("=== Similarité sémantique ===")
    asyncio.run(demo_similarity())
