from pathlib import Path

import httpx
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    TextContent,
    VideoUrl,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MULTIMODAL INPUT — Image, Audio, Vidéo, Document, Texte

Pydantic AI permet d'envoyer du contenu multimodal au LLM en entrée.
Le contenu est passé comme liste dans le prompt : [texte, media, ...].

Deux façons de fournir du contenu :
1. Via URL  → ImageUrl, AudioUrl, VideoUrl, DocumentUrl
   Le provider télécharge le fichier de son côté (par défaut).
2. Via bytes → BinaryContent
   On envoie les données binaires directement.

Types disponibles :
- ImageUrl / BinaryContent(media_type='image/png')
- AudioUrl / BinaryContent(media_type='audio/mp3')
- VideoUrl / BinaryContent(media_type='video/mp4')
- DocumentUrl / BinaryContent(media_type='application/pdf')
- TextContent → texte avec métadonnées (non envoyées au modèle)

force_download :
Par défaut, l'URL est envoyée au provider qui télécharge le fichier.
Si le provider ne peut pas (restrictions d'accès, crawling bloqué…),
on peut forcer le téléchargement côté client :
  ImageUrl(url='...', force_download=True)

Support par provider (résumé) :
- OpenAI Responses  → Image ✅, Audio ✅, Document ✅, Vidéo ❌
- Anthropic         → Image ✅, Document ✅ (PDF), Audio ❌, Vidéo ❌
- Google Vertex     → Tous les types ✅
- Google GLA        → YouTube + Files API, autres URLs à télécharger
- xAI               → Image ✅, Document ✅, Audio ❌, Vidéo ❌

Note : tous les modèles ne supportent pas tous les types d'entrée.
Vérifier la doc du modèle avant utilisation.
"""


# =====================================================================
# PARTIE 1 : Image Input — via URL et via bytes
# =====================================================================

agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant visuel. Réponds en français.",
)

# --- Via URL : le provider télécharge l'image ---
# Le plus simple quand on a une URL publique.


def demo_image_url():
    result = agent.run_sync([
        "De quelle entreprise est ce logo ?",
        ImageUrl(url="https://iili.io/3Hs4FMg.png"),
    ])
    logger.success(f"ImageUrl → {result.output}")


# --- Via BinaryContent : on envoie les bytes ---
# Utile quand l'image est locale ou derrière une auth.


def demo_image_binary():
    image_response = httpx.get("https://iili.io/3Hs4FMg.png")
    result = agent.run_sync([
        "De quelle entreprise est ce logo ?",
        BinaryContent(data=image_response.content, media_type="image/png"),
    ])
    logger.success(f"BinaryContent → {result.output}")


# =====================================================================
# PARTIE 2 : Document Input — PDF via URL et via bytes
# =====================================================================

agent_doc = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant documentaire. Réponds en français.",
)


def demo_document_url():
    result = agent_doc.run_sync([
        "Quel est le contenu principal de ce document ?",
        DocumentUrl(
            url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        ),
    ])
    logger.success(f"DocumentUrl → {result.output}")


# --- Document local via BinaryContent ---
# def demo_document_local():
#     pdf_path = Path("mon_document.pdf")
#     result = agent_doc.run_sync([
#         "Résume ce document.",
#         BinaryContent(data=pdf_path.read_bytes(), media_type="application/pdf"),
#     ])
#     logger.success(f"Document local → {result.output}")


# =====================================================================
# PARTIE 3 : Audio et Vidéo Input
# =====================================================================

# Audio et Vidéo suivent le même pattern que Image et Document.
# Support limité selon le provider.

# --- Audio ---
# agent.run_sync([
#     "Transcris cet audio.",
#     AudioUrl(url="https://example.com/audio.mp3"),
# ])

# --- Vidéo ---
# agent.run_sync([
#     "Décris cette vidéo.",
#     VideoUrl(url="https://example.com/video.mp4"),
# ])


# =====================================================================
# PARTIE 4 : TextContent — texte avec métadonnées
# =====================================================================

# TextContent est comme un str, mais avec des métadonnées.
# Les métadonnées ne sont PAS envoyées au modèle, elles sont
# conservées dans les messages pour un accès programmatique.


def demo_text_content():
    result = agent.run_sync([
        "Résume les points clés de ce texte.",
        TextContent(
            content=(
                "Pydantic AI est un framework d'agents Python. "
                "Il supporte les entrées texte, image, audio, vidéo et document."
            ),
            metadata={"source": "documentation_pydantic_ai.txt"},
        ),
    ])
    logger.success(f"TextContent → {result.output}")


# =====================================================================
# PARTIE 5 : force_download — télécharger côté client
# =====================================================================

# Si le provider ne peut pas télécharger l'URL (accès restreint,
# crawling bloqué…), on force le téléchargement local.

# ImageUrl(url="https://example.com/image.png", force_download=True)
# AudioUrl(url="https://example.com/audio.mp3", force_download=True)
# VideoUrl(url="https://example.com/video.mp4", force_download=True)
# DocumentUrl(url="https://example.com/doc.pdf", force_download=True)


# =====================================================================
# PARTIE 6 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Image via URL ---
    logger.info("=== Image via URL ===")
    demo_image_url()

    # --- Démo 2 : Image via bytes ---
    logger.info("=== Image via BinaryContent ===")
    demo_image_binary()

    # --- Démo 3 : Document PDF ---
    logger.info("=== Document PDF via URL ===")
    demo_document_url()

    # --- Démo 4 : TextContent avec métadonnées ---
    logger.info("=== TextContent ===")
    demo_text_content()
