import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, BinaryImage, ImageGenerationTool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
IMAGE GENERATION TOOL — Génération d'images native via le provider

ImageGenerationTool permet au LLM de générer des images directement.
Les images sont exécutées côté provider, pas par Pydantic AI.

Providers supportés :
- OpenAI Responses ✅ (support complet, modèles >= gpt-5.2)
- Google           ✅ (support limité, modèles gemini-3-pro-image-preview)
- Anthropic        ❌
- xAI              ❌

Comportement :
- Les images générées sont disponibles via result.response.images
  sous forme d'objets BinaryImage
- Avec output_type=BinaryImage, le tool est activé automatiquement
  (pas besoin de le spécifier dans builtin_tools)
- Google génère toujours des images même sans le tool explicite

Paramètres de configuration (support variable selon provider) :
- background        → 'transparent' ou opaque (OpenAI)
- quality           → 'low', 'medium', 'high' (OpenAI)
- size              → '1024x1024', '1024x1536', '1536x1024' (OpenAI)
                      '512', '1K', '2K', '4K' (Google)
- aspect_ratio      → '1:1', '2:3', '3:2' (OpenAI)
                      + '3:4', '4:3', '9:16', '16:9', '21:9' (Google)
- output_format     → 'png', 'jpeg', 'webp' (OpenAI / Google Vertex)
- output_compression → 100 par défaut OpenAI, 75 Google
- moderation        → 'low', 'auto' (OpenAI)
- input_fidelity    → fidélité de l'entrée (OpenAI)
- partial_images    → nombre d'images partielles (OpenAI)

Alternative model-agnostic : la capability ImageGeneration utilise le
tool natif quand supporté, et tombe sur une implémentation locale sinon.
"""


# =====================================================================
# PARTIE 1 : ImageGenerationTool basique
# =====================================================================

# Le LLM génère une image en plus de sa réponse textuelle.
# L'image est accessible via result.response.images.

agent_image = Agent(
    'gateway/openai-responses:gpt-5.2',
    builtin_tools=[ImageGenerationTool()],
    instructions="Tu es un conteur illustrateur. Raconte une histoire courte avec une illustration.",
)


# =====================================================================
# PARTIE 2 : Image comme output direct (output_type=BinaryImage)
# =====================================================================

# Quand on veut UNIQUEMENT l'image en sortie, on utilise
# output_type=BinaryImage. Le tool est activé automatiquement.

agent_image_only = Agent(
    'gateway/openai-responses:gpt-5.2',
    output_type=BinaryImage,
    # Pas besoin de builtin_tools=[ImageGenerationTool()]
    # → activé automatiquement avec output_type=BinaryImage
)


# =====================================================================
# PARTIE 3 : Configuration avancée (qualité, taille, format)
# =====================================================================

agent_image_configured = Agent(
    'gateway/openai-responses:gpt-5.2',
    builtin_tools=[
        ImageGenerationTool(
            background='transparent',
            quality='high',
            size='1024x1024',
            output_format='png',
            output_compression=100,
            moderation='low',
        )
    ],
    output_type=BinaryImage,
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Histoire illustrée (texte + image) ---
    logger.info("=== Histoire avec illustration ===")
    result = agent_image.run_sync(
        "Raconte une histoire de deux phrases sur un axolotl avec une illustration."
    )
    logger.success(f"Réponse : {result.output}")
    if result.response.images:
        logger.info(f"Image générée : {type(result.response.images[0])}")

    # --- Démo 2 : Image seule ---
    logger.info("=== Génération d'image seule ===")
    result = agent_image_only.run_sync("Un paysage de montagne au coucher du soleil")
    if isinstance(result.output, BinaryImage):
        logger.success("Image générée avec succès")

    # --- Démo 3 : Image configurée (haute qualité, PNG, transparent) ---
    logger.info("=== Image haute qualité ===")
    result = agent_image_configured.run_sync("Un logo minimaliste pour une app de météo")
    if isinstance(result.output, BinaryImage):
        logger.success("Image haute qualité générée")
