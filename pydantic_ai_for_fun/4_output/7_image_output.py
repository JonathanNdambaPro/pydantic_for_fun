import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, BinaryImage

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
IMAGE OUTPUT — Image générée comme sortie de l'agent

Certains modèles peuvent générer des images (via ImageGenerationTool
ou CodeExecutionTool pour les graphiques).

Pour utiliser l'image comme sortie :
- output_type=BinaryImage → le modèle DOIT retourner une image
  → ImageGenerationTool est activé automatiquement
- output_type=BinaryImage | str → image OU texte selon la demande
  → Si le modèle génère les deux, l'image prend la priorité
  → Le texte reste accessible via result.response.text

BinaryImage contient les données binaires de l'image.

Providers supportés :
- OpenAI Responses ✅ (gpt-5.2+)
- Google           ✅ (gemini-3-pro-image-preview)
"""


# =====================================================================
# PARTIE 1 : Image obligatoire (output_type=BinaryImage)
# =====================================================================

# Le modèle est forcé de générer une image.
# ImageGenerationTool est activé automatiquement.

agent_image_only = Agent(
    'gateway/openai-responses:gpt-5.2',
    output_type=BinaryImage,
)


# =====================================================================
# PARTIE 2 : Image OU texte (union BinaryImage | str)
# =====================================================================

# Le modèle peut répondre en texte ou avec une image selon la demande.
# Si les deux sont générés, l'image est la sortie et le texte
# reste accessible via result.response.text.

agent_image_or_text = Agent(
    'gateway/openai-responses:gpt-5.2',
    output_type=BinaryImage | str,
)


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Image obligatoire ---
    logger.info("=== Image obligatoire ===")
    result = agent_image_only.run_sync("Génère une image d'un axolotl.")
    if isinstance(result.output, BinaryImage):
        logger.success("Image générée avec succès")
        # Sauvegarder : Path("axolotl.png").write_bytes(result.output.data)

    # --- Démo 2 : Texte sans image ---
    logger.info("=== Union : demande texte seul ===")
    result = agent_image_or_text.run_sync(
        "Raconte une histoire de deux phrases sur un axolotl, pas d'image."
    )
    logger.success(f"Output (str) : {result.output}")

    # --- Démo 3 : Texte + image (image prioritaire) ---
    logger.info("=== Union : demande texte + image ===")
    result = agent_image_or_text.run_sync(
        "Raconte une histoire de deux phrases sur un axolotl avec une illustration."
    )
    if isinstance(result.output, BinaryImage):
        logger.success("Output = image (prioritaire)")
        logger.info(f"Texte associé : {result.response.text}")
    else:
        logger.success(f"Output = texte : {result.output}")
