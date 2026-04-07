import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
OUTPUT — La sortie finale d'un agent

"Output" = la valeur finale retournée par un run d'agent.
Elle peut être :
- Du texte brut (str, par défaut)
- Des données structurées (BaseModel, dataclass…)
- Une image (BinaryImage)
- Le résultat d'un function call

La sortie est wrappée dans AgentRunResult (sync) ou StreamedRunResult
(stream) pour accéder à d'autres données :
- result.output       → la valeur de sortie
- result.usage()      → tokens consommés, nombre de requêtes
- result.all_messages() → historique complet des messages

Un run se termine quand :
- Le modèle répond avec un des output_type définis
- Le modèle répond en texte brut (si str est autorisé)
- Les limites d'usage sont dépassées (UsageLimits)

output_type permet de forcer le modèle à répondre avec un format
structuré précis. Pydantic AI utilise l'API tools/functions du
modèle pour contraindre la sortie.
"""


# =====================================================================
# PARTIE 1 : Sortie texte brut (par défaut)
# =====================================================================

# Sans output_type, l'agent retourne du texte brut (str).

agent_text = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Tu es un assistant. Réponds en français.",
)


# =====================================================================
# PARTIE 2 : Sortie structurée avec un BaseModel
# =====================================================================

# On force le modèle à retourner un objet structuré.
# Pydantic AI valide automatiquement la sortie.


class CityLocation(BaseModel):
    city: str
    country: str


agent_city = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=CityLocation,
)


# =====================================================================
# PARTIE 3 : Sortie structurée complexe
# =====================================================================


class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    summary: str
    pros: list[str]
    cons: list[str]


agent_review = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=MovieReview,
    instructions="Tu es un critique de cinéma. Analyse le film demandé.",
)


# =====================================================================
# PARTIE 4 : Plusieurs output_type possibles
# =====================================================================

# On peut définir plusieurs types de sortie possibles.
# Le modèle choisit le plus adapté selon la question.


class WeatherInfo(BaseModel):
    city: str
    temperature: float
    conditions: str


agent_multi = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[str, CityLocation, WeatherInfo],
    instructions=(
        "Tu es un assistant polyvalent. "
        "Réponds avec le format le plus adapté à la question."
    ),
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Texte brut ---
    logger.info("=== Sortie texte brut ===")
    result = agent_text.run_sync("Dis bonjour")
    logger.success(f"Output : {result.output}")
    logger.info(f"Usage : {result.usage()}")

    # --- Démo 2 : Sortie structurée (CityLocation) ---
    logger.info("=== Sortie structurée : CityLocation ===")
    result = agent_city.run_sync("Où se sont tenus les JO en 2012 ?")
    logger.success(f"Output : {result.output}")
    logger.info(f"City : {result.output.city}, Country : {result.output.country}")

    # --- Démo 3 : Sortie structurée complexe (MovieReview) ---
    logger.info("=== Sortie structurée : MovieReview ===")
    result = agent_review.run_sync("Critique le film Inception")
    logger.success(f"Output : {result.output}")
    logger.info(f"Titre : {result.output.title}, Note : {result.output.rating}")
    logger.info(f"Pros : {result.output.pros}")
    logger.info(f"Cons : {result.output.cons}")

    # --- Démo 4 : Plusieurs output_type ---
    logger.info("=== Multi output_type ===")
    result = agent_multi.run_sync("Où est la Tour Eiffel ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    result = agent_multi.run_sync("Raconte-moi une blague")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")
