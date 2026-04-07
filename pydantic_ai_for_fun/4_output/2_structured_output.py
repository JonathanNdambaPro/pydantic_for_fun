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
STRUCTURED OUTPUT — Forcer le modèle à retourner des données structurées

output_type accepte un large éventail de types :
- Scalaires         → int, float, bool, str
- Collections       → list[T], dict[K, V], TypedDict, StructuredDict
- Modèles           → BaseModel, dataclass
- Unions            → Foo | Bar ou [Foo, Bar]
- En gros, tout ce que Pydantic supporte comme type hint

Comment ça marche sous le capot :
1. Pydantic AI utilise le tool calling du modèle pour forcer la structure
2. Chaque type dans une union est enregistré comme un output tool séparé
   → réduit la complexité du schéma, meilleure fiabilité
3. Si le schéma n'est pas un "object" (ex: int, list[int]),
   il est wrappé dans un objet à un seul champ
4. La validation se fait via Pydantic (JSON schema + validation)

Comportement selon output_type :
- Pas de output_type → texte brut (str)
- str dans la liste   → texte brut OU structuré (le modèle choisit)
- Pas de str          → le modèle est FORCÉ de retourner du structuré

AgentRunResult est générique sur le type de sortie → ton IDE/type
checker connaît le type exact de result.output.
"""


# =====================================================================
# PARTIE 1 : Union de types — texte OU structuré
# =====================================================================

# Le modèle peut répondre en texte (str) si les données sont
# incomplètes, ou en structuré (Box) si tout est OK.
# Très utile pour les cas où on veut un fallback conversationnel.


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent_box = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[Box, str],
    instructions=(
        "Extrais les dimensions d'une boîte. "
        "Si tu ne peux pas extraire toutes les données, "
        "demande à l'utilisateur de préciser."
    ),
)


# =====================================================================
# PARTIE 2 : Union de types non-objet (list[str] | list[int])
# =====================================================================

# Même avec des types non-objet, Pydantic AI les wrappe
# automatiquement dans un objet pour le modèle.

agent_extract = Agent[None, list[str] | list[int]](
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=list[str] | list[int],  # type: ignore
    instructions="Extrais soit les couleurs soit les tailles des formes données.",
)


# =====================================================================
# PARTIE 3 : Types scalaires (int, bool)
# =====================================================================

# On peut aussi forcer un type scalaire simple.

agent_count = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=int,
    instructions="Compte le nombre d'éléments mentionnés.",
)

agent_check = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=bool,
    instructions="Réponds vrai ou faux à la question posée.",
)


# =====================================================================
# PARTIE 4 : Modèle structuré complexe avec validation Pydantic
# =====================================================================

# Pydantic valide automatiquement la sortie du modèle.
# Si le modèle retourne des données invalides, Pydantic lève une erreur.


class Recipe(BaseModel):
    name: str
    servings: int
    prep_time_minutes: int
    ingredients: list[str]
    steps: list[str]


agent_recipe = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Recipe,
    instructions="Tu es un chef cuisinier. Donne une recette structurée.",
)


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Données incomplètes → texte ---
    logger.info("=== Box : données incomplètes ===")
    result = agent_box.run_sync("La boîte fait 10x20x30")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 2 : Données complètes → structuré ---
    logger.info("=== Box : données complètes ===")
    result = agent_box.run_sync("La boîte fait 10x20x30 cm")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 3 : Extraction couleurs (list[str]) ---
    logger.info("=== Extraction couleurs ===")
    result = agent_extract.run_sync("carré rouge, cercle bleu, triangle vert")
    logger.success(f"Output : {result.output}")

    # --- Démo 4 : Extraction tailles (list[int]) ---
    logger.info("=== Extraction tailles ===")
    result = agent_extract.run_sync("carré taille 10, cercle taille 20, triangle taille 30")
    logger.success(f"Output : {result.output}")

    # --- Démo 5 : Type scalaire int ---
    logger.info("=== Comptage (int) ===")
    result = agent_count.run_sync("J'ai un chat, deux chiens et trois poissons")
    logger.success(f"Output : {result.output}")

    # --- Démo 6 : Type scalaire bool ---
    logger.info("=== Vrai/Faux (bool) ===")
    result = agent_check.run_sync("Est-ce que Paris est la capitale de la France ?")
    logger.success(f"Output : {result.output}")

    # --- Démo 7 : Recette structurée ---
    logger.info("=== Recette structurée ===")
    result = agent_recipe.run_sync("Donne-moi une recette de crêpes")
    logger.success(f"Output : {result.output.name} ({result.output.servings} personnes)")
    logger.info(f"Ingrédients : {result.output.ingredients}")
    logger.info(f"Étapes : {len(result.output.steps)} étapes")
