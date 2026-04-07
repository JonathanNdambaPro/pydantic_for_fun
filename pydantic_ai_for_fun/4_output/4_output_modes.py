import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, PromptedOutput, ToolOutput

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
OUTPUT MODES — 3 méthodes pour obtenir des données structurées

Pydantic AI implémente 3 modes différents pour forcer le modèle
à retourner des données structurées :

1. Tool Output (par défaut)
   → Le schéma JSON est fourni comme paramètre d'un "output tool"
   → Le modèle fait un tool call pour retourner la sortie
   → Supporté par quasi tous les modèles, fiable
   → Personnalisable : nom, description, strict mode via ToolOutput()

2. Native Output
   → Utilise le "Structured Outputs" natif du provider
   → Le modèle est FORCÉ de produire du JSON valide selon le schéma
   → Pas supporté par tous les modèles (ex: Gemini ne peut pas
     utiliser tools + structured output en même temps)
   → Via NativeOutput()

3. Prompted Output
   → Le schéma JSON est injecté dans les instructions du modèle
   → Le modèle interprète les instructions (pas de contrainte forcée)
   → Fonctionne avec TOUS les modèles, mais moins fiable
   → Si le provider supporte "JSON Mode", il est activé
   → Pydantic valide le résultat et demande de réessayer si invalide
   → Via PromptedOutput()

Recommandation :
- Commencer par Tool Output (par défaut, fiable)
- Native Output si le provider le supporte et qu'on veut du natif
- Prompted Output en dernier recours ou si meilleure qualité observée

end_strategy (pour Tool Output en parallèle) :
- 'early' (défaut)     → dès qu'un output tool valide est trouvé, stop
- 'exhaustive'         → exécute aussi les function tools (side effects)
"""


# =====================================================================
# Modèles partagés
# =====================================================================


class Fruit(BaseModel):
    """Un fruit avec son nom et sa couleur."""

    name: str
    color: str


class Vehicle(BaseModel):
    """Un véhicule avec son nom et son nombre de roues."""

    name: str
    wheels: int


class Device(BaseModel):
    """Un appareil avec son nom et son type."""

    name: str
    kind: str


# =====================================================================
# PARTIE 1 : Tool Output (par défaut)
# =====================================================================

# Le schéma est envoyé comme output tool. On peut personnaliser
# le nom et la description via ToolOutput().

agent_tool_output = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[
        ToolOutput(Fruit, name="return_fruit"),
        ToolOutput(Vehicle, name="return_vehicle"),
    ],
)


# =====================================================================
# PARTIE 2 : Native Output
# =====================================================================

# Utilise le Structured Outputs natif du provider.
# Le modèle est forcé de produire du JSON valide.
# Attention : pas compatible avec tous les providers/modèles.

agent_native_output = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=NativeOutput(
        [Fruit, Vehicle],
        name="Fruit_or_vehicle",
        description="Retourne un fruit ou un véhicule.",
    ),
)


# =====================================================================
# PARTIE 3 : Prompted Output
# =====================================================================

# Le schéma est injecté dans les instructions. Le modèle doit
# interpréter et suivre les instructions. Moins fiable mais
# fonctionne partout. On peut personnaliser le template.

agent_prompted_output = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=PromptedOutput(
        [Vehicle, Device],
        name="Vehicle or device",
        description="Retourne un véhicule ou un appareil.",
    ),
)

# Avec un template custom :
agent_prompted_custom = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=PromptedOutput(
        [Vehicle, Device],
        template="Réponds en JSON selon ce schéma : {schema}",
    ),
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Tool Output ---
    logger.info("=== Tool Output ===")
    result = agent_tool_output.run_sync("Qu'est-ce qu'une banane ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    result = agent_tool_output.run_sync("Qu'est-ce qu'une Tesla ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 2 : Native Output ---
    logger.info("=== Native Output ===")
    result = agent_native_output.run_sync("Qu'est-ce qu'un Ford Explorer ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 3 : Prompted Output ---
    logger.info("=== Prompted Output ===")
    result = agent_prompted_output.run_sync("Qu'est-ce qu'un MacBook ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 4 : Prompted Output avec template custom ---
    logger.info("=== Prompted Output (template custom) ===")
    result = agent_prompted_custom.run_sync("Qu'est-ce qu'un vélo ?")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")
