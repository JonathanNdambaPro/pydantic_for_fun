import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TOOL SCHEMA (Schéma des outils)

Pydantic AI extrait automatiquement le schéma JSON des paramètres
d'un tool à partir de la signature de la fonction.

Points clés :
- Tous les paramètres (sauf RunContext) sont inclus dans le schéma
- Les docstrings sont parsées (grâce à griffe) pour extraire les
  descriptions des paramètres → ajoutées au schéma
- Formats de docstring supportés : Google, NumPy, Sphinx
- On peut forcer le format avec `docstring_format='google'`
- On peut exiger des descriptions avec `require_parameter_descriptions=True`
  → lève UserError si une description manque

Cas spécial : si un tool a un seul paramètre qui est un objet
(dataclass, TypedDict, BaseModel), le schéma est simplifié pour
être directement celui de cet objet.

Debugging :
En instrumentant avec Logfire, on peut voir les arguments passés,
les retours, le temps d'exécution et les erreurs de chaque tool.
"""


# =====================================================================
# PARTIE 1 : Extraction du schéma depuis la signature + docstring
# =====================================================================

# On utilise FunctionModel pour intercepter ce que le modèle recevrait
# et inspecter le schéma du tool sans faire d'appel LLM réel.

agent_schema = Agent()


@agent_schema.tool_plain(
    docstring_format='google',
    require_parameter_descriptions=True,
)
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'


def print_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel callback : affiche le schéma du tool tel que le modèle le voit."""
    tool = info.function_tools[0]

    logger.info(f"Description du tool : {tool.description}")
    # > Get me foobar.

    logger.info(f"Schéma JSON des paramètres :\n{tool.parameters_json_schema}")
    # {
    #     'additionalProperties': False,
    #     'properties': {
    #         'a': {'description': 'apple pie', 'type': 'integer'},
    #         'b': {'description': 'banana cake', 'type': 'string'},
    #         'c': {
    #             'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
    #             'description': 'carrot smoothie',
    #             'type': 'object',
    #         },
    #     },
    #     'required': ['a', 'b', 'c'],
    #     'type': 'object',
    # }

    return ModelResponse(parts=[TextPart('foobar')])


# =====================================================================
# PARTIE 2 : Schéma simplifié avec un paramètre unique (BaseModel)
# =====================================================================

# Quand un tool a un seul paramètre qui est un BaseModel (ou dataclass,
# TypedDict), Pydantic AI utilise directement le schéma de ce modèle
# comme schéma du tool → plus propre, moins de nesting.

agent_single = Agent()


class Foobar(BaseModel):
    """This is a Foobar"""

    x: int
    y: str
    z: float = 3.14


@agent_single.tool_plain
def foobar_single(f: Foobar) -> str:
    return str(f)


# =====================================================================
# PARTIE 3 : Exécution — inspection des schémas
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : FunctionModel pour voir le schéma envoyé au LLM ---
    logger.info("=== Schéma extrait de la docstring Google ===")
    agent_schema.run_sync("hello", model=FunctionModel(print_schema))

    # --- Démo 2 : TestModel pour inspecter le schéma simplifié ---
    logger.info("=== Schéma simplifié (paramètre unique BaseModel) ===")
    test_model = TestModel()
    result = agent_single.run_sync("hello", model=test_model)
    logger.info(f"Résultat : {result.output}")

    # Inspecter le schéma tel qu'il a été envoyé au modèle
    function_tools = test_model.last_model_request_parameters.function_tools
    for tool in function_tools:
        logger.info(f"Tool: {tool.name}")
        logger.info(f"Description: {tool.description}")
        logger.info(f"Schema: {tool.parameters_json_schema}")
        # Le schéma est directement celui de Foobar (pas de wrapper) :
        # {
        #     'properties': {
        #         'x': {'type': 'integer'},
        #         'y': {'type': 'string'},
        #         'z': {'default': 3.14, 'type': 'number'},
        #     },
        #     'required': ['x', 'y'],
        #     'title': 'Foobar',
        #     'type': 'object',
        # }
