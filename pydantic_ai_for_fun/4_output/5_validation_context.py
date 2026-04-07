from dataclasses import dataclass

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, ValidationInfo, field_validator
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
VALIDATION CONTEXT — Contexte Pydantic pour la validation

validation_context permet de passer un objet de contexte à Pydantic
pour la validation des outputs structurés ET des arguments de tools.

C'est le contexte Pydantic (info.context dans un @field_validator),
PAS le contexte LLM. Il n'est jamais envoyé au modèle.

Deux formes possibles :
1. Valeur statique → validation_context=10
   L'objet est utilisé tel quel pour toutes les validations.

2. Fonction dynamique → validation_context=lambda ctx: ctx.deps.increment
   La fonction reçoit le RunContext et retourne l'objet de contexte.
   Appelée avant chaque validation → contexte dynamique selon les deps.

Cas d'usage :
- Appliquer un offset/multiplicateur aux valeurs du modèle
- Valider selon des règles business qui dépendent du contexte
- Transformer les données du modèle avant de les retourner
- Limites/contraintes qui varient selon l'utilisateur (deps)
"""


# =====================================================================
# PARTIE 1 : Validation context statique
# =====================================================================

# On passe un entier comme contexte de validation.
# Le field_validator l'utilise pour incrémenter la valeur du modèle.


class Value(BaseModel):
    x: int

    @field_validator("x")
    def increment_value(cls, value: int, info: ValidationInfo):
        # info.context contient le validation_context passé à l'agent
        increment = info.context or 0
        logger.info(f"Validation : {value} + {increment} = {value + increment}")
        return value + increment


agent_static = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Value,
    validation_context=10,  # Contexte statique : toujours 10
)


# =====================================================================
# PARTIE 2 : Validation context dynamique (via deps)
# =====================================================================

# Le contexte de validation est calculé à partir des deps du run.
# Chaque run peut avoir un contexte différent.


@dataclass
class Deps:
    increment: int


agent_dynamic = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Value,
    deps_type=Deps,
    validation_context=lambda ctx: ctx.deps.increment,
)


# =====================================================================
# PARTIE 3 : Validation context avec règles business
# =====================================================================

# Un cas plus réaliste : validation de prix avec une limite
# qui dépend du contexte utilisateur.


@dataclass
class UserConfig:
    max_price: float
    currency: str


class Product(BaseModel):
    name: str
    price: float

    @field_validator("price")
    def validate_price(cls, value: float, info: ValidationInfo):
        if info.context:
            max_price = info.context.get("max_price", float("inf"))
            if value > max_price:
                logger.warning(f"Prix {value} > limite {max_price}, plafonné")
                return max_price
        return value


agent_product = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=Product,
    deps_type=UserConfig,
    validation_context=lambda ctx: {
        "max_price": ctx.deps.max_price,
        "currency": ctx.deps.currency,
    },
    instructions="Tu es un assistant e-commerce. Retourne le produit demandé avec son prix.",
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Contexte statique (+10) ---
    logger.info("=== Validation context statique ===")
    result = agent_static.run_sync("Donne-moi une valeur de 5")
    logger.success(f"Output : {result.output}")
    # Le modèle retourne 5, la validation ajoute 10 → x=15

    # --- Démo 2 : Contexte dynamique via deps ---
    logger.info("=== Validation context dynamique (increment=20) ===")
    result = agent_dynamic.run_sync("Donne-moi une valeur de 5", deps=Deps(increment=20))
    logger.success(f"Output : {result.output}")
    # Le modèle retourne 5, la validation ajoute 20 → x=25

    logger.info("=== Validation context dynamique (increment=0) ===")
    result = agent_dynamic.run_sync("Donne-moi une valeur de 5", deps=Deps(increment=0))
    logger.success(f"Output : {result.output}")
    # Le modèle retourne 5, la validation ajoute 0 → x=5

    # --- Démo 3 : Validation business (prix plafonné) ---
    logger.info("=== Validation business (max_price=50) ===")
    result = agent_product.run_sync(
        "Un iPhone 16",
        deps=UserConfig(max_price=50.0, currency="EUR"),
    )
    logger.success(f"Output : {result.output.name} → {result.output.price}€")
