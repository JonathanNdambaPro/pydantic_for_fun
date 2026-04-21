import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
REFLECTION & SELF-CORRECTION (ModelRetry)

Quand un tool échoue, au lieu de crasher, on peut renvoyer l'erreur
au LLM pour qu'il se corrige lui-même et réessaie.

Flux :
1. Le LLM appelle un tool avec des mauvais paramètres
2. Le tool lève `ModelRetry("explication de l'erreur")`
3. Pydantic AI renvoie ce message au LLM comme feedback
4. Le LLM comprend son erreur et réessaie avec de meilleurs paramètres
5. Répété jusqu'à `retries` max tentatives

Ce n'est PAS un simple try/except — c'est une boucle de correction
où le LLM apprend de ses erreurs en temps réel.
"""


# =====================================================================
# Simulation d'une base de données de produits
# =====================================================================

CATALOGUE = {
    "MacBook Pro 14 M4": {"prix": 2399, "stock": 5},
    "iPhone 16 Pro": {"prix": 1199, "stock": 12},
    "AirPods Pro 3": {"prix": 279, "stock": 0},
    "iPad Air M3": {"prix": 699, "stock": 3},
}


class InfoProduit(BaseModel):
    nom: str
    prix: int
    stock: int
    disponible: bool


agent = Agent(
    "gateway/openai:gpt-4o",
    instructions=(
        "Tu es un vendeur Apple. Aide le client à trouver des produits. "
        "Utilise TOUJOURS les tools pour chercher ou commander. "
        "Ne demande JAMAIS de précision au client, essaie directement avec le nom le plus probable."
    ),
    output_type=str,
)


# =====================================================================
# Tool avec retry : le LLM doit donner le nom EXACT du produit
# =====================================================================


@agent.tool(retries=3)  # Max 3 tentatives avant de crasher
async def chercher_produit(ctx: RunContext, nom_produit: str) -> str:
    """Cherche un produit dans le catalogue Apple.

    Args:
        nom_produit: Le nom EXACT du produit (ex: "MacBook Pro 14 M4", pas juste "MacBook")
    """
    logger.info(f"Tentative {ctx.retry + 1} : recherche de '{nom_produit}'")

    # Recherche exacte
    produit = CATALOGUE.get(nom_produit) 
    if produit is not None:
        dispo = "en stock" if produit["stock"] > 0 else "rupture de stock"
        logger.success(f"Produit trouvé : {nom_produit} — {produit['prix']}€ — {dispo}")
        return f"{nom_produit} — {produit['prix']}€ — {dispo} ({produit['stock']} unités)"

    # Pas trouvé → on donne un indice au LLM pour qu'il se corrige
    noms_disponibles = ", ".join(CATALOGUE.keys())
    logger.warning(f"Produit '{nom_produit}' introuvable → retry")
    raise ModelRetry(
        f"Produit '{nom_produit}' introuvable. "
        f"Les produits disponibles sont : {noms_disponibles}. "
        f"Réessaie avec le nom exact."
    )

agent.output_validator

# =====================================================================
# Tool avec retry : validation métier
# =====================================================================


@agent.tool(retries=2)
async def commander_produit(ctx: RunContext, nom_produit: str, quantite: int) -> str:
    """Passe une commande pour un produit.

    Args:
        nom_produit: Le nom EXACT du produit
        quantite: Le nombre d'unités à commander (entre 1 et 10)
    """
    logger.info(f"Commande : {quantite}x '{nom_produit}'")

    produit = CATALOGUE.get(nom_produit)
    if produit is None:
        noms = ", ".join(CATALOGUE.keys())
        logger.warning(f"Produit '{nom_produit}' introuvable → retry")
        raise ModelRetry(f"Produit introuvable. Produits valides : {noms}")

    if produit["stock"] == 0:
        logger.warning(f"'{nom_produit}' en rupture de stock → retry")
        raise ModelRetry(f"'{nom_produit}' est en rupture de stock. Propose un produit alternatif au client.")

    if quantite > produit["stock"]:
        logger.warning(f"Stock insuffisant pour '{nom_produit}' : {produit['stock']} dispo, {quantite} demandé → retry")
        raise ModelRetry(
            f"Stock insuffisant pour '{nom_produit}' : "
            f"seulement {produit['stock']} unités disponibles, "
            f"tu en as demandé {quantite}. Réduis la quantité."
        )

    if quantite < 1 or quantite > 10:
        logger.error(f"Quantité invalide : {quantite}")
        raise ModelRetry("La quantité doit être entre 1 et 10.")

    total = produit["prix"] * quantite
    logger.success(f"Commande confirmée : {quantite}x {nom_produit} = {total}€")
    return f"Commande confirmée : {quantite}x {nom_produit} = {total}€"


# =====================================================================
# EXÉCUTION — Le LLM va probablement se tromper puis se corriger
# =====================================================================

logger.info("TEST 1 : Recherche avec un nom approximatif")
# Le LLM va sûrement envoyer "MacBook" ou "MacBook Pro"
# → ModelRetry lui dira le nom exact → il réessaie
result = agent.run_sync("T'as des MacBook en stock ?")
logger.success(f"Réponse finale : {result.output}")

logger.info("TEST 2 : Commande d'un produit en rupture de stock")
# Les AirPods Pro 3 sont en rupture → ModelRetry lui dit de proposer une alternative
result2 = agent.run_sync("Je veux commander 2 AirPods Pro 3")
logger.success(f"Réponse finale : {result2.output}")
