import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
TOOL SEARCH — Lazy loading des tools pour les agents avec beaucoup d'outils

Problème : quand un agent a 20+ tools, envoyer toutes les définitions
au LLM à chaque requête :
- Bouffe du contexte inutilement
- Dégrade la qualité de sélection du bon tool

Solution : defer_loading=True masque le tool du contexte initial.
Pydantic AI injecte automatiquement un tool `search_tools` que le LLM
utilise pour chercher un tool par mot-clé quand il en a besoin.

C'est du lazy loading pour les tools :
- Le modèle ne voit que les tools non-deferred + search_tools
- Il appelle search_tools("mortgage") → découvre mortgage_calculator
- Il appelle ensuite mortgage_calculator normalement

Inspiré du "Tool Search Tool" d'Anthropic.
Fonctionne avec n'importe quel modèle (implémenté côté Pydantic AI).

Compromis :
Sans defer_loading : LLM voit le tool → l'appelle → 1 étape.
Avec defer_loading : LLM appelle search_tools → découvre le tool →
l'appelle → 2 étapes. Plus lent, plus de tokens consommés, et le
LLM peut mal formuler sa recherche et rater le bon tool.

Quand l'utiliser :
- Agents avec 20+ tools
- MCP servers exposant des dizaines d'endpoints (.defer_loading())
- En dessous de ~15-20 tools → le coût du lazy loading dépasse le gain

Alternative : si t'as trop de tools, découpe plutôt en plusieurs
agents spécialisés. C'est souvent une meilleure architecture.
"""


# =====================================================================
# PARTIE 1 : Tools avec defer_loading
# =====================================================================

# On simule un agent "banque" avec plein d'outils.
# Les outils spécialisés sont cachés derrière search_tools,
# seul le tool général reste visible directement.

agent_bank = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant bancaire. "
        "Utilise les outils disponibles pour répondre. "
        "Si tu ne trouves pas le bon outil, cherche-le."
    ),
)


# --- Tool toujours visible (pas de defer_loading) ---
@agent_bank.tool_plain
def get_account_balance(account_id: str) -> str:
    """Consulte le solde d'un compte bancaire."""
    logger.info(f"Solde du compte {account_id}")
    return f"Compte {account_id} : 3 450,00 €"


# --- Tools cachés (defer_loading=True) ---
# Le LLM ne les voit pas au départ. Il doit appeler search_tools
# pour les découvrir.

@agent_bank.tool_plain(defer_loading=True)
def mortgage_calculator(principal: float, rate: float, years: int) -> str:
    """Calcule la mensualité d'un prêt immobilier."""
    monthly_rate = rate / 100 / 12
    n_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / (
        (1 + monthly_rate) ** n_payments - 1
    )
    logger.info(f"Prêt : {principal}€, {rate}%, {years} ans → {payment:.2f}€/mois")
    return f"{payment:.2f} €/mois"


@agent_bank.tool_plain(defer_loading=True)
def savings_projection(monthly_amount: float, rate: float, years: int) -> str:
    """Projette l'épargne accumulée avec des versements mensuels."""
    monthly_rate = rate / 100 / 12
    n_months = years * 12
    total = monthly_amount * (((1 + monthly_rate) ** n_months - 1) / monthly_rate)
    logger.info(f"Épargne : {monthly_amount}€/mois, {rate}%, {years} ans → {total:.2f}€")
    return f"Épargne projetée : {total:.2f} €"


@agent_bank.tool_plain(defer_loading=True)
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convertit un montant d'une devise à une autre."""
    # Taux fictifs
    rates = {"EUR_USD": 1.08, "USD_EUR": 0.93, "EUR_GBP": 0.86, "GBP_EUR": 1.16}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    converted = amount * rate
    logger.info(f"Conversion : {amount} {from_currency} → {converted:.2f} {to_currency}")
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


# =====================================================================
# PARTIE 2 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Tool visible directement ---
    logger.info("=== Solde (tool visible) ===")
    result = agent_bank.run_sync("Quel est le solde du compte A-1234 ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Tool caché, le LLM doit le chercher ---
    logger.info("=== Prêt immobilier (tool caché, découvert via search) ===")
    result = agent_bank.run_sync(
        "Calcule la mensualité pour un prêt de 250 000€ à 3.5% sur 25 ans"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Autre tool caché ---
    logger.info("=== Conversion (tool caché) ===")
    result = agent_bank.run_sync("Convertis 1000 euros en dollars")
    logger.success(f"Réponse : {result.output}")
