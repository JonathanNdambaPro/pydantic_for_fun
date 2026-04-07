import re

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, TextOutput

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
OUTPUT FUNCTIONS — La sortie est le résultat d'une fonction

Au lieu de retourner du texte ou un BaseModel directement, on peut
faire en sorte que la sortie de l'agent soit le RÉSULTAT d'une
fonction appelée avec les arguments fournis par le modèle.

Output functions vs Tools :
- Tool   → le modèle appelle la fonction, reçoit le résultat, continue
- Output → le modèle appelle la fonction, le run S'ARRÊTE, le résultat
           est la sortie finale (pas renvoyé au modèle)

Cas d'usage :
- Post-traitement / validation des données du modèle
- Hand-off vers un autre agent
- Transformation avant retour (ex: SQL → résultat)

Fonctionnalités :
- Arguments validés par Pydantic
- Peut prendre RunContext en premier argument
- Peut lever ModelRetry pour demander au modèle de réessayer
- Peut être mixé avec des BaseModel dans output_type=[fn, Model]
- Ne PAS enregistrer aussi comme @agent.tool (confusion pour le modèle)

TextOutput :
- Wraps une fonction qui prend un str
- Le modèle répond en texte brut (pas via tool call)
- La fonction transforme le texte avant de le retourner

partial_output (streaming) :
- En streaming, la fonction est appelée plusieurs fois (partiel + final)
- ctx.partial_output == True → sortie partielle (pas de side effects)
- ctx.partial_output == False → sortie finale (OK pour side effects)
"""


# =====================================================================
# PARTIE 1 : Output function basique — SQL query runner
# =====================================================================

# Le modèle génère une requête SQL, la fonction l'exécute sur une
# "base de données" simulée. Si la requête est invalide, ModelRetry
# guide le modèle à corriger.


class Row(BaseModel):
    name: str
    country: str


# Base de données simulée
tables = {
    "capital_cities": [
        Row(name="Amsterdam", country="Netherlands"),
        Row(name="Paris", country="France"),
        Row(name="Tokyo", country="Japan"),
    ]
}


class SQLFailure(BaseModel):
    """Échec irrécupérable. À utiliser quand la requête ne peut pas être corrigée."""

    explanation: str


def run_sql_query(query: str) -> list[Row]:
    """Exécute une requête SQL sur la base de données."""
    logger.info(f"SQL reçu : {query}")

    select_table = re.match(r"SELECT (.+) FROM (\w+)", query)
    if select_table:
        column_names = select_table.group(1)
        if column_names != "*":
            raise ModelRetry(
                "Seul 'SELECT *' est supporté, le filtrage de colonnes doit être fait manuellement."
            )

        table_name = select_table.group(2)
        if table_name not in tables:
            raise ModelRetry(
                f"Table inconnue '{table_name}'. Tables disponibles : {', '.join(tables.keys())}."
            )

        logger.success(f"Requête OK → {len(tables[table_name])} lignes")
        return tables[table_name]

    raise ModelRetry(f"Requête non supportée : '{query}'.")


agent_sql = Agent[None, list[Row] | SQLFailure](
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[run_sql_query, SQLFailure],
    instructions=(
        "Tu es un agent SQL. Génère des requêtes SQL pour répondre "
        "aux questions. Si tu ne peux pas, retourne un SQLFailure."
    ),
)


# =====================================================================
# PARTIE 2 : TextOutput — transformer le texte brut
# =====================================================================

# Le modèle répond en texte brut, la fonction transforme le texte
# avant de le retourner. Pas de tool call, juste du texte.


def split_into_words(text: str) -> list[str]:
    """Découpe la réponse en mots."""
    return text.split()


agent_text_output = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=TextOutput(split_into_words),
    instructions="Réponds en une phrase courte.",
)


# =====================================================================
# PARTIE 3 : Output function avec RunContext
# =====================================================================

# La fonction peut accéder au contexte (deps, messages, etc.)
# et vérifier partial_output pour le streaming.


class ProcessedData(BaseModel):
    name: str
    value: int


def save_record(ctx: RunContext, record: ProcessedData) -> ProcessedData:
    """Sauvegarde un enregistrement. Side effects uniquement sur la sortie finale."""
    if ctx.partial_output:
        # En streaming : sortie partielle, pas de side effects
        return record

    # Sortie finale → on peut faire des side effects
    logger.info(f"Sauvegarde : {record.name} = {record.value}")
    return record


agent_save = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=save_record,
    instructions="Crée un enregistrement avec le nom et la valeur demandés.",
)


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : SQL output function ---
    logger.info("=== SQL : requête valide ===")
    result = agent_sql.run_sync("Donne-moi toutes les capitales")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    logger.info("=== SQL : table inexistante ===")
    result = agent_sql.run_sync("Donne-moi tous les animaux de compagnie")
    logger.success(f"Output ({type(result.output).__name__}) : {result.output}")

    # --- Démo 2 : TextOutput ---
    logger.info("=== TextOutput : découpage en mots ===")
    result = agent_text_output.run_sync("Qui était Albert Einstein ?")
    logger.success(f"Output : {result.output}")

    # --- Démo 3 : Output function avec RunContext ---
    logger.info("=== Save record ===")
    result = agent_save.run_sync("Crée un enregistrement 'test' avec la valeur 42")
    logger.success(f"Output : {result.output}")
