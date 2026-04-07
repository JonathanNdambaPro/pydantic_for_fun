from datetime import date

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
SYSTEM PROMPTS (Static + Dynamic)

Les system prompts sont PERSISTÉS dans le message_history.
→ Si tu passes l'historique d'un Agent A à un Agent B,
  le system prompt de A sera toujours visible par B.

C'est utile quand tu veux que le contexte "voyage" entre agents.
Mais c'est un piège si les agents ont des personnalités différentes
(les system prompts s'accumulent et peuvent se contredire).

⚠️  Pydantic AI recommande d'utiliser `instructions` par défaut (voir 15_instructions.py).
    N'utilise `system_prompt` que si tu as besoin de cette persistance dans l'historique.
"""


# =====================================================================
# PARTIE 1 : Static + Dynamic system prompts
# =====================================================================

# 1. STATIC SYSTEM PROMPT (Le socle immuable)
# Défini à l'initialisation. Il ne changera jamais pour cet agent.
# Idéal pour définir le "Persona" ou les règles globales.
agent_conseiller = Agent(
    'gateway/openai:gpt-4o',
    deps_type=str,  # On s'attend à recevoir le nom du client (str) en dépendance
    system_prompt="Tu es un conseiller client poli. Utilise toujours le nom du client.",
)


# 2. DYNAMIC SYSTEM PROMPT (Injection de Contexte lié aux Dépendances)
# Évalué UNIQUEMENT au moment du `run()`.
# Permet de lire la base de données ou le contexte (ctx) pour adapter le prompt.
@agent_conseiller.system_prompt
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"Le nom du client actuel est {ctx.deps}."


# 3. DYNAMIC SYSTEM PROMPT (Injection de Contexte Temporel/Système)
# Le paramètre ctx est optionnel si on n'a pas besoin des dépendances.
# Essentiel pour que le LLM ait la notion du temps (il n'a pas d'horloge interne).
@agent_conseiller.system_prompt
def add_the_date() -> str:
    return f"La date d'aujourd'hui est {date.today()}."


# --- EXÉCUTION ---
# Pydantic AI concatène les 3 blocs (1 statique + 2 dynamiques)
# pour former le System Prompt final envoyé à l'API :
#   "Tu es un conseiller client poli. Utilise toujours le nom du client.
#    Le nom du client actuel est Thomas.
#    La date d'aujourd'hui est 2026-04-02."
result = agent_conseiller.run_sync('Quelle est la date du jour ?', deps='Thomas')
logger.info(f"Conseiller : {result.output}")


# =====================================================================
# PARTIE 2 : Le piège — system prompts persistés dans l'historique
# =====================================================================

# On crée un 2ème agent avec une personnalité DIFFÉRENTE
agent_blagueur = Agent(
    'gateway/openai:gpt-4o',
    system_prompt="Tu es un comique. Réponds toujours avec une blague.",
)

# On passe l'historique du conseiller au blagueur.
# PROBLÈME : le system prompt "Tu es un conseiller client poli..."
# est toujours dans les messages → le blagueur voit les DEUX personnalités !
result_blagueur = agent_blagueur.run_sync(
    'Raconte-moi une blague',
    message_history=result.all_messages(),
)
logger.info(f"Blagueur (avec historique du conseiller) : {result_blagueur.output}")

# Le LLM reçoit :
#   [system] "Tu es un conseiller client poli..."   ← RÉSIDU du 1er agent !
#   [system] "Tu es un comique..."                   ← le 2ème agent
#   [user] "Raconte-moi une blague"
#
# → Les deux system prompts cohabitent, ce qui peut créer de la confusion.
# → C'est pour ça que `instructions` existe (voir 15_instructions.py).
