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
INSTRUCTIONS (la version recommandée des system prompts)

La différence clé avec system_prompt :
- system_prompt  → PERSISTÉ dans le message_history (voyage entre agents)
- instructions   → JAMAIS persisté. À chaque run, seules les instructions
                    de l'agent ACTUEL sont injectées. L'historique est "propre".

Quand utiliser quoi :
- instructions   → par défaut, toujours (recommandé par Pydantic AI)
- system_prompt  → uniquement si tu veux que le contexte persiste entre agents

Les instructions supportent les mêmes 3 formes que system_prompt :
1. Static   → paramètre `instructions=` du constructeur
2. Dynamic  → décorateur `@agent.instructions`
3. Runtime  → paramètre `instructions=` de `run()`
"""


# =====================================================================
# PARTIE 1 : Static + Dynamic instructions (même API que system_prompt)
# =====================================================================

agent_conseiller = Agent(
    'gateway/openai:gpt-4o',
    deps_type=str,
    # 1. STATIC INSTRUCTION — le socle, identique à chaque run
    instructions="Tu es un conseiller client poli. Utilise toujours le nom du client.",
)


# 2. DYNAMIC INSTRUCTION — injecte le contexte à chaque run
@agent_conseiller.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"Le nom du client actuel est {ctx.deps}."


@agent_conseiller.instructions
def add_the_date() -> str:
    return f"La date d'aujourd'hui est {date.today()}."


result = agent_conseiller.run_sync('Quelle est la date du jour ?', deps='Thomas')
logger.info(f"Conseiller : {result.output}")


# =====================================================================
# PARTIE 2 : La différence — l'historique est PROPRE
# =====================================================================

agent_blagueur = Agent(
    'gateway/openai:gpt-4o',
    instructions="Tu es un comique. Réponds toujours avec une blague.",
)

# On passe l'historique du conseiller au blagueur.
# Cette fois, le system prompt "Tu es un conseiller client poli..."
# n'est PAS dans les messages → le blagueur ne voit QUE ses propres instructions.
result_blagueur = agent_blagueur.run_sync(
    'Raconte-moi une blague',
    message_history=result.all_messages(),
)
logger.info(f"Blagueur (historique propre) : {result_blagueur.output}")

# Le LLM reçoit :
#   [system] "Tu es un comique..."   ← SEUL le 2ème agent, pas de résidu !
#   [user] "Quelle est la date du jour ?"
#   [assistant] "Bonjour Thomas, nous sommes le..."
#   [user] "Raconte-moi une blague"
#
# → Pas de confusion entre personnalités. C'est propre.


# =====================================================================
# PARTIE 3 : Runtime instructions (instructions ponctuelles par run)
# =====================================================================

# On peut aussi ajouter des instructions spécifiques à UN SEUL run
# sans toucher à la config de l'agent.
result_formel = agent_conseiller.run_sync(
    'Comment vas-tu ?',
    deps='Marie',
    instructions='Réponds en langage soutenu, vouvoie le client.',
)
logger.info(f"Conseiller (formel) : {result_formel.output}")

# Les runtime instructions s'AJOUTENT aux static + dynamic instructions.
# Utile pour adapter le comportement à la volée (ex: ton, langue, format).
