import logfire
from dotenv import load_dotenv
from loguru import logger

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MODEL ERRORS & DIAGNOSTIC AVEC capture_run_messages

Quand un agent plante (retry épuisés, API 503, réponse incohérente...),
il lève `UnexpectedModelBehavior`. Le problème : sans contexte,
tu ne sais pas POURQUOI il a planté.

`capture_run_messages` capture TOUS les messages échangés pendant le run :
- Les requêtes envoyées au LLM
- Les réponses reçues
- Les appels de tools (avec les args)
- Les messages de retry

C'est l'équivalent d'un "tcpdump" pour ton agent :
tu vois exactement ce qui s'est passé avant le crash.
"""


# =====================================================================
# PARTIE 1 : Un tool qui refuse tout sauf une réponse spécifique
# =====================================================================

agent = Agent(
    'gateway/openai:gpt-4o',
    instructions="Tu DOIS toujours utiliser le tool calc_volume pour calculer un volume. Ne calcule JAMAIS toi-même.",
)


@agent.tool_plain(retries=1)  # 1 seul retry → crash rapide pour la démo
def calc_volume(size: int) -> int:
    """Calcule le volume d'un cube.

    Args:
        size: La taille du côté du cube.
    """
    if size == 42:
        logger.success(f"Calcul OK : {size}³ = {size ** 3}")
        return size ** 3

    # Le LLM va envoyer 6 (comme demandé par l'utilisateur)
    # → On refuse et on lui dit de réessayer
    logger.warning(f"Taille {size} refusée → retry")
    raise ModelRetry('Mauvaise taille, réessaie.')


# =====================================================================
# PARTIE 2 : capture_run_messages — la boîte noire de l'agent
# =====================================================================

logger.info("Test avec capture_run_messages...")

# Le context manager capture tous les messages échangés pendant le run
with capture_run_messages() as messages:
    try:
        # Le LLM va appeler calc_volume(size=6)
        # → ModelRetry → le LLM réessaie → encore size=6 → max retries atteint → CRASH
        result = agent.run_sync('Calcule le volume d\'un cube de côté 6.')

    except UnexpectedModelBehavior as e:
        # L'agent a planté — analysons la boîte noire
        logger.error(f"Agent planté : {e}")
        logger.error(f"Cause racine : {e.__cause__!r}")

        # Voilà la valeur de capture_run_messages :
        # on peut voir EXACTEMENT ce qui s'est passé
        logger.info(f"Nombre de messages échangés : {len(messages)}")
        logger.info(f"Contenu des messages : {messages}")

        for i, msg in enumerate(messages):
            logger.debug(f"Message {i} : {msg.__class__.__name__}")
            for part in msg.parts:
                logger.debug(f"  └─ {part.__class__.__name__} : {part}")

    else:
        # Si par miracle ça marche (spoiler : ça ne marchera pas)
        logger.success(f"Résultat : {result.output}")


# =====================================================================
# PARTIE 3 : Le cas qui marche (pour comparaison)
# =====================================================================

logger.info("Test avec la bonne taille (42)...")

with capture_run_messages() as messages_ok:
    result_ok = agent.run_sync('Calcule le volume d\'un cube de côté 42.')
    logger.success(f"Résultat : {result_ok.output}")
    logger.info(f"Messages échangés : {len(messages_ok)}")
