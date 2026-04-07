import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, BinaryContent, ToolReturn

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
ADVANCED TOOL RETURN — Contrôle fin sur le retour des tools

Quand un tool classique retourne une valeur, celle-ci est envoyée
directement au modèle comme résultat du tool call.

ToolReturn permet d'aller plus loin en séparant :

1. return_value  → la valeur structurée renvoyée au modèle comme
                   résultat du tool (ce qui apparaît dans tool_result).
                   Peut inclure du contenu multimodal.

2. content       → du contenu envoyé comme message utilisateur SÉPARÉ,
                   après le tool_result. Utile pour envoyer du contexte
                   riche (screenshots, images…) en dehors du résultat
                   du tool proprement dit.

3. metadata      → des métadonnées pour VOTRE application, qui ne sont
                   PAS envoyées au LLM. Utile pour le logging, le debug,
                   ou du traitement applicatif. Certains frameworks
                   appellent ça des "artifacts".

Cas d'usage typiques :
- Outils d'automatisation (clic + capture d'écran avant/après)
- Outils qui produisent des données structurées + du contexte visuel
- Quand on veut logger/tracer des infos sans polluer le prompt du LLM
"""


# =====================================================================
# PARTIE 1 : ToolReturn avec contenu riche et métadonnées
# =====================================================================

# Exemple : un outil d'automatisation qui clique à des coordonnées
# et capture des screenshots avant/après. Le modèle reçoit le
# texte de confirmation + les images, tandis que l'application
# récupère les métadonnées structurées.

agent_automation = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant d'automatisation. "
        "Tu peux cliquer sur des éléments à l'écran et observer le résultat. "
        "Décris ce qui s'est passé après chaque action."
    ),
)


@agent_automation.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    """Clique aux coordonnées données et capture des screenshots avant/après."""
    logger.info(f"Clic aux coordonnées ({x}, {y})")

    # Simulation de screenshots (en vrai, on capturerait l'écran)
    before_screenshot = BinaryContent(data=b'\x89PNG...before', media_type='image/png')
    after_screenshot = BinaryContent(data=b'\x89PNG...after', media_type='image/png')

    return ToolReturn(
        # return_value → envoyé au modèle comme résultat du tool call
        return_value=f'Clic effectué avec succès aux coordonnées ({x}, {y})',

        # content → envoyé comme message utilisateur séparé (après le tool result)
        # Utile pour du contenu visuel qui ne fait pas partie du "résultat" à proprement parler
        content=[
            'Avant le clic :',
            before_screenshot,
            'Après le clic :',
            after_screenshot,
        ],

        # metadata → accessible par l'application, JAMAIS envoyé au LLM
        # Parfait pour le logging, le debug, les métriques…
        metadata={
            'coordinates': {'x': x, 'y': y},
            'action_type': 'click_and_capture',
            'screenshots_count': 2,
        },
    )


# =====================================================================
# PARTIE 2 : ToolReturn simple — séparer retour et contexte
# =====================================================================

# Un cas plus simple : un outil de recherche qui retourne le résultat
# structuré au modèle mais envoie aussi du contexte supplémentaire
# comme message utilisateur séparé.

agent_search = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions=(
        "Tu es un assistant de recherche. "
        "Utilise les outils pour trouver des informations et résume les résultats."
    ),
)


@agent_search.tool_plain
def search_database(query: str) -> ToolReturn:
    """Recherche dans la base de données et retourne les résultats."""
    logger.info(f"Recherche : {query}")

    # Simulation de résultats
    results = [
        {"id": 1, "title": "Introduction à Python", "score": 0.95},
        {"id": 2, "title": "Guide Pydantic AI", "score": 0.87},
    ]

    return ToolReturn(
        # Le modèle reçoit les résultats structurés
        return_value=results,

        # Contexte additionnel envoyé comme message utilisateur
        content=[
            f"La recherche '{query}' a trouvé {len(results)} résultats.",
            "Les scores de pertinence sont élevés, les résultats sont fiables.",
        ],

        # Métadonnées pour l'application (pas envoyées au LLM)
        metadata={
            'query': query,
            'total_results': len(results),
            'search_time_ms': 42,
        },
    )


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Clic et capture ---
    logger.info("=== Automatisation : clic sur un bouton ===")
    result = agent_automation.run_sync(
        "Clique sur le bouton 'Valider' qui se trouve en position (150, 300)"
    )
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Recherche avec ToolReturn ---
    logger.info("=== Recherche dans la base ===")
    result = agent_search.run_sync("Trouve-moi des tutoriels sur Python")
    logger.success(f"Réponse : {result.output}")
