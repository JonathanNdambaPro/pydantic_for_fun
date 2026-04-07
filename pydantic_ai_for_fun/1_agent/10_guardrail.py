
import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModelSettings

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

# Agent configuré pour la production (gestion de la charge)


"""
INPUT/OUTPUT GUARDRAILS NATIVS (SÉCURITÉ & COMPLIANCE)

Objectif : Bloquer les prompts toxiques (jailbreak, insultes) ou les générations 
problématiques directement à la source, via l'API du fournisseur.

* regarder Guardrails AI également
"""

# Utilisation d'un modèle via le Gateway Pydantic (ici Gemini 3 Flash)
agent = Agent('gateway/gemini:gemini-3-flash-preview')

try:
    # On soumet un prompt conçu pour déclencher les filtres de sécurité
    result = agent.run_sync(
        'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',

        # GoogleModelSettings permet d'injecter des paramètres propriétaires (Google) 
        # tout en conservant les paramètres génériques (temperature)
        model_settings=GoogleModelSettings(
            temperature=0.0, 

            # Paramètres spécifiques à Google : Configuration des seuils de tolérance
            # BLOCK_LOW_AND_ABOVE = Tolérance zéro.
            gemini_safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
            ],
        ),
    )
except UnexpectedModelBehavior as e:
    # GESTION D'ERREUR CRITIQUE
    # Si le filtre Google s'active, il coupe la génération. 
    # Pydantic AI lève cette exception spécifique pour t'éviter un crash inattendu.
    logger.error(f"Modération — payload rejeté par les filtres de sécurité : {e}")