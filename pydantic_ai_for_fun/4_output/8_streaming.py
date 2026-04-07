import asyncio
from datetime import date

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import ValidationError
from typing_extensions import NotRequired, TypedDict
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
STREAMING — Recevoir la réponse au fur et à mesure

Au lieu d'attendre la réponse complète, le streaming permet de
recevoir les tokens au fur et à mesure qu'ils sont générés.

3 types de streaming :

1. stream_text()         → texte brut, accumulé (chaque item = tout le texte jusque-là)
   stream_text(delta=True) → texte brut, par delta (chaque item = nouveau morceau)

2. stream_output()       → sortie structurée, construite progressivement
   Le TypedDict/BaseModel se remplit au fur et à mesure

3. stream_responses()    → accès à la ModelResponse brute
   Pour du contrôle fin sur la validation (validate_response_output)

API :
- agent.run_stream(prompt) → retourne un context manager async
- async with agent.run_stream(...) as result:
      async for chunk in result.stream_text():
          ...

Notes importantes :
- stream_text(delta=True) → le message final n'est PAS ajouté
  à result.all_messages() (voir docs Messages and chat history)
- stream_output() avec output_type structuré utilise Tool Output
  par défaut → nécessite que le modèle supporte le streaming
  d'arguments de tools. Sinon, utiliser NativeOutput ou PromptedOutput.
"""


# =====================================================================
# PARTIE 1 : Streaming texte (accumulé)
# =====================================================================

# Chaque item contient TOUT le texte généré jusque-là.
# "The first" → "The first known" → "The first known use of..."

agent_text = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    instructions="Réponds en français.",
)


async def demo_stream_text():
    async with agent_text.run_stream("D'où vient 'hello world' ?") as result:
        async for message in result.stream_text():
            logger.info(f"Accumulé : {message[:80]}...")


# =====================================================================
# PARTIE 2 : Streaming texte par delta
# =====================================================================

# Chaque item contient UNIQUEMENT le nouveau morceau de texte.
# "The first" → " known" → " use of" → ...


async def demo_stream_delta():
    async with agent_text.run_stream("D'où vient 'hello world' ?") as result:
        async for chunk in result.stream_text(delta=True):
            logger.info(f"Delta : {chunk}")


# =====================================================================
# PARTIE 3 : Streaming de sortie structurée
# =====================================================================

# Le TypedDict se construit progressivement au fur et à mesure
# que le modèle génère les champs.


class UserProfile(TypedDict):
    name: str
    dob: NotRequired[date]
    bio: NotRequired[str]


agent_profile = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=UserProfile,
    instructions="Extrais un profil utilisateur à partir du texte.",
)


async def demo_stream_output():
    user_input = (
        "Je m'appelle Ben, né le 28 janvier 1990, "
        "j'aime le vélo et la montagne."
    )
    async with agent_profile.run_stream(user_input) as result:
        async for profile in result.stream_output():
            logger.info(f"Profil partiel : {profile}")


# =====================================================================
# PARTIE 4 : Streaming avec validation fine (stream_responses)
# =====================================================================

# Pour un contrôle total : on reçoit la ModelResponse brute
# et on valide manuellement avec validate_response_output.


class UserProfileFull(TypedDict, total=False):
    name: str
    dob: date
    bio: str


agent_profile_full = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=UserProfileFull,
    instructions="Extrais un profil utilisateur à partir du texte.",
)


async def demo_stream_responses():
    user_input = (
        "Je m'appelle Ben, né le 28 janvier 1990, "
        "j'aime le vélo et la montagne."
    )
    async with agent_profile_full.run_stream(user_input) as result:
        async for message, last in result.stream_responses(debounce_by=0.01):
            try:
                profile = await result.validate_response_output(
                    message,
                    allow_partial=not last,
                )
            except ValidationError:
                continue
            logger.info(f"Validé ({'final' if last else 'partiel'}) : {profile}")


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================


async def main():
    # --- Démo 1 : Streaming texte accumulé ---
    logger.info("=== Stream texte (accumulé) ===")
    await demo_stream_text()

    # --- Démo 2 : Streaming texte par delta ---
    logger.info("=== Stream texte (delta) ===")
    await demo_stream_delta()

    # --- Démo 3 : Streaming sortie structurée ---
    logger.info("=== Stream output structuré ===")
    await demo_stream_output()

    # --- Démo 4 : Streaming avec validation fine ---
    logger.info("=== Stream responses (validation manuelle) ===")
    await demo_stream_responses()


if __name__ == "__main__":
    asyncio.run(main())
