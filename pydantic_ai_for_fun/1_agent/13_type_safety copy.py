import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])


"""
LE TYPAGE FORT AVEC BASEMODEL & FIELD

Au lieu de faire un long prompt système pour expliquer au LLM comment formater 
sa réponse, on laisse le schéma Pydantic faire le travail.
"""

# 1. On définit notre schéma strict avec des descriptions sémantiques
class UserProfile(BaseModel):
    first_name: str = Field(
        description="Le prénom de l'utilisateur, toujours avec la première lettre en majuscule."
    )
    age: int = Field(
        description="L'âge extrait du texte. Si on a qu'une date de naissance, calcule l'âge en sachant qu'on est en 2026."
    )
    risk_level: str = Field(
        description="Niveau de risque évalué selon le contexte. Doit être STRICTEMENT 'low', 'medium' ou 'high'."
    )

# 2. On passe le modèle à l'Agent via output_type
agent = Agent(
    'gateway/openai:gpt-4o',
    output_type=UserProfile,
    system_prompt="Tu es un extracteur de données implacable. Lis le texte et remplis le profil."
)

# 3. Exécution
result = agent.run_sync(
    "Salut, c'est thomas, je suis né en 1996 et je fais beaucoup de moto sur circuit sans assurance."
)

# --- LA MAGIE DU TYPAGE (IDE & MyPy) ---

# result.data est GARANTI d'être une instance de UserProfile.
# Si tu tapes "result.data.", ton IDE va te proposer "first_name", "age", etc.
profil: UserProfile = result.output

logger.info(f"Prénom : {profil.first_name}")   # Sortie garantie (str) : "Thomas" (maj respectée)
logger.info(f"Âge : {profil.age}")             # Sortie garantie (int) : 30 (calculé car on est en 2026)
logger.info(f"Risque : {profil.risk_level}")
