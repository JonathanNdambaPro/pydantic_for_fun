from typing import Any

import logfire
from anthropic.lib.tools import BetaAbstractMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, MemoryTool

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
MEMORY TOOL — Mémoire persistante pour l'agent (Anthropic)

MemoryTool permet au LLM de stocker et récupérer des informations
entre les échanges. Le modèle peut créer, lire, modifier et supprimer
des "fichiers mémoire" via des commandes structurées.

C'est un built-in tool Anthropic uniquement. Le SDK Anthropic fournit
une classe abstraite BetaAbstractMemoryTool qu'on sous-classe pour
implémenter le stockage (base de données, fichiers, cloud…).

Commandes disponibles :
- view        → lire le contenu d'un fichier mémoire
- create      → créer un nouveau fichier mémoire
- str_replace → modifier du texte dans un fichier
- insert      → insérer du texte à une ligne donnée
- delete      → supprimer un fichier mémoire
- rename      → renommer un fichier mémoire

Comment ça marche :
1. On crée une sous-classe de BetaAbstractMemoryTool avec notre logique
2. On ajoute MemoryTool() dans builtin_tools
3. On crée un @agent.tool_plain qui forward les commandes vers notre impl
4. Le LLM appelle automatiquement la mémoire quand il en a besoin

En prod, on remplacerait le fake par un vrai stockage :
- LocalFilesystemMemoryTool (fourni par Anthropic comme exemple)
- Base de données (PostgreSQL, Redis…)
- Cloud storage (S3, GCS…)
- Fichiers chiffrés
"""


# =====================================================================
# PARTIE 1 : Implémentation de la mémoire (sous-classe)
# =====================================================================

# On sous-classe BetaAbstractMemoryTool pour définir comment les
# données sont stockées et récupérées. Ici c'est un fake en mémoire.


class InMemoryTool(BetaAbstractMemoryTool):
    """Implémentation simple de la mémoire en dict Python."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        content = self.files.get(command.path, "")
        logger.info(f"[mémoire] Lecture de '{command.path}' → {content[:50]}...")
        return content or f"Fichier '{command.path}' vide ou inexistant."

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        self.files[command.path] = command.content
        logger.info(f"[mémoire] Création de '{command.path}'")
        return f"Fichier créé : {command.path}"

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        if command.path in self.files:
            self.files[command.path] = self.files[command.path].replace(
                command.old_str, command.new_str
            )
            logger.info(f"[mémoire] Modification de '{command.path}'")
            return f"Fichier modifié : {command.path}"
        return f"Fichier non trouvé : {command.path}"

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        logger.info(f"[mémoire] Insertion dans '{command.path}' à la ligne {command.insert_line}")
        return f"Texte inséré dans {command.path}"

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        self.files.pop(command.path, None)
        logger.info(f"[mémoire] Suppression de '{command.path}'")
        return f"Fichier supprimé : {command.path}"

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        if command.old_path in self.files:
            self.files[command.new_path] = self.files.pop(command.old_path)
            logger.info(f"[mémoire] Renommage '{command.old_path}' → '{command.new_path}'")
        return f"Renommé : {command.old_path} → {command.new_path}"

    def clear_all_memory(self) -> str:
        self.files.clear()
        logger.info("[mémoire] Tout effacé")
        return "Mémoire vidée."


# =====================================================================
# PARTIE 2 : Branchement sur l'agent
# =====================================================================

# On crée l'instance de mémoire et on la branche sur l'agent via
# un @agent.tool_plain qui forward les commandes.

memory_storage = InMemoryTool()

agent_memory = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    builtin_tools=[MemoryTool()],
    instructions=(
        "Tu es un assistant avec de la mémoire. "
        "Tu peux retenir des informations sur l'utilisateur. "
        "Réponds en français."
    ),
)


@agent_memory.tool_plain
def memory(**command: Any) -> Any:
    """Forward les commandes mémoire vers notre implémentation."""
    return memory_storage.call(command)


# =====================================================================
# PARTIE 3 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Stocker une info ---
    logger.info("=== Mémoriser une info ===")
    result = agent_memory.run_sync("Retiens que j'habite à Lyon et que j'aime le Python")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 2 : Récupérer l'info ---
    logger.info("=== Rappeler une info ===")
    result = agent_memory.run_sync("Où est-ce que j'habite ?")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 3 : Vérifier le contenu de la mémoire ---
    logger.info(f"=== Contenu mémoire : {memory_storage.files} ===")
