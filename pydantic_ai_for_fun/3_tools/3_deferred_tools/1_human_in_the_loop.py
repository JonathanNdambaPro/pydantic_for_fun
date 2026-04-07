import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HUMAN-IN-THE-LOOP TOOL APPROVAL — Approbation humaine avant exécution

Certains tools sont sensibles (suppression de fichiers, envoi d'emails,
paiements…). On veut qu'un humain valide AVANT que le tool s'exécute.

Le flow en 2 runs :
  Run 1 : prompt → LLM appelle le tool → tool deferred
          → agent retourne DeferredToolRequests.approvals

          ... l'humain approuve ou refuse (UI, Slack, formulaire…) ...

  Run 2 : on relance avec message_history + DeferredToolResults
          → l'agent reprend et exécute (ou pas) selon l'approbation

Deux façons de demander une approbation :
- requires_approval=True → le tool est TOUJOURS deferred
- ApprovalRequired()     → deferred CONDITIONNELLEMENT (selon le contexte)
  → ctx.tool_call_approved est True au run 2 si approuvé

Important :
- DeferredToolRequests doit être dans output_type de l'agent
- On peut le mettre au niveau de l'agent OU au niveau du run

Résultats possibles dans DeferredToolResults.approvals :
- True                           → approuvé, exécuter le tool
- ToolApproved(override_args=..) → approuvé avec args modifiés
- False                          → refusé (message par défaut)
- ToolDenied("raison")           → refusé avec message custom

Voir aussi : 2_external_tool_execution.py pour les tools exécutés
en dehors du process Python (frontend, worker, API externe…).
"""


# =====================================================================
# PARTIE 1 : Tool avec requires_approval=True (toujours deferred)
# =====================================================================

# delete_file demande TOUJOURS une approbation avant exécution.
# Le code dans la fonction peut supposer que l'approbation a été donnée.

PROTECTED_FILES = {".env", ".secrets"}

agent_files = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    output_type=[str, DeferredToolRequests],
    instructions=(
        "Tu es un gestionnaire de fichiers. "
        "Utilise les outils pour modifier et supprimer des fichiers."
    ),
)


@agent_files.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """Supprime un fichier. Nécessite toujours une approbation."""
    logger.warning(f"Fichier supprimé : {path}")
    return f"Fichier '{path}' supprimé avec succès."


# =====================================================================
# PARTIE 2 : Tool avec ApprovalRequired conditionnelle
# =====================================================================

# update_file s'exécute normalement SAUF pour les fichiers protégés.
# Pour ceux-là, on lève ApprovalRequired → le tool est deferred.


@agent_files.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    """Met à jour un fichier. Approbation requise pour les fichiers protégés."""
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        logger.info(f"Fichier protégé '{path}' → approbation requise")
        raise ApprovalRequired(metadata={"reason": "protected"})

    logger.info(f"Fichier '{path}' mis à jour")
    return f"Fichier '{path}' mis à jour avec : '{content}'"


# =====================================================================
# PARTIE 3 : Gérer les approbations (DeferredToolResults)
# =====================================================================


def handle_approvals(requests: DeferredToolRequests) -> DeferredToolResults:
    """Simule un processus d'approbation humaine.

    En prod, ça serait une UI, un formulaire, un Slack bot, etc.
    """
    results = DeferredToolResults()

    for call in requests.approvals:
        logger.info(f"Tool en attente : {call.tool_name}({call.args})")

        if call.tool_name == "update_file":
            # On approuve les mises à jour de fichiers protégés
            results.approvals[call.tool_call_id] = True
            logger.success(f"  → Approuvé : {call.tool_name}")

        elif call.tool_name == "delete_file":
            # On refuse les suppressions
            results.approvals[call.tool_call_id] = ToolDenied(
                "La suppression de fichiers n'est pas autorisée."
            )
            logger.warning(f"  → Refusé : {call.tool_name}")

    return results


# =====================================================================
# PARTIE 4 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Run 1 : l'agent propose des actions ---
    logger.info("=== Run 1 : demande d'actions sur les fichiers ===")
    result = agent_files.run_sync(
        "Supprime '__init__.py', écris 'Hello' dans 'README.md', "
        "et vide le fichier '.env'"
    )
    messages = result.all_messages()

    if isinstance(result.output, DeferredToolRequests):
        logger.info(f"Tools en attente d'approbation : {len(result.output.approvals)}")

        # --- Processus d'approbation ---
        deferred_results = handle_approvals(result.output)

        # --- Run 2 : on relance avec les résultats ---
        logger.info("=== Run 2 : reprise après approbation ===")
        result = agent_files.run_sync(
            message_history=messages,
            deferred_tool_results=deferred_results,
        )
        logger.success(f"Réponse finale : {result.output}")
    else:
        # Pas de tools deferred → réponse directe
        logger.success(f"Réponse : {result.output}")
