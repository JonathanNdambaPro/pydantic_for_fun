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
HUMAN IN THE LOOP (DEFERRED TOOLS) — Approbation humaine

Pydantic AI permet de gérer des "deferred tools" (outils différés), utiles quand :
- Un outil nécessite une approbation humaine explicite (ce qui est traité ici).
- Un outil dépend d'un service externe asynchrone (attente de résultat) ou 
  possède une exécution trop longue (ce qui est traité dans le fichier 10_external_loop.py).

Comment ça marche pour l'approbation humaine ?
1. L'agent détecte l'appel à un outil nécessitant une approbation.
2. Le run s'arrête prématurément et retourne un objet `DeferredToolRequests` contenant les requêtes en attente.
3. L'application (ou l'humain via une UI) effectue la validation (approuve ou refuse).
4. On relance l'agent avec l'historique des messages ET le `DeferredToolResults` contenant les décisions d'approbation.

Important :
- Il faut ajouter `DeferredToolRequests` dans l'`output_type` de l'agent (`output_type=[str, DeferredToolRequests]`)
  pour que le typage et le comportement de sortie s'adaptent bien.
"""

# =====================================================================
# Démo : Approbation Humaine pour de la manipulation de fichiers
# =====================================================================

# L'agent peut renvoyer soit une réponse texte (str), soit une requête d'outil différée a valider
agent = Agent(
    'gateway/openai:gpt-4o', # Vous pouvez utiliser 'gateway/anthropic:claude-3-5-sonnet-latest'
    output_type=[str, DeferredToolRequests],
    retries=3,
    instructions="Tu es un assistant de gestion de fichiers."
)

PROTECTED_FILES = {'.env'}

@agent.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    """Met à jour le contenu d'un fichier."""
    # Si le fichier est protégé et que l'appel n'a pas encore été approuvé,
    # on lève ApprovalRequired pour pauser l'agent et demander une approbation humaine/externe.
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        logger.info(f"Fichier protégé détecté ({path}). L'agent va se mettre en pause pour demander approbation.")
        raise ApprovalRequired(metadata={'reason': 'protected'})
    return f"Fichier {path!r} mis à jour : {content!r}"

# requires_approval=True force TOUJOURS l'approbation pour ce tool.
@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """Supprime un fichier."""
    return f"Fichier {path!r} supprimé"


if __name__ == "__main__":
    logger.info("=== Étape 1 : Run initial de l'agent ===")
    prompt = "Supprime '__init__.py', écris 'Hello, world!' dans 'README.md', et vide le fichier '.env'"
    
    # On lance l'agent avec ces instructions. Puisqu'il utilise des deferred tools (delete_file 
    # et update_file sur .env), il va s'interrompre.
    result = agent.run_sync(prompt)
    messages = result.all_messages()
    
    # L'agent a terminé prématurément pour demander des approbations (DeferredToolRequests)
    assert isinstance(result.output, DeferredToolRequests)
    requests = result.output
    
    logger.info("⏸️  L'agent est en pause. Actions en attente d'approbation :")
    for call in requests.approvals:
        logger.warning(f"  - Outil : {call.tool_name} | Args : {call.args}")
        
    # --- On simule ici l'intervention humaine (Human-in-the-loop) ---
    logger.info("=== Étape 2 : L'humain (ou système externe) prend des décisions ===")
    
    results = DeferredToolResults()
    for call in requests.approvals:
        decision = False
        
        if call.tool_name == 'update_file':
            # L'humain approuve systématiquement toutes les modifications
            logger.success(f"✔️  Approbation accordée pour la modification (Args: {call.args})")
            decision = True
            
        elif call.tool_name == 'delete_file':
            # L'humain refuse systématiquement toute suppression
            logger.error(f"❌  Approbation refusée pour la suppression (Args: {call.args})")
            # ToolDenied envoie un message d'erreur/raison au modèle
            decision = ToolDenied('La suppression de fichiers n\'est pas autorisée par l\'administrateur.')
            
        # On affecte notre décision (True, False, ToolApproved() ou ToolDenied()) à cet appel spécifique
        results.approvals[call.tool_call_id] = decision
        
    logger.info("=== Étape 3 : Relance de l'agent avec validation ===")
    
    # Deuxième run : on passe les anciens messages ET les `deferred_tool_results`
    # L'agent comprendra comment les outils se sont "virtuellement" exécutés/refusés, et répondra en fonction.
    final_result = agent.run_sync(
        'Maintenant, crée juste une sauvegarde de README.md',
        message_history=messages,
        deferred_tool_results=results,
    )
    
    logger.success("=== Réponse finale de l'agent ===")
    print(final_result.output)
