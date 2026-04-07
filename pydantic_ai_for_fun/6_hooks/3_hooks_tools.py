from typing import Any

import logfire
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

"""
HOOKS TOOLS — Intercepter validation et exécution des outils

Les tool hooks permettent d'intercepter les outils à deux niveaux :
1. Validation  → quand les arguments JSON du modèle sont parsés/validés
2. Exécution   → quand la fonction de l'outil tourne

Chaque niveau a ses hooks before/after/wrap/error.
Tous reçoivent call (ToolCallPart) et tool_def (ToolDefinition).

Filtrage par nom :
- Le paramètre tools=['nom'] cible des outils spécifiques
- Les autres outils passent sans être affectés

Skip :
- SkipToolValidation(args) → sauter la validation
- SkipToolExecution(result) → sauter l'exécution
"""


# =====================================================================
# PARTIE 1 : Tool execution hooks — before/after
# =====================================================================

# before_tool_execute reçoit les args validés (dict[str, Any]).
# after_tool_execute reçoit le résultat de l'exécution.

hooks_exec = Hooks()
exec_log: list[str] = []


@hooks_exec.on.before_tool_execute
async def log_before_exec(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Logge avant chaque exécution d'outil."""
    exec_log.append(f"before: {call.tool_name}")
    logger.info(f"[before_exec] {call.tool_name} avec args={args}")
    return args


@hooks_exec.on.after_tool_execute
async def log_after_exec(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    result: Any,
) -> Any:
    """Logge après chaque exécution d'outil."""
    exec_log.append(f"after: {call.tool_name}")
    logger.info(f"[after_exec] {call.tool_name} → {result}")
    return result


agent_exec = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_exec],
    instructions="Tu es un assistant avec des outils. Utilise-les quand c'est pertinent.",
)


@agent_exec.tool_plain
def get_weather(city: str) -> str:
    """Retourne la météo d'une ville."""
    return f"Il fait 22°C à {city}"


@agent_exec.tool_plain
def get_time(timezone: str) -> str:
    """Retourne l'heure dans un fuseau horaire."""
    return f"Il est 14:30 à {timezone}"


# =====================================================================
# PARTIE 2 : Tool hooks avec filtre par nom (tools=[...])
# =====================================================================

# On peut cibler un hook sur des outils spécifiques.
# Seuls les outils nommés déclenchent le hook.

hooks_filtered = Hooks()
audit_log: list[str] = []


@hooks_filtered.on.before_tool_execute(tools=["send_email"])
async def audit_email_only(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Audite uniquement les appels à send_email."""
    audit_log.append(f"AUDIT: {call.tool_name}({args})")
    logger.warning(f"[audit] {call.tool_name} appelé avec {args}")
    return args


agent_filtered = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_filtered],
    instructions="Tu es un assistant email.",
)


@agent_filtered.tool_plain
def send_email(to: str, subject: str) -> str:
    """Envoie un email."""
    return f"Email envoyé à {to} : {subject}"


@agent_filtered.tool_plain
def read_inbox() -> str:
    """Lit la boîte de réception."""
    return "3 emails non lus."


# =====================================================================
# PARTIE 3 : Tool validation hooks
# =====================================================================

# Les hooks de validation se déclenchent quand les arguments JSON
# du modèle sont parsés et validés par Pydantic.

hooks_validate = Hooks()


@hooks_validate.on.before_tool_validate
async def log_raw_args(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> None:
    """Logge les arguments bruts avant validation Pydantic."""
    logger.info(f"[validate] Arguments bruts pour {call.tool_name}: {call.args}")


@hooks_validate.on.after_tool_validate
async def log_validated_args(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Logge les arguments après validation Pydantic."""
    logger.info(f"[validate] Arguments validés pour {call.tool_name}: {args}")
    return args


agent_validate = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_validate],
    instructions="Tu es un assistant.",
)


@agent_validate.tool_plain
def calculate(a: int, b: int, operation: str) -> str:
    """Effectue un calcul entre deux nombres."""
    if operation == "add":
        return str(a + b)
    elif operation == "multiply":
        return str(a * b)
    return f"Opération inconnue: {operation}"


# =====================================================================
# PARTIE 4 : prepare_tools — Filtrer les outils visibles
# =====================================================================

# prepare_tools modifie les définitions d'outils que le modèle voit.
# Contrôle la visibilité, pas l'exécution.

hooks_prepare = Hooks()


@hooks_prepare.on.prepare_tools
async def hide_dangerous_tools(
    ctx: RunContext[None],
    tool_defs: list[ToolDefinition],
) -> list[ToolDefinition]:
    """Cache les outils dangereux selon le contexte."""
    # Exemple : on cache 'delete_file' si on est en mode lecture seule
    safe_tools = [t for t in tool_defs if not t.name.startswith("delete_")]
    hidden = len(tool_defs) - len(safe_tools)
    if hidden:
        logger.warning(f"[prepare] {hidden} outil(s) caché(s) du modèle")
    return safe_tools


agent_prepare = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    capabilities=[hooks_prepare],
    instructions="Tu es un assistant fichiers.",
)


@agent_prepare.tool_plain
def list_files() -> str:
    """Liste les fichiers."""
    return "fichier1.txt, fichier2.txt"


@agent_prepare.tool_plain
def delete_file(name: str) -> str:
    """Supprime un fichier (dangereux !)."""
    return f"{name} supprimé"


# =====================================================================
# PARTIE 5 : Exécution
# =====================================================================

if __name__ == "__main__":
    # --- Démo 1 : Tool execution hooks ---
    logger.info("=== Tool execution hooks (before/after) ===")
    result = agent_exec.run_sync("Quelle est la météo à Paris ?")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Exec log : {exec_log}")

    # --- Démo 2 : Tool hook filtré par nom ---
    logger.info("=== Tool hook filtré (audit send_email) ===")
    result = agent_filtered.run_sync("Envoie un email à alice@test.com sujet 'Hello'")
    logger.success(f"Réponse : {result.output}")
    logger.info(f"Audit log : {audit_log}")

    # --- Démo 3 : Tool validation hooks ---
    logger.info("=== Tool validation hooks ===")
    result = agent_validate.run_sync("Calcule 3 + 7")
    logger.success(f"Réponse : {result.output}")

    # --- Démo 4 : prepare_tools (filtrage visibilité) ---
    logger.info("=== prepare_tools (cacher delete_file) ===")
    result = agent_prepare.run_sync("Liste les fichiers disponibles")
    logger.success(f"Réponse : {result.output}")
