import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Literal

import logfire
from dotenv import load_dotenv
from loguru import logger
from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])


# ==============================================================================
# 1. WRAPPER PLAYWRIGHT (bas niveau)
# ==============================================================================

KEY_MAP: dict[str, str] = {
    "enter": "Enter", "return": "Enter", "tab": "Tab",
    "escape": "Escape", "esc": "Escape", "backspace": "Backspace",
    "delete": "Delete", "space": "Space", "up": "ArrowUp",
    "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
}


async def resolve_element(page: Page, coordinate: list[int] | None, selector: str | None) -> tuple[float, float]:
    """Résout les coordonnées d'un élément via coordonnées directes ou sélecteur CSS."""
    if coordinate:
        return float(coordinate[0]), float(coordinate[1])
    if selector:
        el = await page.query_selector(selector)
        if not el:
            raise ModelRetry(f"Élément introuvable avec '{selector}'. Utilise 'read_page' pour inspecter le DOM.")
        box = await el.bounding_box()
        if not box:
            raise ModelRetry(f"Élément '{selector}' trouvé mais caché (pas de bounding box).")
        return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    raise ModelRetry("Tu dois fournir 'coordinate' ou 'selector'.")


# ==============================================================================
# 2. MODÈLES DE SORTIE
# ==============================================================================

class TaskResult(BaseModel):
    """Résultat final structuré retourné par l'agent."""
    success: bool = Field(description="True si la tâche a été accomplie avec succès")
    summary: str = Field(description="Résumé concis de ce qui a été fait")
    actions_taken: list[str] = Field(description="Liste des actions principales effectuées")
    final_url: str = Field(description="URL de la page à la fin de la tâche")


# ==============================================================================
# 3. AGENT ET DÉPENDANCES
# ==============================================================================

@dataclass
class BrowserDeps:
    page: Page


agent = Agent(
    'gateway/anthropic:claude-sonnet-4-6',
    deps_type=BrowserDeps,
    output_type=TaskResult,
    retries=5,
)


@agent.system_prompt
async def dynamic_browser_context(ctx: RunContext[BrowserDeps]) -> str:
    page = ctx.deps.page
    try:
        url = page.url
        title = await page.title()
    except Exception:
        url, title = "Inconnu", "Inconnu"

    return (
        "Tu es un agent web autonome expert (Computer Use).\n"
        f"Page actuelle : '{title}' ({url})\n\n"
        "Règles :\n"
        "1. Ne devine jamais un sélecteur. Utilise read_page d'abord.\n"
        "2. Si une action échoue, observe la page avant de réessayer.\n"
        "3. Utilise screenshot pour voir l'état visuel de la page.\n"
        "4. Préfère les sélecteurs CSS aux coordonnées quand possible."
    )


# ==============================================================================
# 4. TOOLS — Navigation
# ==============================================================================

@agent.tool
async def navigate(ctx: RunContext[BrowserDeps], url: str) -> str:
    """Navigue vers une URL.

    Args:
        url: URL complète (ex: 'https://example.com').
    """
    logger.info(f"[navigate] → {url}")
    await ctx.deps.page.goto(url, wait_until="domcontentloaded")
    return f"Navigué vers {url}"


# ==============================================================================
# 5. TOOLS — Observation
# ==============================================================================

@agent.tool
async def screenshot(ctx: RunContext[BrowserDeps]) -> dict[str, Any]:
    """Prend une capture d'écran de la page actuelle. Utilise cet outil pour voir l'état visuel."""
    logger.info("[screenshot]")
    data = await ctx.deps.page.screenshot()
    return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"}]}


@agent.tool
async def zoom(ctx: RunContext[BrowserDeps], x0: int, y0: int, x1: int, y1: int) -> dict[str, Any]:
    """Capture une zone précise de la page pour voir les détails.

    Args:
        x0: Coordonnée X du coin supérieur gauche.
        y0: Coordonnée Y du coin supérieur gauche.
        x1: Coordonnée X du coin inférieur droit.
        y1: Coordonnée Y du coin inférieur droit.
    """
    logger.info(f"[zoom] région ({x0},{y0}) → ({x1},{y1})")
    data = await ctx.deps.page.screenshot(clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
    return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"}]}


@agent.tool
async def read_page(
    ctx: RunContext[BrowserDeps],
    filter: Literal["all", "interactive"] = "interactive",
) -> str:
    """Lit l'arbre DOM de la page. Indispensable pour trouver les bons sélecteurs avant de cliquer.

    Args:
        filter: 'interactive' pour les éléments cliquables uniquement, 'all' pour tout.
    """
    logger.info(f"[read_page] filtre={filter}")
    result = await ctx.deps.page.evaluate(f"""
        (function() {{
            function getAccessibleTree(root, filter) {{
                const results = [];
                const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
                let node = walker.currentNode;
                while (node) {{
                    const el = node;
                    const isInteractive = el.matches('a, button, input, select, textarea, [onclick], [role="button"]');
                    if (!filter || filter === 'all' || (filter === 'interactive' && isInteractive)) {{
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {{
                            results.push({{
                                tag: el.tagName.toLowerCase(),
                                text: el.textContent?.slice(0, 100) || '',
                                role: el.getAttribute('role') || '',
                                id: el.id || '',
                                class: el.className || '',
                                href: el.href || '',
                            }});
                        }}
                    }}
                    node = walker.nextNode();
                }}
                return results;
            }}
            return JSON.stringify({{
                url: window.location.href,
                title: document.title,
                elements: getAccessibleTree(document.body, '{filter}')
            }});
        }})()
    """)
    return result


@agent.tool
async def get_page_text(ctx: RunContext[BrowserDeps]) -> str:
    """Récupère tout le texte visible de la page (utile pour lire le contenu)."""
    logger.info("[get_page_text]")
    text = await ctx.deps.page.evaluate("document.body.innerText")
    if len(text) > 10000:
        text = text[:10000] + "..."
    return text


# ==============================================================================
# 6. TOOLS — Interaction (clic, saisie, formulaire)
# ==============================================================================

@agent.tool
async def click(
    ctx: RunContext[BrowserDeps],
    coordinate: list[int] | None = None,
    selector: str | None = None,
    button: Literal["left", "right", "middle"] = "left",
    click_count: int = 1,
) -> str:
    """Clique sur un élément de la page.

    Args:
        coordinate: Coordonnées [x, y] du clic (optionnel si selector fourni).
        selector: Sélecteur CSS de l'élément (ex: '#search', 'a.link'). Préféré aux coordonnées.
        button: Bouton de la souris ('left', 'right', 'middle').
        click_count: Nombre de clics (1=simple, 2=double, 3=triple).
    """
    x, y = await resolve_element(ctx.deps.page, coordinate, selector)
    logger.info(f"[click] ({x},{y}) button={button} count={click_count}")
    await ctx.deps.page.mouse.move(x, y)
    for _ in range(click_count):
        await ctx.deps.page.mouse.down(button=button)
        await ctx.deps.page.mouse.up(button=button)
    return f"Cliqué à ({x}, {y})"


@agent.tool
async def type_text(ctx: RunContext[BrowserDeps], text: str) -> str:
    """Tape du texte au clavier. Assure-toi d'avoir cliqué sur le champ de saisie avant.

    Args:
        text: Le texte à taper.
    """
    logger.info(f"[type_text] '{text[:50]}{'...' if len(text) > 50 else ''}'")
    await ctx.deps.page.keyboard.type(text)
    return f"Tapé : {text[:50]}{'...' if len(text) > 50 else ''}"


@agent.tool
async def press_key(ctx: RunContext[BrowserDeps], key: str, repeat: int = 1) -> str:
    """Appuie sur une touche du clavier.

    Args:
        key: Nom de la touche (ex: 'Enter', 'Tab', 'Escape', 'Backspace', 'ArrowDown').
        repeat: Nombre de fois à appuyer.
    """
    mapped = KEY_MAP.get(key.lower(), key)
    logger.info(f"[press_key] {mapped} ×{repeat}")
    for _ in range(repeat):
        await ctx.deps.page.keyboard.press(mapped)
    return f"Touche {mapped} appuyée {repeat}x"


@agent.tool
async def form_input(ctx: RunContext[BrowserDeps], selector: str, value: str) -> str:
    """Remplit un champ de formulaire directement par son sélecteur CSS.

    Args:
        selector: Sélecteur CSS du champ (ex: '#email', 'input[name=search]').
        value: Valeur à insérer.
    """
    logger.info(f"[form_input] {selector} = '{value[:50]}'")
    result = await ctx.deps.page.evaluate(f"""
        (function() {{
            const el = document.querySelector('{selector}');
            if (!el) return {{ success: false, message: 'Élément non trouvé' }};
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) {{
                el.value = {json.dumps(value)};
                el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                return {{ success: true }};
            }}
            return {{ success: false, message: 'Pas un élément de formulaire' }};
        }})()
    """)
    if not result["success"]:
        raise ModelRetry(f"form_input a échoué : {result.get('message')}. Vérifie le sélecteur avec read_page.")
    return f"Valeur '{value[:50]}' insérée dans {selector}"


# ==============================================================================
# 7. TOOLS — Scroll et déplacement
# ==============================================================================

@agent.tool
async def scroll(
    ctx: RunContext[BrowserDeps],
    coordinate: list[int],
    direction: Literal["up", "down", "left", "right"],
    amount: int = 3,
) -> str:
    """Fait défiler la page dans une direction.

    Args:
        coordinate: Position [x, y] du curseur pour le scroll.
        direction: Direction du défilement ('up', 'down', 'left', 'right').
        amount: Intensité du scroll (1=peu, 5=beaucoup). Chaque unité = 100px.
    """
    pixels = amount * 100
    dx = {"left": -pixels, "right": pixels}.get(direction, 0)
    dy = {"up": -pixels, "down": pixels}.get(direction, 0)
    logger.info(f"[scroll] {direction} ×{amount} à ({coordinate[0]},{coordinate[1]})")
    await ctx.deps.page.mouse.move(coordinate[0], coordinate[1])
    await ctx.deps.page.mouse.wheel(dx, dy)
    return f"Scrollé {direction} de {amount} unités"


@agent.tool
async def drag(
    ctx: RunContext[BrowserDeps],
    end_coordinate: list[int],
    start_coordinate: list[int] | None = None,
) -> str:
    """Glisse un élément d'un point à un autre (drag & drop).

    Args:
        end_coordinate: Coordonnées [x, y] de fin du drag.
        start_coordinate: Coordonnées [x, y] de début (optionnel, utilise la position actuelle).
    """
    sx, sy = start_coordinate or [0, 0]
    ex, ey = end_coordinate
    logger.info(f"[drag] ({sx},{sy}) → ({ex},{ey})")
    await ctx.deps.page.mouse.move(sx, sy)
    await ctx.deps.page.mouse.down()
    await ctx.deps.page.mouse.move(ex, ey)
    await ctx.deps.page.mouse.up()
    return f"Glissé de ({sx},{sy}) à ({ex},{ey})"


# ==============================================================================
# 8. TOOLS — Utilitaires
# ==============================================================================

@agent.tool
async def wait(ctx: RunContext[BrowserDeps], duration: float) -> str:
    """Attend un certain temps (utile pour laisser charger une page ou une animation).

    Args:
        duration: Durée en secondes (ex: 1.5).
    """
    logger.info(f"[wait] {duration}s")
    await ctx.deps.page.wait_for_timeout(int(duration * 1000))
    return f"Attendu {duration}s"


# ==============================================================================
# 9. EXÉCUTION
# ==============================================================================

async def main():
    logger.info("Démarrage de Playwright...")

    async with async_playwright() as p:
        browser_instance = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser_instance.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        deps = BrowserDeps(page=page)

        prompt = (
            "Va sur le site https://news.ycombinator.com/, "
            "cherche la barre de recherche en bas, et cherche 'AI Agents'."
        )
        logger.info(f"Ordre : {prompt}")

        result = await agent.run(prompt, deps=deps)

        output: TaskResult = result.output
        logger.success(
            f"RÉPONSE FINALE | Succès: {output.success} | "
            f"Résumé: {output.summary} | "
            f"Actions: {', '.join(output.actions_taken)} | "
            f"URL: {output.final_url}"
        )

        await page.wait_for_timeout(4000)
        await browser_instance.close()


if __name__ == "__main__":
    asyncio.run(main())
