import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union

from playwright.async_api import Page, async_playwright
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry


# ==============================================================================
# 1. OUTIL BROWSER (Converti en 100% Asynchrone)
# ==============================================================================
class AsyncBrowserTool:
    def __init__(self, page: Page) -> None:
        self.page = page

    async def execute(self, action: str, **kwargs: Any) -> dict[str, Any]:
        """Exécute l'action de manière asynchrone."""
        try:
            match action:
                case "navigate": return await self._navigate(kwargs["url"])
                case "screenshot": return await self._screenshot()
                case "zoom": return await self._zoom(kwargs["region"])
                case "left_click": return await self._click("left", 1, **kwargs)
                case "right_click": return await self._click("right", 1, **kwargs)
                case "middle_click": return await self._click("middle", 1, **kwargs)
                case "double_click": return await self._click("left", 2, **kwargs)
                case "triple_click": return await self._click("left", 3, **kwargs)
                case "left_click_drag": return await self._drag(kwargs.get("start_coordinate"), kwargs["coordinate"])
                case "left_mouse_down": return await self._mouse_down(kwargs["coordinate"])
                case "left_mouse_up": return await self._mouse_up(kwargs["coordinate"])
                case "type": return await self._type(kwargs["text"])
                case "key" | "press_key": return await self._press_key(kwargs["text"], kwargs.get("repeat", 1))
                case "hold_key": return await self._hold_key(kwargs["text"], kwargs["duration"])
                case "scroll": return await self._scroll(kwargs["coordinate"], kwargs["scroll_direction"], kwargs["scroll_amount"])
                case "scroll_to": return await self._scroll_to(kwargs["ref"])
                case "read_page": return await self._read_page(kwargs.get("filter"))
                case "get_page_text": return await self._get_page_text()
                case "wait": return await self._wait(kwargs["duration"])
                case "form_input": return await self._form_input(kwargs["ref"], kwargs["value"])
                case "create_tab" | "switch_tab" | "close_tab": return self._error(f"{action} not supported")
                case "list_tabs": return self._text("1 tab open")
                case _: return self._error(f"Unknown action: {action}")
        except Exception as e:
            return self._error(f"Browser action failed: {e}")

    async def _navigate(self, url: str) -> dict:
        await self.page.goto(url, wait_until="domcontentloaded")
        return self._text(f"Navigated to {url}")

    async def _screenshot(self) -> dict:
        data = await self.page.screenshot()
        return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"}]}

    async def _zoom(self, region: list[int]) -> dict:
        x0, y0, x1, y1 = region
        data = await self.page.screenshot(clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
        return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": "image/png"}]}

    async def _click(self, button: str, click_count: int, coordinate: list[int] | None = None, ref: str | None = None, selector: str | None = None, **_: Any) -> dict:
        if coordinate:
            x, y = coordinate
        elif ref or selector:
            q = f'[data-ref="{ref}"]' if ref else selector
            el = await self.page.query_selector(q)
            if not el: return self._error(f"Element introuvable avec '{q}'")
            box = await el.bounding_box()
            if not box: return self._error(f"Element '{q}' caché")
            x = box["x"] + box["width"] / 2
            y = box["y"] + box["height"] / 2
        else:
            return self._error("coordinate, ref, ou selector requis")

        await self.page.mouse.move(x, y)
        btn = button if button in ("left", "right", "middle") else "left"
        for _ in range(click_count):
            await self.page.mouse.down(button=btn)
            await self.page.mouse.up(button=btn)
        return self._text(f"Clicked at ({x}, {y})")

    async def _drag(self, start: list[int] | None, end: list[int]) -> dict:
        sx, sy = (start or [0, 0])
        ex, ey = end
        await self.page.mouse.move(sx, sy)
        await self.page.mouse.down()
        await self.page.mouse.move(ex, ey)
        await self.page.mouse.up()
        return self._text(f"Dragged from ({sx}, {sy}) to ({ex}, {ey})")

    async def _mouse_down(self, coordinate: list[int]) -> dict:
        await self.page.mouse.move(coordinate[0], coordinate[1])
        await self.page.mouse.down()
        return self._text(f"Mouse down at {coordinate}")

    async def _mouse_up(self, coordinate: list[int]) -> dict:
        await self.page.mouse.move(coordinate[0], coordinate[1])
        await self.page.mouse.up()
        return self._text(f"Mouse up at {coordinate}")

    async def _type(self, text: str) -> dict:
        await self.page.keyboard.type(text)
        return self._text(f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")

    _KEY_MAP: dict[str, str] = {
        "enter": "Enter", "return": "Enter", "tab": "Tab", "escape": "Escape", "esc": "Escape",
        "backspace": "Backspace", "delete": "Delete", "space": "Space", "up": "ArrowUp",
        "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight", "home": "Home",
        "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
    }

    async def _press_key(self, text: str, repeat: int = 1) -> dict:
        keys = [self._KEY_MAP.get(k.lower(), k) for k in text.split()]
        for _ in range(repeat):
            for key in keys: await self.page.keyboard.press(key)
        return self._text(f"Pressed key(s): {text}")

    async def _hold_key(self, text: str, duration: float) -> dict:
        key = self._KEY_MAP.get(text.lower(), text)
        await self.page.keyboard.down(key)
        await self.page.wait_for_timeout(int(duration * 1000))
        await self.page.keyboard.up(key)
        return self._text(f"Held {text} for {duration}s")

    async def _scroll(self, coordinate: list[int], direction: str, amount: int) -> dict:
        pixels = amount * 100
        dx = {"left": -pixels, "right": pixels}.get(direction, 0)
        dy = {"up": -pixels, "down": pixels}.get(direction, 0)
        await self.page.mouse.move(coordinate[0], coordinate[1])
        await self.page.mouse.wheel(dx, dy)
        return self._text(f"Scrolled {direction} by {amount}")

    async def _scroll_to(self, ref: str) -> dict:
        await self.page.evaluate(f"""
            (function() {{
                const el = document.querySelector('[data-ref="{ref}"]') || document.getElementById('{ref}');
                if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }})()
        """)
        return self._text(f"Scrolled to element: {ref}")

    async def _read_page(self, filter_: str | None = None) -> dict:
        result = await self.page.evaluate(f"""
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
                    elements: getAccessibleTree(document.body, '{filter_ or ""}')
                }});
            }})()
        """)
        return self._text(result)

    async def _get_page_text(self) -> dict:
        text = await self.page.evaluate("document.body.innerText")
        if len(text) > 10000:
            text = text[:10000] + "..."
        return self._text(text)

    async def _wait(self, duration: float) -> dict:
        await self.page.wait_for_timeout(int(duration * 1000))
        return self._text(f"Waited {duration}s")

    async def _form_input(self, ref: str, value: str) -> dict:
        result = await self.page.evaluate(f"""
            (function() {{
                const el = document.querySelector('[data-ref="{ref}"]') || document.getElementById('{ref}');
                if (!el) return {{ success: false, message: 'Element not found' }};
                if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) {{
                    el.value = {json.dumps(value)};
                    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    return {{ success: true }};
                }}
                return {{ success: false, message: 'Not a form element' }};
            }})()
        """)
        if not result["success"]: return self._error(result.get("message", "Failed"))
        return self._text(f"Set value for {ref}")

    @staticmethod
    def _text(text: str) -> dict: return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def _error(message: str) -> dict: return {"content": [{"type": "text", "text": message}], "is_error": True}


# ==============================================================================
# 2. MODÈLES PYDANTIC (Avec Docstrings pour guider l'IA)
# ==============================================================================

class ActionNavigate(BaseModel):
    """Ouvre une URL cible."""
    action: Literal["navigate"]
    url: str

class ActionScreenshot(BaseModel):
    action: Literal["screenshot"]

class ActionZoom(BaseModel):
    action: Literal["zoom"]
    region: list[int] = Field(description="Région [x0, y0, x1, y1]")

class ActionClick(BaseModel):
    """Clique sur un élément. Privilégie le 'selector' CSS si tu le connais."""
    action: Literal["left_click", "right_click", "middle_click", "double_click", "triple_click"]
    coordinate: Optional[list[int]] = Field(None, description="Coordonnées exactes [x, y]")
    ref: Optional[str] = Field(None, description="Attribut data-ref ou ID de l'élément cible")
    selector: Optional[str] = Field(None, description="Sélecteur CSS (ex: '#btn-login', 'a.link')")

class ActionDrag(BaseModel):
    action: Literal["left_click_drag"]
    coordinate: list[int] = Field(description="Coordonnées de fin [x, y]")
    start_coordinate: Optional[list[int]] = Field(None, description="Coordonnées de départ [x, y]")

class ActionMouse(BaseModel):
    action: Literal["left_mouse_down", "left_mouse_up"]
    coordinate: list[int]

class ActionType(BaseModel):
    """Tape du texte. Tu dois t'assurer d'avoir cliqué sur le champ de saisie avant !"""
    action: Literal["type"]
    text: str = Field(description="Le texte à taper au clavier")

class ActionKey(BaseModel):
    action: Literal["key", "press_key"]
    text: str = Field(description="Ex: 'Enter', 'Tab', 'Escape'")
    repeat: int = 1

class ActionHoldKey(BaseModel):
    action: Literal["hold_key"]
    text: str
    duration: float

class ActionScroll(BaseModel):
    action: Literal["scroll"]
    coordinate: list[int]
    scroll_direction: Literal["up", "down", "left", "right"]
    scroll_amount: int

class ActionScrollTo(BaseModel):
    action: Literal["scroll_to"]
    ref: str

class ActionReadPage(BaseModel):
    """Scanner l'arbre DOM (très utile pour trouver les bons sélecteurs d'une page)."""
    action: Literal["read_page"]
    filter: Optional[Literal["all", "interactive"]] = Field("interactive")

class ActionGetPageText(BaseModel):
    action: Literal["get_page_text"]

class ActionWait(BaseModel):
    """Attend quelques secondes (ex: pour laisser charger une animation)."""
    action: Literal["wait"]
    duration: float

class ActionFormInput(BaseModel):
    action: Literal["form_input"]
    ref: str
    value: str

class ActionTabs(BaseModel):
    action: Literal["create_tab", "switch_tab", "close_tab", "list_tabs"]

BrowserAction = Annotated[
    ActionNavigate | ActionScreenshot | ActionZoom | ActionClick | ActionDrag | ActionMouse | ActionType | ActionKey | ActionHoldKey | ActionScroll | ActionScrollTo | ActionReadPage | ActionGetPageText | ActionWait | ActionFormInput | ActionTabs,
    Field(discriminator="action")
]


# ==============================================================================
# 3. AGENT ET DÉPENDANCES
# ==============================================================================

@dataclass
class BrowserDeps:
    browser_tool: AsyncBrowserTool

agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    deps_type=BrowserDeps,
    retries=3, # 🔥 Accorde à l'agent 3 essais de correction automatique
)

# 🧠 CONTEXTE DYNAMIQUE : Evalué avant chaque réflexion de l'IA !
@agent.system_prompt
async def dynamic_browser_context(ctx: RunContext[BrowserDeps]) -> str:
    page = ctx.deps.browser_tool.page
    try:
        url = page.url
        title = await page.title()
    except Exception:
        url, title = "Inconnu", "Inconnu"

    return (
        "Tu es un agent web autonome expert (Computer Use).\n"
        f"📍 [ÉTAT ACTUEL] Tu te trouves sur la page : '{title}' ({url})\n\n"
        "Règles d'or :\n"
        "1. Ne devine jamais un sélecteur à l'aveugle. Fais un 'read_page' si tu as un doute.\n"
        "2. Si ton outil plante (Timeout ou Inexistant), n'abandonne pas, observe la page pour trouver la bonne cible."
    )

@agent.tool
async def computer_use(ctx: RunContext[BrowserDeps], command: BrowserAction) -> dict[str, Any]:
    """Exécute une action sur le navigateur via l'outil Computer Use."""
    kwargs = command.model_dump(exclude_none=True)
    action = kwargs.pop("action")

    print(f"🤖 [Action IA] : {action} | args: {kwargs}")

    result = await ctx.deps.browser_tool.execute(action, **kwargs)

    # 🛡️ LA MAGIE DU MODELRETRY ICI
    if result.get("is_error"):
        error_msg = result.get("content", [{}])[0].get("text", "Erreur inconnue")
        print(f"⚠️ [Erreur interceptée] : {error_msg} -> L'IA va se corriger.")

        # Lève l'exception : Bloque l'itération classique et ordonne à Claude de se corriger
        raise ModelRetry(  # noqa: TRY003
            f"L'action a échoué avec l'erreur : '{error_msg}'. "
            "Réessaie avec de nouveaux paramètres, ou utilise 'read_page' pour analyser la structure de la page."
        )

    return result


# ==============================================================================
# 4. EXÉCUTION ASYNCHRONE
# ==============================================================================

async def main():
    print("Démarrage de Playwright en Asynchrone...")
    
    # "async with" libère le thread principal, offrant d'excellentes performances
    async with async_playwright() as p:
        browser_instance = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"] # Petite astuce anti-bot
        )
        
        # Figer le viewport aide grandement l'IA avec la gestion de l'espace / des coordonnées
        context = await browser_instance.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        tool = AsyncBrowserTool(page)
        deps = BrowserDeps(browser_tool=tool)
        
        prompt = "Va sur le site https://news.ycombinator.com/, cherche la barre de recherche en bas, et cherche 'AI Agents'."
        print(f"\n▶️ Ordre : {prompt}\n")
        
        # 🚀 Lancement avec await agent.run()
        result = await agent.run(prompt, deps=deps)
        
        print("\n✅ === RÉPONSE FINALE DE L'AGENT ===")
        print(result.data)

        await page.wait_for_timeout(4000)
        await browser_instance.close()

if __name__ == "__main__":
    asyncio.run(main())