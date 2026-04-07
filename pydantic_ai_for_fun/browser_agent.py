import asyncio

from datetime import date

import httpx
import logfire
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from loguru import logger
from pydantic_ai import Agent, RunContext

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])

# On définit le modèle à utiliser
agent = Agent(
    "gateway/openai:gpt-4o",
    system_prompt=(
        "Tu es un Agent Web Explorateur. Ton but est de naviguer sur Internet pour répondre "
        "aux questions de l'utilisateur ou simplement explorer un sujet.\n"
        "Comporte-toi comme un humain : si tu ne connais pas la réponse ou si l'information doit être récente, fais d'ABORD une recherche web via tes outils.\n"
        "Pense à bien donner un résumé structuré et clair avec le nom de tes sources (l'URL)."
    )
)

@agent.system_prompt
def add_today_date() -> str:
    """Injecte la date du jour pour que le LLM sache si les informations sont récentes."""
    return f"Information de Contexte Système : La date d'aujourd'hui est le {date.today()}."

@agent.tool
async def search_web(ctx: RunContext[None], query: str) -> str:
    """Recherche des informations ou actualités récentes sur le web et retourne une liste de titres, résumés et de liens (URLs)."""
    logger.info("🔍 [Tool] Recherche web lancée pour : '{query}'...", query=query)
    
    # DDGS().news contourne les blocages classiques de la recherche web HTML.
    # En cas d'erreur ou d'absence, le fallback vers 'text' est fait, mais .news est très puissant et récent.
    try:
        # DDGS est asynchrone par nature dans les scripts mais l'import direct de la lib est synchrone.
        # On utilise to_thread pour éviter de bloquer l'event loop.
        results = await asyncio.to_thread(DDGS().news, query, max_results=5)
    except Exception as e:
        return f"Erreur lors de la recherche : {str(e)}"
    
    if not results:
        # Fallback au cas où il n'y a pas d'actualités
        try:
            results = await asyncio.to_thread(DDGS().text, query, max_results=5)
        except Exception:
            return "Aucun résultat trouvé sur DuckDuckGo (API bloquée ou aucune réponse)."
            
    if not results:
        return "Aucun résultat trouvé."
        
    formatted_results = []
    for r in results:
        # Gère la structure de DDGS().news ou DDGS().text
        title = r.get('title', 'Sans Titre')
        url = r.get('url', r.get('href', 'Pas de lien'))
        body = r.get('body', '')
        date_str = r.get('date', 'Date inconnue')
        formatted_results.append(f"Titre: {title}\nDate: {date_str}\nURL: {url}\nRésumé court: {body}")
        
    return "\n\n---\n\n".join(formatted_results)

@agent.tool
async def read_url(ctx: RunContext[None], url: str) -> str:
    """Lit le contenu d'une page web (texte brut) à partir de son URL."""
    logger.info("🌐 [Tool] Visite de la page : {url}...", url=url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # On parse le HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # On nettoie le HTML en enlevant les scripts et le style
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                
            # Extrait le texte brut
            text = soup.get_text(separator=' ', strip=True)
            
            # On limite la taille pour ne pas exploser la fenêtre de contexte du LLM
            # (Environ 8000 caractères, ce qui fait ~2000 tokens)
            if len(text) > 8000:
                text = text[:8000] + "\n...[Contenu tronqué]..."
                
            return text
            
        except Exception as e:
            return f"Impossible de lire la page. Erreur : {str(e)}"

async def main():
    logger.info("🚀 Démarrage de l'Agent Web Explorateur...")
    
    # Tu peux changer la requête ici !
    requete = "Quelles sont les dernières actualités majeures sur l'IA générative aujourd'hui ?"
    logger.info("🧠 Utilisateur demande : {requete}", requete=requete)
    
    # On lance l'agent. Il va chercher sur DDG, puis lire un des liens, puis construire sa réponse !
    result = await agent.run(requete)
    
    logger.success("Réponse Finale de l'Agent reçue !")
    print("\n" + "="*50)
    print("✅ Réponse Finale de l'Agent :")
    print("="*50)
    print(getattr(result, "data", getattr(result, "output", str(result))))

if __name__ == "__main__":
    asyncio.run(main())
