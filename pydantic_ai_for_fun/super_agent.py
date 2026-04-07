import os
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

import asyncio
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import List

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Assurez-vous d'importer vos settings
from pydantic_ai_for_fun.settings import settings

# 1. LOGFIRE : C'est ce qui rend Pydantic AI magique en production.
# Ça va tracer *chaque* étape du raisonnement de l'agent, les appels d'outils, etc.
# Vous pourrez tout voir sur l'interface Logfire !
logfire.configure(send_to_logfire='if-token-present')


# 2. OUTPUT STRUCTURÉ GARANTI (Pydantic Models)
# L'agent n'a pas le choix, il DOIT retourner des données qui matchent ce schéma.
# Finis les vieux JSON parsés à la main avec plein d'erreurs !
class DailyActivity(BaseModel):
    time: str = Field(description="Ex: 'Matin', '14:00'.")
    activity: str = Field(description="Description de l'activité.")
    weather_dependent: bool = Field(description="True si ça nécessite qu'il fasse beau.")

class TravelDay(BaseModel):
    day_date: date = Field(description="Date du jour.")
    expected_weather: str = Field(description="Résumé de la météo prévue.")
    activities: List[DailyActivity]

class TravelItinerary(BaseModel):
    destination: str = Field(description="Ville de destination.")
    total_days: int
    days: List[TravelDay]
    packing_tips: List[str] = Field(description="Les affaires indispensables selon la météo !")


# 3. DÉPENDANCES INJECTÉES (L'agent a un contexte sécurisé)
# C'est ultra puissant car on ne passe pas les clés API ou la DB dans le prompt systeme,
# on l'injecte proprement au moment du `run()`.
@dataclass
class PlannerDependencies:
    user_name: str
    openweathermap_api_key: str


# 4. DÉFINITION DE L'AGENT
# On passe par OpenRouter pour avoir accès facilement à Gemini ou autre sans proxy Google capricieux !
os.environ["OPENAI_API_KEY"] = settings.OPENROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

agent = Agent(
    "openai:google/gemini-2.5-pro", # Modèle puissant via OpenRouter
    deps_type=PlannerDependencies,
    output_type=TravelItinerary, # On force le retour structuré
    retries=3, # On gère les erreurs de parsage JSON du LLM en auto-retentant
    system_prompt=(
        "Tu es un agent de voyage expert, IA de luxe."
        "Ta mission est de générer un itinéraire parfait."
        "Tu DOIS utiliser tes outils pour vérifier la météo de la destination "
        "avant de proposer des activités. Adapte les activités à la météo (ex: musée s'il pleut)."
        "Sois précis, créatif, et retourne un plan structuré."
    )
)

# 5. SYTEM PROMPT DYNAMIQUE
# Il s'actualise à chaque run grâce aux dépendances
@agent.system_prompt
def add_dynamic_context(ctx: RunContext[PlannerDependencies]) -> str:
    return f"Le client actuel s'appelle {ctx.deps.user_name}. Aujourd'hui, nous sommes le {date.today()}."

# 6. TOOL #1 : Trouver les coordonnées geographiques
@agent.tool
def get_coordinates(ctx: RunContext[PlannerDependencies], city_name: str) -> dict:
    """Trouve la latitude et la longitude d'une ville donnée."""
    print(f"🌍 [Tool] Recherche des coordonnées pour : {city_name}...")
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={urllib.parse.quote(city_name)}&limit=1&appid={ctx.deps.openweathermap_api_key}"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        if not data:
            return {"error": "Ville introuvable"}
        return {"lat": data[0]["lat"], "lon": data[0]["lon"]}
    except Exception as e:
        return {"error": str(e)}

# 7. TOOL #2 : Prendre la météo
@agent.tool
def get_weather(ctx: RunContext[PlannerDependencies], lat: float, lon: float) -> dict:
    """Récupère les prévisions météo pour les jours à venir aux coordonnées données."""
    print(f"🌤️ [Tool] Recherche de la météo pour : {lat}, {lon}...")
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={ctx.deps.openweathermap_api_key}"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read())
        # On retourne juste un condensé pour ne pas exploser le token limit
        forecasts = []
        for item in data.get("list", [])[:16:2]: # Un point toutes les 6 heures environ
            forecasts.append({
                "date": item["dt_txt"], 
                "temp": item["main"]["temp"], 
                "description": item["weather"][0]["description"]
            })
        return {"forecast": forecasts}
    except Exception as e:
        return {"error": str(e)}


async def main():
    # On initialise nos dépendances
    deps = PlannerDependencies(
        user_name="Jonathan", # On a vu ça dans le pyproject.toml ;)
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY
    )

    print("🚀 Lancement de l'Agent pydantic-ai de ouf...")
    destination = "Tokyo"
    print(f"Demande : Je veux partir à {destination} pour 2 jours.")

    # 8. LE RUN ASYNCHRONE
    # C'est là que la magie opère. L'agent va lire le prompt, utiliser les outils (météo), 
    # et construire la réponse typée (TravelItinerary).
    try:
        result = await agent.run(
            f"Je souhaite partir 2 jours à {destination}. Cree un super itinéraire !",
            deps=deps
        )
        
        # Le resultat (result.data) est un objet Pydantic TravelItinerary 100% valide
        itinerary: TravelItinerary = result.data
        
        print("\n\n✅ RÉSULTAT OBTENU (Parsé automatiquement en objets Python !)")
        print(f"Destination : {itinerary.destination} ({itinerary.total_days} jours)")
        print("-" * 40)
        
        for day in itinerary.days:
            print(f"📅 {day.day_date} | Météo : {day.expected_weather}")
            for act in day.activities:
                meteo_icon = "☀️" if act.weather_dependent else "☂️"
                print(f"  [{act.time}] {meteo_icon} {act.activity}")
        
        print("-" * 40)
        print("🎒 Packing Tips :")
        for tip in itinerary.packing_tips:
            print(f" - {tip}")

    except Exception as e:
        print(f"Erreur pendant l'exécution: {e}")

if __name__ == "__main__":
    import os
    # Un petit check pour s'assurer que Logfire ne bloque pas si t'as pas start
    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"
    asyncio.run(main())
