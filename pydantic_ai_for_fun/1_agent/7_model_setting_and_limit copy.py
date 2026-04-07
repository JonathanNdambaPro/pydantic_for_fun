from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

"""
RAPPEL : EFFET DE LA TEMPÉRATURE (T) SUR LE LLM

La température agit comme un scalaire sur les logits avant la fonction softmax.
- T = 0.0 (Greedy Decoding) : Déterministe. Le modèle prend le token le plus probable (argmax). Focus absolu sur la précision, 0 hallucination (idéal JSON/SQL/RAG).
- T = 1.0 (Baseline) : Conservation des probabilités natives du modèle.
- T > 1.0 (Haute Entropie) : Lisse la distribution. Favorise la diversité des mots générés, idéal pour l'idéation, mais augmente les hallucinations.
"""

# Niveau 1 : Model (Baseline globale)
model = OpenAIChatModel(
    'gpt-4o',
    settings=ModelSettings(temperature=0.8, max_tokens=500)
)

# Niveau 2 : Agent (Surcharge contextuelle)
agent = Agent(
    model,
    model_settings=ModelSettings(temperature=0.5)
)

# Niveau 3 : Run-time (Override prioritaire à l'exécution)
result = agent.run_sync(
    'Question précise',
    model_settings=ModelSettings(temperature=0.0)
)

# Payload final envoyé à l'API : {temperature: 0.0, max_tokens: 500}
