---
name: generate-example
description: Génère un fichier exemple Pydantic AI avec la structure conventionnelle du projet. Utilise quand l'utilisateur demande de créer un fichier d'exemple, de démo ou de sample.
argument-hint: <nom_du_concept>
allowed-tools: Write Read Glob
---

# Générer un fichier exemple Pydantic AI

Génère un fichier exemple pour le concept : $ARGUMENTS

## Structure obligatoire

Chaque fichier exemple DOIT suivre cette structure dans l'ordre :

### 1. Imports
- Triés : stdlib → third-party → pydantic_ai
- Séparés par des lignes vides entre chaque groupe

### 2. Boilerplate (toujours présent)
```python
import logfire
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
logfire.configure()
logfire.instrument_pydantic_ai()
logger.configure(handlers=[logfire.loguru_handler()])
```

### 3. Docstring explicatif
- Entre triple guillemets `"""`
- En français
- Titre en MAJUSCULES + tiret + description courte
- Explication du concept
- Liste des cas d'usage ou points clés

### 4. Parties numérotées
- Séparées par des blocs commentaires :
```python
# =====================================================================
# PARTIE N : Titre de la partie
# =====================================================================
```
- Chaque partie couvre un aspect du concept
- Commentaires explicatifs avant chaque bloc de code

### 5. Agents
- Modèle : `'gateway/anthropic:claude-sonnet-4-6'`
- Instructions en français
- Typer les deps si nécessaire

### 6. Logging
- `logger.info()` pour les actions
- `logger.success()` pour les résultats
- `logger.warning()` pour les alertes

### 7. Exécution
- Dernière partie : `if __name__ == "__main__":`
- Démos numérotées avec des commentaires `# --- Démo N : ... ---`
- Chaque démo illustre un aspect différent du concept

## Où placer le fichier
- Dans le dossier correspondant au thème sous `pydantic_ai_for_fun/`
- Nommé avec un numéro de séquence : `N_nom_du_concept.py`
- Vérifier les fichiers existants pour trouver le bon numéro
