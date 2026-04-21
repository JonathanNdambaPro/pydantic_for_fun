---
name: Structure des dossiers et nommage des fichiers exemples
description: Fichiers dans leur dossier numéroté, numérotation interne à partir de 1 (ex: 14_evals/1_evals.py)
type: feedback
---

Toujours créer les fichiers exemples dans leur dossier numéroté correspondant, pas à la racine de `pydantic_ai_for_fun/`. La numérotation des fichiers repart à 1 dans chaque dossier.

**Why:** Le projet est organisé par dossiers numérotés (1_agent/, 2_dependencies/, etc.) contenant chacun un `__init__.py` et des fichiers numérotés à partir de 1 (ex: 1_base_setup.py, 2_base_setup.py...). L'utilisateur a corrigé deux erreurs : fichier créé à la racine, et numérotation reprenant le numéro du dossier au lieu de 1.

**How to apply:** Pour un dossier `N_concept/`, le premier fichier s'appelle `1_xxx.py`, le second `2_xxx.py`, etc. Toujours vérifier les fichiers existants dans le dossier pour trouver le bon numéro.
