# Tumor Lab — simulateur mécaniste de croissance tumorale

Tumor Lab est un simulateur mécaniste de croissance tumorale et un jeu de stratégie. Il modélise l'hypoxie, la radiothérapie, l'évolution darwinienne et le dilemme contrôle / toxicité — le tout validé sur des données expérimentales publiées.

> **Important :** c'est un modèle éducatif et théorique. Ce n'est pas un dispositif médical, un planificateur de traitement ou un prédicteur patient.

## Architecture du projet

```
tumeur/                    Moteur scientifique Python
  parameters.py            Paramètres et sources bibliographiques
  transport.py             Solveur de réaction-diffusion conservatif
  metabolism.py            ATP, croissance, acidité
  growth.py                Croissance avasculaire à frontière libre
  vascular.py              Angiogenèse et scénario vasculaire
  therapy.py               Radiothérapie LQ/OER
  toxicity.py              NTCP tissu sain
  agents.py                Modèle à agents 3D (validation croisée)
  evolution.py             Mutation, dérive et sélection
  game.py                  Environnement de jeu et observations
  ai.py                    IA explicable (adversaire déterministe)
  validation.py            Vérification et validation
  plotting.py              Figures scientifiques reproductibles

site/                      Application web Flask
  app.py                   Serveur web (lance avec : python site/app.py)
  templates/               Templates Jinja2
  static/                  CSS, images de résultats
  tumorlab.db              Base SQLite

resultats/                 Figures, logs JSON et rapport Markdown
unity/                     Scripts C# pour l'intégration Unity
tests/                     Tests unitaires
server.py                  Adaptateur JSON Lines / HTTP
run_simulation.py          Études et figures régénérées
check_setup.py             Diagnostic de l'environnement
```

## Démarrage rapide

### Site web

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install flask
python3 site/app.py
# → http://localhost:5000
```

### Simulation scientifique

```bash
python3 -m pip install -r requirements.txt
python3 check_setup.py
python3 run_simulation.py --validate
python3 run_simulation.py --vascular
```

### Serveur Unity

```bash
python3 server.py
```

## Ce que modélise le moteur

- **Microenvironnement** : O₂, glucose, lactate et pH extracellulaire résolus par volumes finis conservatifs.
- **Vascularisation** : angiogenic switch hypoxique, perfusion imparfaite, pression anti-angiogénique.
- **Radiothérapie** : modèle linéaire-quadratique avec OER, fractionnement clinique et NTCP tissu sain.
- **Évolution darwinienne** : mutation, dérive génétique et sélection intratumorale.
- **IA explicable** : adversaire déterministe ne lisant que les observations publiques.
- **Jeu de stratégie** : actions partiellement observables, score pénalisant la toxicité.

## Deux régimes

| Régime | Usage | Statut de preuve |
|---|---|---|
| `in_vitro_strict` | Sphéroïde EMT6/Ro en culture contrôlée | Calibré/validé contre des données publiées |
| `vascular_human` | Tumeur fictive avec angiogenèse | Scénario basé sur des mécanismes ; pas des mesures patient |

## Résultats

Les figures de validation et les logs sont dans `resultats/` et affichés sur le site web dans la section « Résultats Scientifiques ».

- 31/32 checks réussis contre les données publiées ; l'écart restant est documenté
- Validation croisée avec le modèle à agents 3D
- Spectre allélique validé par la loi 1/f

## Dossier de validation scientifique

La procédure de soutenance, le périmètre de preuve, les limites connues et le déploiement PythonAnywhere sont rassemblés dans [`VALIDATION_DOSSIER.md`](VALIDATION_DOSSIER.md). Le résultat revendicable reste 31/32 critères passés : l'écart de nécrose en milieu riche est conservé et documenté.

## Déploiement PythonAnywhere

Le point d'entrée WSGI est `wsgi.py`. Définir `TUMORLAB_SECRET_KEY` dans les variables d'environnement, désactiver le debug et utiliser un seul worker pour la démonstration : les parties sont actuellement conservées en mémoire. Le navigateur sérialise les actions pendant le calcul du solveur ; cela évite les doubles tours, mais ne supprime pas le coût scientifique d'environ une seconde par tour.

## Reproductibilité

- Utiliser une graine fixe pour les comparaisons
- Enregistrer la graine et le régime modèle dans les rapports
- Ne jamais détendre une tolérance scientifique pour faire passer un test
- Lancer `check_setup.py` et les tests avant un pull request

## Notes de sécurité

- `server.py --http` est un adaptateur de développement, pas un service de production.
- Le leaderboard local est éditable par l'utilisateur.
- Ne jamais afficher `truth` aux joueurs.
- Ne jamais commettre de secrets, données patient ou artwork non licencié.
