# Tumor Lab — a mechanistic tumour growth simulator

Tumor Lab is a mechanistic tumour growth simulator and a strategy game. It models hypoxia, radiotherapy, Darwinian evolution and the control/toxicity dilemma — all validated against published experimental data.

> **Important:** this is an educational and theoretical model. It is not a medical device, a treatment planner or a patient predictor.

## Project architecture

```
tumeur/                    Python scientific engine
  parameters.py            Parameters and bibliographic sources
  transport.py             Conservative reaction-diffusion solver
  metabolism.py            ATP, growth, acidity
  growth.py                Avasculary free-boundary growth
  vascular.py              Angiogenesis and vascular scenario
  therapy.py               LQ/OER radiotherapy
  toxicity.py              NTCP in healthy tissue
  agents.py                3D agent-based model (cross-validation)
  evolution.py             Mutation, drift and selection
  game.py                  Game environment and observations
  ai.py                    Explainable AI (deterministic adversary)
  validation.py            Verification and validation
  plotting.py              Reproducible scientific figures

site/                      Flask web application
  app.py                   Web server (run with: python site/app.py)
  templates/               Jinja2 templates
  static/                  CSS, result images
  tumorlab.db              SQLite database

resultats/                 Figures, JSON logs and Markdown report
unity/                     C# scripts for Unity integration
tests/                     Unit tests
server.py                  Lines JSON / HTTP adapter
run_simulation.py          Studies and regenerated figures
check_setup.py             Environment diagnostics
```

## Quick Start

### Website

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install flask
python3 site/app.py
# → http://localhost:5000
```

### Scientific simulation

```bash
python3 -m pip install -r requirements.txt
python3 check_setup.py
python3 run_simulation.py --validate
python3 run_simulation.py --vascular
```

### Unity server

```bash
python3 server.py
```

## What the engine models

- **Microenvironment**: O₂, glucose, lactate and extracellular pH modelled using conservative finite volumes.
- **Vascularisation**: hypoxic angiogenic switch, imperfect perfusion, anti-angiogenic pressure.
- **Radiotherapy**: linear-quadratic model with OER, clinical fractionation and NTCP in healthy tissue.
- **Darwinian evolution**: mutation, genetic drift and intratumoural selection.
- **Explainable AI**: deterministic adversary that only reads public observations.
- **Strategy game**: partially observable actions, score penalising toxicity.

## Two systems

| Régime            | Usage                                   | Statut de preuve                                       |
| ----------------- | --------------------------------------- | ------------------------------------------------------ |
| `in_vitro_strict` | EMT6/Ro spheroids in controlled culture | Calibrated/validated against published data            |
| `vascular_human`  | Fictitious tumour with angiogenesis     | Scenario based on mechanisms; not patient measurements |

## Results

The validation figures and logs are in `resultats/` and are displayed on the website in the ‘Scientific Results’ section.

- 31 out of 32 checks passed against published data; the remaining discrepancy is documented
- Cross-validation with the 3D agent-based model
- Allelic spectrum validated by the 1/f law

## PythonAnywhere deployment

The WSGI entry point is `wsgi.py`. Set `TUMORLAB_SECRET_KEY` in the environment variables, disable debugging and use a single worker for the demonstration: the game pieces are currently kept in memory. The browser serialises the actions whilst the solver is calculating; this prevents double turns, but does not eliminate the computational cost of approximately one second per turn.

## Reproducibility

- **Use a fixed seed for comparisons
- Record the seed and the test suite in the reports
- Never relax a scientific tolerance just to pass a test
- Run `check_setup.py` and the tests before submitting a pull request**

## Safety notes

- `server.py --http` is a development adapter, not a production service.
- The local leaderboard can be edited by the user.
- Never display `truth` to players.
- Never commit secrets, patient data or unlicensed artwork.
