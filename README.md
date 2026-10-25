# Tumor Lab — Interactive Tumor Detection, Diagnosis & Radiotherapy Simulator

Tumor Lab is an interactive oncology simulation platform and turn-based clinical strategy game. It models **diagnostic uncertainty (noisy scans, serum biomarkers, pO₂ hypoxia imaging, needle biopsies)** alongside **reaction-diffusion tumor physics, Darwinian clonal resistance, and healthy-organ toxicity** — all grounded in peer-reviewed experimental literature (31/32 validation checks passed).

> **Important:** Tumor Lab is an educational and theoretical model. It is not a medical device, a clinical diagnostic tool, or a patient treatment planner.

---

## Designed for Everyone — Zero Biology Background Required

Following preliminary mentor feedback, Tumor Lab is designed on two levels so **complete beginners** can play and learn immediately while **researchers and judges** can inspect every equation:

- **1-Click Guest Mode**: Jump straight into any simulation without creating an account.
- **5-Step Interactive Guided Tutorial**: Walks first-time visitors through reading noisy scans, running an Oxygen Scan (`pO₂`), performing a Needle Biopsy, and applying safe radiation doses.
- **Beginner Help Toggle (`ON` by default)**: Plain-language translations under every metric, visual `400 µm` target ring and `1 mm` scale bar, real-time *"What just happened?"* turn summaries, and `"In plain words"` explanations under all 9 scientific figures.
- **3-Minute Visual Guide (`/learn`)**: Everyday analogies, a live interactive Dose × Oxygen radiobiology sandbox, a jargon-free glossary, and a quick self-check quiz.
- **Bilingual Interface (`EN` / `FR`)**: Instant 1-click language switch in the top navigation bar, plus Light/Dark mode.

---

## Project Architecture

```text
tumeur/                    Python scientific & neural network engine
  parameters.py            Parameters with explicit bibliographic provenance
  transport.py             Conservative finite-volume reaction-diffusion solver
  metabolism.py            ATP bioenergetics, proliferation, extracellular pH
  growth.py                Free-boundary spheroid PDE solver (SSP-RK2 + van Leer TVD)
  vascular.py              Hypoxic angiogenic switch & vascular scenario
  therapy.py               Linear-Quadratic + OER radiotherapy model
  toxicity.py              Lyman-Kutcher-Burman (LKB) healthy-organ NTCP
  agents.py                3D cell-by-cell agent model (cross-validation)
  evolution.py             Darwinian mutation, genetic drift & 1/f power law
  game.py                  Partially observable clinical environment (POMDP)
  neural_net.py            Pure-NumPy Actor-Critic MLP (forward, backprop, Adam, gradcheck)
  ai.py                    Trained PolicyValueNet inference & input-gradient saliency
  ai_weights.npz           Saved trained neural network weights (5,575 parameters)
  validation.py            32 numerical verification & experimental validation checks
  plotting.py              Reproducible scientific matplotlib figures

site/                      Flask web application
  app.py                   Web server, 1-click Guest sessions, JSON API
  i18n.py                  English / French translations & plain-language figure guides
  templates/               Simple, clean Jinja2 HTML templates
  static/                  Lightweight CSS & generated scientific figures (fig1–fig9)

train_ai.py                End-to-end supervised + Actor-Critic (A2C) RL training pipeline
run_simulation.py          Scientific studies and validation report generator
server.py                  JSON Lines / HTTP engine bridge for Unity
tests/                     Automated unit, neural-net gradcheck & web integration tests
```

---

## Real Trained Actor-Critic Neural Network (`PolicyValueNet`)

Instead of a hardcoded rule, the in-game AI advisor (`tumeur/neural_net.py`, `tumeur/ai.py`, `train_ai.py`) is a genuine multi-head neural network implemented from scratch in NumPy:

- **Architecture**: `14 public clinical inputs → 64 (tanh) → 64 (tanh)` feeding 3 heads (**5,575 trainable parameters**):
  1. **Discrete Action Policy $\pi_\theta(a \mid s)$**: Masked Softmax over `wait`, `irradiate`, `biopsy`, `po2_imaging`, `antiangiogenic`.
  2. **Continuous Dose Head $\mu_\theta(s)$**: Sigmoid-scaled fraction size in `[0.5, 5.0] Gy`.
  3. **State-Value Critic $V_\theta(s)$**: Expected normalized clinical return.
- **Strict Partial Observability (POMDP)**: Encodes **only** noisy public scans, biomarkers, diagnostic exam history, and organ tolerance (`TD50 / sparing`) — zero access to hidden `truth_state` or exact cell count.
- **Two-Stage Training & Loss Verification**:
  - **Analytical Backpropagation Gradcheck**: Verified against central finite differences to relative error **`5.4 × 10⁻⁶`** (supervised) and **`8.3 × 10⁻⁸`** (A2C).
  - **Stage 1 (Supervised Adam)**: Minimizes $\mathcal{L}_{\text{CE}} + 2.5\mathcal{L}_{\text{dose}} + 0.4\mathcal{L}_{\text{critic}} + \mathcal{L}_{L2}$ on counterfactual PDE rollouts (`86.8%` held-out episode action accuracy).
  - **Stage 2 (On-Policy A2C RL)**: Fine-tuned on live stochastic `TumorGame` episodes with standardized advantages $\hat{A}_t = G_t - V_\theta(s_t)$ and entropy regularization.
  - **Held-Out Clinical Benchmark (`n_grid=160`)**: Scores **`1617.8`** mean clinical points across scenarios — outperforming the legacy heuristic (`1578.4`), aggressive `5 Gy/day` (`1184.2`, lethal on fragile lung), random (`1159.3`), and untreated (`1100.1`).
- **Explainability**: Computes exact input-gradient saliency $\left|\frac{\partial z_{a^*}}{\partial x_i}\right|$ at each turn to highlight which diagnostic signals drove its recommendation.

---

## Quick Start

### 1. Run the Web Platform

```bash
python3 -m pip install -r requirements.txt
python3 site/app.py
# Open http://localhost:5000
```

### 2. Run Automated Tests (Engine + Neural Net Gradcheck + Web App)

```bash
python3 -m unittest discover -s tests -v
python3 check_setup.py
```

### 3. Re-train the Neural Network or Re-run Scientific Validation

```bash
python3 train_ai.py                # Trains PolicyValueNet & generates fig9_ia_training.png
python3 run_simulation.py --validate
```

---

## Two Scientific Regimes

| Regime | Purpose | Evidence Status |
| --- | --- | --- |
| `in_vitro_strict` | EMT6/Ro spheroids in controlled culture | Calibrated & validated against published data (31/32 checks) |
| `vascular_human` | Mechanistic tumor with angiogenesis | Scenario grounded in literature mechanisms; not patient predictions |

## PythonAnywhere Deployment

`wsgi.py` is the WSGI entry point. Set `TUMORLAB_SECRET_KEY` in environment variables and use a single worker. Pre-computed validation checks (`resultats/validation_checks.json`) and trained neural network weights (`tumeur/ai_weights.npz`) load in milliseconds with zero heavy external ML dependencies.
