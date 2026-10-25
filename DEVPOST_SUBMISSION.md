# CodeTheCure 2026 — Guide & Textes prêts à coller sur Devpost

Ce fichier contient **tout ce qu'il te faut** pour mettre à jour ta soumission sur Devpost (catégorie **Detection & Diagnosis**) avant la deadline du **28 octobre 2026**, en répondant directement au feedback du jury !

---

## 1. Abstract court (< 300 mots — Exigence n°1 de Devpost)

> *(Exactement **268 mots** — tu peux le copier-coller tel quel dans le champ "Project Description Abstract" !)*

Cancer treatment never happens with perfect visibility: doctors cannot count individual tumor cells inside a patient. Instead, oncology relies on **detection and diagnosis under uncertainty**—combining noisy size scans, blood biomarkers, oxygen imaging (`pO₂`), and needle biopsies to uncover hidden dangers before prescribing therapy.

**Tumor Lab** is a free, interactive web simulator and clinical strategy game that turns real tumor physics and diagnostic imaging into an accessible, hands-on experience. Built from scratch on a reaction-diffusion partial differential equation (PDE) engine benchmarked against 30 peer-reviewed oncology papers (**31/32 scientific checks passed**), Tumor Lab lets anyone step into the shoes of a clinician monitoring a sub-millimeter tumor.

Following preliminary mentor feedback, we redesigned the onboarding so **users with zero biology or oncology background** can understand and play in one click—while keeping a clean, lightweight interface:
1. **1-Click Guest Access & 5-Step Guided Tutorial**: Walks beginners through reading noisy scans, running an Oxygen Scan (`pO₂`) to detect radiation-resistant *hypoxia*, and performing a Needle Biopsy to catch evolving resistant clones.
2. **Plain-Language Mode & 3-Minute Visual Sandbox**: Translates every medical unit (`µm`, `Gy`, `pO₂`, `NTCP`) into everyday analogies, interactive sliders, a glossary, and `"In plain words"` summaries beneath all 9 scientific figures (in English and French).
3. **Explainable Actor-Critic Neural Network (`PolicyValueNet`)**: A genuine 5,575-parameter neural network trained from scratch with Adam and Advantage Actor-Critic (A2C) reinforcement learning strictly on noisy public diagnostics—displaying live Softmax probabilities and input-gradient saliency to teach *why* diagnostic scans save healthy organs.

*Disclaimer: Educational simulation only, not a medical device.*

---

## 2. Texte complet de la page projet Devpost (Section "Story")

```markdown
## Inspiration
When someone hears terms like *hypoxia*, *biopsy*, *biomarker noise*, or *fractionated radiotherapy*, oncology can feel like an impenetrable black box. Yet behind these words lies a vital **Detection & Diagnosis** challenge: two tumors that look identical on a basic size scan can respond completely differently to treatment because one has an oxygen-starved core (*hypoxia*) or hidden radiation-resistant cells. I built **Tumor Lab** so students, patients' families, and curious learners can visually *experience* how diagnostic tools uncover what basic scans miss.

## How We Acted on Preliminary Judge Feedback
During the preliminary feedback round, mentors highlighted our **factual oncology foundation and scientific figures**, and asked a key question:
> *"For people who want to play around with your app but have no prior knowledge regarding biology/oncology/tumors, how will you support them in understanding how to use your app?"*

We addressed this head-on while keeping the app fast, clean, and simple:
- **Zero-Signup 1-Click Guest Mode**: Anyone can click **Start Guided Tutorial** and play immediately without creating an account.
- **Interactive 5-Step Walkthrough & Beginner Help Mode**: Highlights which diagnostic scan to run first, explains every turn in plain everyday sentences (*"What just happened?"*), draws a clear `400 µm` target ring and `1 mm` scale bar, and labels organ sensitivity in plain words (`Easy` to `Very Fragile`).
- **3-Minute Interactive Guide (`/learn`)**: Uses everyday analogies (like an avocado's layers), a live **Dose × Oxygen Sandbox** powered by real radiobiology equations, a jargon-free glossary, and a 3-question quiz.
- **Plain-Language Figure Takeaways & Bilingual (`EN`/`FR`) Toggle**: Every scientific plot now includes an `"In plain words"` summary box, in both English and French.

## What It Does
1. **Mechanistic Tumor & Diagnostic Engine**: Solves oxygen, glucose, lactate, and pH reaction-diffusion equations (`Ward & King 1997`, `Freyer & Sutherland 1986`), ATP-driven necrosis, Linear-Quadratic + OER radiation response (`Alper & Howard-Flanders 1956`), LKB healthy-organ toxicity (`Emami/Burman 1991`), and Bayesian clonal resistance (`Britten 1996`, `Alfonso & Berk 2019`).
2. **Detection Under Uncertainty (POMDP)**: Players observe noisy imaging diameters (`±5%`) and blood biomarkers (`±20%`), and decide when to invest time in **pO₂ Oxygen Imaging** or **Needle Biopsies**.
3. **Real Actor-Critic Neural Network (`PolicyValueNet`)**: Implemented from scratch in NumPy (`14 → 64 → 64` tanh trunk, 5,575 parameters) with analytical backpropagation verified by finite-difference gradcheck (`error = 5.4e-6`). Trained with mini-batch Adam (Cross-Entropy + Dose MSE + Critic MSE) and on-policy **Advantage Actor-Critic (A2C)** reinforcement learning on noisy public observations only, it achieves **#1 clinical score (1617.8)** on held-out seeds—beating both brute-force radiation and hand-coded heuristics—while explaining its choices via Softmax bars and input-gradient saliency!

## Validation & Honesty
- **31 / 32 scientific checks passed** (9 numerical verification checks + 22/23 experimental benchmarks, cross-validated against a 3D cell-by-cell agent model). The single rich-medium necrosis discrepancy is documented honestly rather than hidden with ad-hoc tuning.
- Explicitly separates validated *in vitro* dynamics from educational vascular scenarios. Never a medical device.
```

---

## 3. Script Vidéo Démo (Max 3 minutes — Chronométré pour gagner !)

- **`0:00 – 0:25` | Hook & Le problème (Detection & Diagnosis)**
  - *À l'écran :* Page d'accueil de Tumor Lab en anglais.
  - *Voix :* "Why can't doctors just blast a tumor once—and why do patients need oxygen scans, blood tests, and biopsies? Because basic scans are noisy and hide what happens inside a tumor. I'm Harold, and I built **Tumor Lab**: an interactive tumor detection and treatment simulator validated against 30 oncology papers—designed so anyone, even with zero biology background, can play and understand."
- **`0:25 – 1:15` | Réponse directe au feedback : Expérience Débutant en 1 clic**
  - *À l'écran :* Clique sur **`Learn the Basics in 3 Min`**, bouge les deux curseurs du **Interactive Sandbox** (baisse l'oxygène `pO₂` à 2 mmHg pour montrer que les cellules survivent aux rayons), puis clique sur **`Start Guided Tutorial (No Account Needed)`**.
  - *Voix :* "With zero medical background and no login required, beginners can test how low oxygen blocks radiation in 5 seconds, then jump straight into a 5-step Guided Tutorial."
- **`1:15 – 2:10` | Boucle Détection → Diagnostic → Traitement + Vrai Réseau de Neurones**
  - *À l'écran :* Dans la partie :
    1. Clique sur **`Oxygen Scan (pO₂ Imaging)`** (Étape 1 du guide) → montre le rapport d'hypoxie et l'encadré *"What just happened?"* en langage courant.
    2. Irradie à **`2.0 Gy`** → montre le cercle vert qui se rapproche du cercle pointillé cible `400 µm` sans faire monter la toxicité.
    3. Fais une **`Needle Biopsy`**, puis clique sur **`Ask Neural Net`** → montre les barres de probabilités Softmax et la valeur du Critique $V(s)$.
  - *Voix :* "Instead of guessing, we run an Oxygen Scan and a Needle Biopsy to diagnose hidden hypoxia and clonal resistance. And our AI advisor isn't a fake rule: it's a real 5,575-parameter Actor-Critic Neural Network trained from scratch with Adam and Reinforcement Learning strictly on noisy public scans, showing live Softmax probabilities and gradient saliency."
- **`2:10 – 2:55` | Crédibilité scientifique (31/32 checks + Loss Curve Fig. 9)**
  - *À l'écran :* Ouvre l'onglet **`Validation (31/32)`** (qui charge instantanément) puis remonte sur **`Fig. 9 — Actor-Critic Neural Network`** sur l'accueil + montre le switch `FR / EN` et le bouton `Dark`.
  - *Voix :* "Under the simple interface, 31 out of 32 numerical and experimental checks pass against published data, and Figure 9 shows our neural network's train/validation loss convergence and #1 benchmark score—explained in plain words in both English and French. Thank you!"

---

## 4. Commandes pour mettre à jour ton GitHub et PythonAnywhere

Depuis ton dossier local `tumor-lab` (ou après avoir récupéré ces fichiers modifiés) :

```bash
# 1. Vérifier que les 13 tests automatiques passent
python3 -m unittest discover -s tests -v

# 2. Pousser sur GitHub
git add README.md DEVPOST_SUBMISSION.md train_ai.py \
        tumeur/neural_net.py tumeur/ai.py tumeur/ai_weights.npz tumeur/game.py \
        resultats/validation_checks.json resultats/ai_training_metrics.json resultats/fig9_ia_training.png \
        site/app.py site/i18n.py site/static/style.css site/static/results/fig9_ia_training.png \
        site/templates/ tests/test_game.py
git commit -m "Add beginner onboarding, EN/FR support, and real trained Actor-Critic neural network"
git push origin main
```

Sur **PythonAnywhere** (dans une console Bash PythonAnywhere) :
```bash
cd ~/tumor-lab
git pull origin main
```
Puis va dans l'onglet **Web** de PythonAnywhere et clique sur le bouton vert **Reload** ! *(Aucune librairie lourde à installer : tout le réseau de neurones tourne en `numpy` pur et pèse moins de 60 Ko !)*
