# CodeTheCure 2026 — Dossier complet prêt à coller sur Devpost

Voici le contenu exact, champ par champ, à copier-coller sur ta page de soumission **Devpost** pour le **2026 International CodeTheCure Hackathon** (catégorie **Detection & Diagnosis**). Tous les textes à coller sur Devpost sont rédigés en anglais clair, sobre et rigoureux pour le jury.

---

## 1. Champs rapides Devpost

- **Project Name :** `Tumor Lab`
- **Elevator Pitch / Tagline (max 140 caractères) :**
  `Interactive tumor detection, noisy diagnostics & explainable Actor-Critic AI simulator validated against 30 oncology papers.`
- **Track / Category :** `Detection & Diagnosis`
- **Built With (tags à ajouter) :**
  `python`, `numpy`, `scipy`, `matplotlib`, `flask`, `sqlite`, `javascript`, `html5`, `css3`, `neural-networks`, `reinforcement-learning`, `computational-biology`

---

## 2. Project Description Abstract (Exigence officielle : MAX 300 mots)

*(Exactement **269 mots** — prêt à copier-coller tel quel dans le champ d'abstract court)*

```text
Doctors never treat cancer with perfect visibility: basic size scans carry measurement noise and cannot reveal whether a tumor hides an oxygen-starved core (hypoxia) or evolving radiation-resistant cells. Effective care starts with detection and diagnosis under uncertainty—knowing when to order oxygen imaging, blood biomarkers, or needle biopsies before prescribing therapy.

Tumor Lab is a free, bilingual (EN/FR) interactive web simulator and turn-based clinical strategy game built on a reaction-diffusion partial differential equation (PDE) engine benchmarked against 30 peer-reviewed oncology papers (31/32 validation checks passed). Players monitor a sub-millimeter tumor through realistic, noisy clinical diagnostics: morphological scans (±5% error), serum biomarkers (±20% noise), quantitative pO2 hypoxia imaging, and 60-cell needle biopsies, balancing tumor eradication (<400 µm) against healthy-organ toxicity (Lyman-Kutcher-Burman NTCP).

To support complete beginners with zero biology background, Tumor Lab provides 1-click Guest access, a 5-step Guided Tutorial centered on diagnostic decision-making, an interactive Dose-vs-Oxygen radiobiology sandbox, plain-language translations on every metric, and accessible summaries beneath all nine scientific figures.

At its core, Tumor Lab features PolicyValueNet, a genuine 5,575-parameter Actor-Critic neural network implemented from scratch in NumPy (14 public clinical inputs -> 64 -> 64 tanh -> Softmax action head, continuous Gy dose head, and state-value Critic). Verified by finite-difference gradient checking (5.4e-6 relative error), it is trained via mini-batch Adam (Cross-Entropy + Dose MSE + Value MSE) and Advantage Actor-Critic (A2C) reinforcement learning strictly on noisy public diagnostics—never hidden ground truth. On held-out clinical benchmarks, PolicyValueNet ranks #1 (1617.8 mean score), outperforming hand-coded rules and avoiding lethal lung toxicity while explaining every recommendation through live Softmax probabilities and input-gradient saliency. Educational simulation only; not a medical device.
```

---

## 3. Page complète du projet Devpost (Section Markdown "Story")

*(À copier-coller directement dans le grand éditeur Markdown de Devpost)*

```markdown
## Inspiration: Why Detection and Diagnosis Come Before Treatment
In real oncology, clinicians never get to see an exact cell-by-cell count inside a patient. Two tumors that appear identical on a basic size scan can respond completely differently to radiotherapy:
- One may hide an oxygen-starved interior (**hypoxia**, `pO2 < 10 mmHg`), where the absence of oxygen prevents radiation from fixing DNA damage—making central tumor cells up to **3x more resistant** (`Alper & Howard-Flanders, 1956`).
- Another may harbor expanding **radioresistant subclones** selected by overly aggressive doses (`Britten et al., 1996`; `Zhang et al., 2017`), or sit next to a fragile organ such as the lung (`TD50 = 24.5 Gy`), where blind high-dose radiation triggers lethal toxicity before shrinking the tumor.

I built **Tumor Lab** to turn this hidden clinical reality into an interactive, transparent simulation: users step into the role of a clinician who must combine **noisy scans, blood biomarkers, oxygen imaging, and biopsies** to understand a tumor before deciding how—and whether—to treat it.

---

## Supporting Beginners with Zero Biology Background (Acting on Mentor Feedback)
During preliminary feedback, mentors praised our factual biological modeling and scientific plots, and asked how we could support visitors with **zero prior biology or oncology background**. We redesigned the user journey on two levels—accessible in seconds for newcomers, while keeping every differential equation inspectable:

- **Zero-Friction 1-Click Guest Access**: Visitors launch the simulator immediately without creating an account.
- **5-Step Guided Diagnostic Tutorial**: Walks first-time players step by step through ordering an **Oxygen Scan (`pO2`)**, reading hypoxia reports, delivering a safe `2.0 Gy` fraction, running a **Needle Biopsy**, and consulting the neural network advisor.
- **Plain-Language Mode (`ON` by default)**: Adds clear everyday explanations under technical metrics (`µm`, `Gy`, `pO2`, `NTCP`), displays a `400 µm` target ring and `1 mm` scale bar on the live cross-section, translates organ tolerance into plain risk levels (`Resilient` to `Very Fragile`), and explains *"What just happened?"* after every turn.
- **3-Minute Interactive Guide (`/learn`)**: Features everyday visual analogies, a live **Dose x Oxygen Radiobiology Sandbox** driven by the Linear-Quadratic + OER equations, a jargon-free glossary, and a 3-question self-check quiz.
- **Plain-Language Figure Summaries & Bilingual (`EN`/`FR`) Interface**: All 9 scientific figures include an *"In plain words"* takeaway box, switchable in one click between English and French.

---

## Diagnostic Tools & Partially Observable Clinical Environment (POMDP)
Tumor Lab formulates clinical decision-making as a **Partially Observable Markov Decision Process (POMDP)**. The underlying PDE engine computes exact spatial concentrations and cell populations (`truth_state`), **but never exposes them to the player or the AI**. Instead, decisions rely on four realistic diagnostic channels:

1. **Morphological Size Scan (`±5%` Gaussian measurement noise)**: Estimates outer tumor diameter (`µm`) at every turn relative to the `< 400 µm` durable control target, without revealing internal necrosis or clonal makeup.
2. **Serum Biomarker (`±20%` log-normal noise)**: A blood-test proxy (analogous to PSA or ctDNA) that tracks whether the hidden viable cell burden is rising or falling across turns despite single-measurement fluctuations.
3. **Quantitative Oxygen Imaging (`pO2` Scan, costs 1.5 days)**: Maps central oxygen tension (`mmHg`) and the **hypoxic fraction (%)** of the tumor. Detecting severe hypoxia alerts the user that high single doses will waste healthy-organ tolerance unless fractionated sessions allow tumor shrinkage and **reoxygenation** (or vascular normalization in the vascular scenario).
4. **Needle Biopsy (`n = 60` sampled cells, costs 2.0 days)**: Draws a stochastic binomial sample from the tumor's internal `ResistanceDistribution` (`Alfonso & Berk, 2019`) to estimate the percentage of surviving **radioresistant clones** (`alpha_factor < 0.75`) and average radiosensitivity, preventing blind dose escalation.

---

## Real Trained Actor-Critic Neural Network (`PolicyValueNet`)
Rather than faking clinical AI with hardcoded `if/else` rules, we built and trained a genuine multi-head **Actor-Critic neural network** from scratch in pure NumPy (`tumeur/neural_net.py`, `tumeur/ai.py`, `train_ai.py`):

### 1. Network Architecture (5,575 Trainable Parameters)
`PolicyValueNet` takes a **14-dimensional normalized vector of strictly public clinical observations** (`x in R^14`):
- Noisy diameter ratio, recent diameter velocity, noisy biomarker log-burden, normalized clinical day, cumulative delivered dose (`Gy`), current LKB `NTCP` toxicity, organ radiosensitivity (`TD50 / sparing_factor`), time elapsed since last `pO2` scan, measured hypoxic fraction and central `pO2`, time elapsed since last biopsy, measured resistant clone fraction, vascular regime indicator, and measured vascular activity.

These 14 diagnostic inputs pass through two dense hidden layers (`14 -> 64 (tanh) -> 64 (tanh)`) into **three output heads**:
- **Discrete Policy Head `pi_theta(a | s)`**: Masked Softmax over the 5 clinical actions (`po2_imaging`, `biopsy`, `irradiate`, `antiangiogenic`, `wait`), automatically masking `antiangiogenic` in avascular modes.
- **Continuous Dose Head `mu_theta(s)`**: Sigmoid output scaled to `[0.5, 5.0] Gy` per fraction.
- **State-Value Critic `V_theta(s)`**: Linear head predicting expected normalized clinical return from the current diagnostic state.

### 2. Analytical Backpropagation & Two-Stage Training
- **Finite-Difference Gradient Verification**: Before training, analytical backpropagation through all three heads and action masks is verified against central finite differences (`epsilon = 1e-5`), achieving **`5.44e-6` max relative error** on the supervised loss and **`8.29e-8`** on the A2C reinforcement learning loss.
- **Stage 1 — Counterfactual Supervised Pre-Training (140 Epochs, Mini-Batch Adam)**: Across 653 clinical states generated by counterfactual PDE rollouts (episode-split into 403 training / 250 held-out validation states), Adam (`lr = 2.5e-3`, cosine decay) minimizes a composite objective:
  `L = L_CE(pi_theta) + 2.5 * L_MSE(mu_theta) + 0.4 * L_MSE(V_theta) + 1e-4 * ||theta||_2^2`
  Training loss drops from `0.997` to `0.192` (`92.3%` accuracy), while held-out episode validation loss converges from `0.937` to **`0.466`** (**`86.8%` unseen-episode action accuracy**).
- **Stage 2 — On-Policy Advantage Actor-Critic (A2C) Reinforcement Learning**: Fine-tuned directly on live stochastic `TumorGame` trajectories using standardized advantages `A_t = G_t - V_theta(s_t)` and policy entropy regularization (`-0.015 H(pi_theta)`), teaching the network when ordering a diagnostic `pO2` scan or biopsy yields higher long-term survival than immediate irradiation.

### 3. Held-Out Clinical Benchmark & Input-Gradient Explainability
Evaluated on unseen random seeds across all four clinical scenarios at full PDE resolution (`n_grid = 160`):
- **`PolicyValueNet` (Trained Neural Net)**: **1617.8 mean score** (**Rank #1**, beating every baseline in every scenario).
- **Hand-Crafted Clinical Heuristic**: `1578.4` mean score.
- **Aggressive `5.0 Gy/day` Protocol**: `1184.2` mean score (causes **`-802.2` lethal pulmonary toxicity** in the Critical Organ lung scenario).
- **Random Policy**: `1159.3` mean score (`-847.8` on lung).
- **Untreated (`wait`)**: `1100.1` mean score (`25%` control).

Every time a user clicks **Ask Neural Net**, the interface displays the network's live **Softmax probability bars**, continuous **Gy dose output**, **Critic value `V(s)`**, and **input-gradient saliency (`|d z_a* / d x_i|`)**—highlighting in plain language *which* diagnostic signals (e.g., unmeasured `pO2`, rising resistant clones on biopsy, or tight lung `TD50` margin) drove its decision.

---

## Mechanistic Biophysical Engine & Scientific Validation (31 / 32 Checks)
Underneath the web interface, Tumor Lab solves coupled biophysical equations calibrated from 30 peer-reviewed publications (`Freyer & Sutherland 1986`, `Mueller-Klieser 1982`, `Ward & King 1997`, `Bertuzzi 2010`, `Bozic 2010`, `Williams 2016`, `Emami & Burman 1991`):
- **Conservative finite-volume reaction-diffusion PDEs** for oxygen, glucose, lactate, and extracellular `pH`, coupled to free-boundary tumor growth (`SSP-RK2` + `van Leer` TVD advection) and ATP-threshold necrosis.
- **Cross-validation against a 3D center-based agent model** (`Drasdo & Hohme 2005`) and neutral `1/f` subclonal evolution (`R^2 = 0.988`).
- **31 out of 32 automated verification & experimental validation checks passed** (9/9 numerical solver checks + 22/23 published lab benchmarks). The single discrepancy—rich-medium necrosis onset (`627 µm` vs `516 µm`, `Casciari 1992`)—is kept and documented transparently without ad-hoc parameter fudging.
- **Clear scientific boundaries**: Strictly separates experimentally validated avascular spheroid physics (`in_vitro_strict`) from educational human vascular scenarios (`vascular_human`).

*Medical Disclaimer: Tumor Lab is an educational and theoretical simulation platform. It is not a medical device and must never be used for clinical diagnosis or patient treatment decisions.*
```

---

## 4. Captures d'écran & visuels recommandés à ajouter dans la galerie Devpost

Devpost permet d'uploader plusieurs images en haut de ta page projet. Pour maximiser ta note auprès du jury, mets dans cet ordre :
1. **Capture de la page de jeu (`/play/tutoriel` ou `/play/organe_critique`)** montrant à la fois la coupe de la tumeur (avec l'anneau cible `400 µm`), les résultats d'un **Oxygen Scan (`pO2`)** + **Needle Biopsy**, et le panneau **Trained Actor-Critic Neural Network** ouvert avec les barres de probabilités Softmax.
2. **La figure `resultats/fig9_ia_training.png`** (les 4 panneaux de preuve du réseau de neurones : descente de la Train/Validation Loss, précision hors-échantillon `86.8%`, récompense RL A2C + gradcheck `5.4e-6`, et Benchmark #1 `1617.8`).
3. **Capture de la page `/learn`** montrant le **Mini-laboratoire interactif (Dose x Oxygène)** et le glossaire débutant sans jargon.
4. **Capture de la page `/tests` (`Validation 31/32`)** montrant le tableau `PASS` du réseau de neurones + les 31/32 vérifications scientifiques.
