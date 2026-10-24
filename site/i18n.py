"""
Traductions simples Anglais / Français pour Tumor Lab.
L'anglais est la langue par défaut pour le jury international (CodeTheCure),
et un bouton FR / EN dans la barre de navigation permet de changer en 1 clic.
"""

SCENARIOS_I18N = {
    "en": {
        "tutoriel": {
            "name": "Guided Tutorial",
            "badge": "🟢 Start Here · Beginner",
            "desc": "Step-by-step interactive walkthrough. Zero biology knowledge needed!",
            "icon": "🎓",
        },
        "standard": {
            "name": "Standard Clinic",
            "badge": "🟡 Intermediate",
            "desc": "Diagnose hypoxia & resistance while managing a 40 Gy radiation budget.",
            "icon": "⚔️",
        },
        "organe_critique": {
            "name": "Fragile Organ",
            "badge": "🟠 Hard",
            "desc": "Surrounding lung tissue is very sensitive. Aggressive doses will fail.",
            "icon": "💔",
        },
        "vascular_human": {
            "name": "Vascular Tumor",
            "badge": "🔵 Clinical Scenario",
            "desc": "Tumor recruits abnormal blood vessels. Combines imaging, radiation & anti-angiogenics.",
            "icon": "🩸",
        },
        "in_vitro_strict": {
            "name": "Petri Dish Lab",
            "badge": "🔬 Expert Sandbox",
            "desc": "Strict EMT6/Ro lab cells doubling every 20h. Standard hospital doses won't stop it!",
            "icon": "🔬",
        },
    },
    "fr": {
        "tutoriel": {
            "name": "Tutoriel guidé",
            "badge": "🟢 Débutant · Commencez ici",
            "desc": "Guide interactif pas-à-pas. Aucune connaissance en biologie requise !",
            "icon": "🎓",
        },
        "standard": {
            "name": "Clinique Standard",
            "badge": "🟡 Intermédiaire",
            "desc": "Diagnostiquez l'hypoxie et la résistance avec un budget limité à 40 Gy.",
            "icon": "⚔️",
        },
        "organe_critique": {
            "name": "Organe fragile",
            "badge": "🟠 Difficile",
            "desc": "L'organe voisin est très sensible aux rayons. Aucune stratégie brutale ne passe.",
            "icon": "💔",
        },
        "vascular_human": {
            "name": "Tumeur vascularisée",
            "badge": "🔵 Scénario clinique",
            "desc": "La tumeur crée ses propres vaisseaux sanguins. Imagerie + rayons + anti-angiogénique.",
            "icon": "🩸",
        },
        "in_vitro_strict": {
            "name": "Boîte de Petri (Labo)",
            "badge": "🔬 Bac à sable expert",
            "desc": "Culture EMT6/Ro stricte (double en 20h). Montre pourquoi le labo diffère d'un patient.",
            "icon": "🔬",
        },
    },
}


FIGURES_I18N = {
    "en": [
        {
            "img": "results/fig1_croissance.png",
            "title": "Fig. 1 — Avascular Tumor Growth",
            "tech": "Diameter kinetics, cell populations and central microenvironment of an EMT6/Ro spheroid.",
            "plain": "💡 In plain words: A tiny cluster of cancer cells grows fast at first, then slows down as cells in the middle run out of food and oxygen.",
        },
        {
            "img": "results/fig2_profils_radiaux.png",
            "title": "Fig. 2 — Oxygen & Nutrient Gradients",
            "tech": "Spatial distribution of oxygen, glucose, pH and viability from center to outer rim.",
            "plain": "💡 In plain words: Outer cells eat most of the oxygen and sugar first, leaving the center suffocating (hypoxic) and acidic.",
        },
        {
            "img": "results/fig3_coupe_spheroide.png",
            "title": "Fig. 3 — Inside Cross-Section",
            "tech": "2D view of internal structure: dividing outer rim, hypoxic ring and central necrosis.",
            "plain": "💡 In plain words: Like cutting an avocado in half — a living outer layer surrounds a suffocating middle ring and a dead core.",
        },
        {
            "img": "results/fig4_milieux.png",
            "title": "Fig. 4 — Effect of Oxygen & Sugar Supply",
            "tech": "Reproduction of Freyer & Sutherland (1986): O₂ and glucose supply control saturation size.",
            "plain": "💡 In plain words: Less oxygen and sugar around the tumor forces it to stop growing at a much smaller size.",
        },
        {
            "img": "results/fig5_radiotherapie.png",
            "title": "Fig. 5 — Fractionated Radiotherapy",
            "tech": "Control vs. irradiated tumor: central hypoxia induces measurable radioresistance.",
            "plain": "💡 In plain words: Radiation kills well-oxygenated outer cells easily, while oxygen-poor central cells survive up to 3× better.",
        },
        {
            "img": "results/fig6_modele_agents.png",
            "title": "Fig. 6 — 3D Cell-by-Cell Simulation",
            "tech": "3D agent-based model used to cross-validate the fast continuous equations.",
            "plain": "💡 In plain words: We double-checked our fast game math against a slower 3D simulation tracking thousands of individual cells.",
        },
        {
            "img": "results/fig7_validation.png",
            "title": "Fig. 7 — Benchmark vs. Real Lab Data",
            "tech": "31/32 checks passed against published measurements; the single rich-medium necrosis gap is documented.",
            "plain": "💡 In plain words: 31 out of 32 predictions match real cancer lab experiments published by scientists.",
        },
        {
            "img": "results/fig8_evolution.png",
            "title": "Fig. 8 — Darwinian Evolution & Resistance",
            "tech": "Mutation, drift and selection — allelic spectrum validated against the 1/f power law (R² ≈ 0.99).",
            "plain": "💡 In plain words: Just like bacteria evolving against antibiotics, hitting a tumor can leave behind tougher, radiation-resistant cells.",
        },
    ],
    "fr": [
        {
            "img": "results/fig1_croissance.png",
            "title": "Fig. 1 — Croissance avasculaire",
            "tech": "Cinétique du diamètre, populations cellulaires et microenvironnement central d'un sphéroïde EMT6/Ro.",
            "plain": "💡 En clair : Un petit amas de cellules grossit vite au début, puis ralentit quand le centre manque d'oxygène et de sucre.",
        },
        {
            "img": "results/fig2_profils_radiaux.png",
            "title": "Fig. 2 — Profils d'oxygène et nutriments",
            "tech": "Distribution spatiale de l'oxygène, du glucose, du pH et de la viabilité du centre vers le bord.",
            "plain": "💡 En clair : Les cellules du bord consomment l'oxygène en premier ; le centre étouffe (hypoxie) et devient acide.",
        },
        {
            "img": "results/fig3_coupe_spheroide.png",
            "title": "Fig. 3 — Coupe interne de la tumeur",
            "tech": "Visualisation 2D de la structure interne : couronne viable, cœur hypoxique et nécrose centrale.",
            "plain": "💡 En clair : Comme un fruit coupé en deux : une couche extérieure vivante entoure une zone qui manque d'air et un cœur inerte.",
        },
        {
            "img": "results/fig4_milieux.png",
            "title": "Fig. 4 — Effet des nutriments",
            "tech": "Reproduction du protocole Freyer & Sutherland (1986) : l'apport en O₂ et glucose contrôle la taille finale.",
            "plain": "💡 En clair : Moins il y a d'oxygène et de sucre disponibles, plus la tumeur s'arrête de grossir tôt.",
        },
        {
            "img": "results/fig5_radiotherapie.png",
            "title": "Fig. 5 — Radiothérapie fractionnée",
            "tech": "Comparaison témoin / irradié : l'hypoxie induit une radiorésistance mesurable dans le cœur tumoral.",
            "plain": "💡 En clair : Les rayons détruisent bien le bord oxygéné, mais les cellules du centre sans oxygène résistent jusqu'à 3× mieux.",
        },
        {
            "img": "results/fig6_modele_agents.png",
            "title": "Fig. 6 — Modèle cellule par cellule 3D",
            "tech": "Simulation cellule par cellule en 3D, validation croisée du modèle continu rapide.",
            "plain": "💡 En clair : Les calculs rapides du jeu ont été vérifiés face à une simulation 3D qui suit chaque cellule individuellement.",
        },
        {
            "img": "results/fig7_validation.png",
            "title": "Fig. 7 — Validation face aux vraies expériences",
            "tech": "31/32 checks réussis contre les données publiées. L'écart connu sur la nécrose en milieu riche est documenté.",
            "plain": "💡 En clair : 31 prédictions sur 32 tombent dans les mesures réelles publiées par des chercheurs en laboratoire.",
        },
        {
            "img": "results/fig8_evolution.png",
            "title": "Fig. 8 — Évolution et résistance",
            "tech": "Mutation, dérive et sélection — spectre allélique validé par la loi 1/f (R² ≈ 0.99).",
            "plain": "💡 En clair : Comme les bactéries face aux antibiotiques, éliminer les cellules fragiles peut laisser la place aux cellules résistantes.",
        },
    ],
}


CHECK_NAMES_EN = {
    "Solution analytique (profil parabolique d'O₂)": "Analytical solution (parabolic O₂ profile)",
    "Ordre de convergence en maillage": "Spatial mesh convergence order",
    "Conservation de la masse (flux O₂ = consommation)": "Mass conservation (surface O₂ flux = consumption)",
    "Exactitude du jacobien analytique": "Analytical Jacobian accuracy",
    "Séparation d'échelles (τ_diffusion / τ_croissance)": "Timescale separation (τ_diffusion / τ_growth)",
    "Positivité des concentrations": "Positivity of chemical concentrations",
    "Indépendance au pas de temps": "Time-step independence",
    "Validation croisée continu / agents (rayon)": "Cross-validation: continuous PDE vs. 3D agent model",
    "Reproductibilité à graine fixée (et divergence sinon)": "Deterministic seed reproducibility",
    "Limite de diffusion de l'O₂ (vs Grimes 2014 : 233±22 µm)": "O₂ diffusion limit (vs. Grimes 2014: 233±22 µm)",
    "OCR volumique (vs 22-27 mmHg/s publiés)": "Volumetric O₂ consumption (vs. published 22–27 mmHg/s)",
    "Couronne viable ~constante aux grands diamètres": "Constant viable outer rim thickness at large sizes",
    "OER borné entre 1 (anoxie) et 3 (normoxie)": "Oxygen Enhancement Ratio bounded in [1, 3]",
    "Radiorésistance hypoxique (dose isoeffet anoxie/normoxie)": "Hypoxic radioresistance (~3× dose needed without O₂)",
    "Survie d'une lignée mutante (vs théorie de Bozic 2010)": "Mutant clone survival (vs. Bozic 2010 theory)",
    "Spectre neutre en 1/f (vs Williams et al. 2016)": "Neutral evolution 1/f power-law spectrum (Williams 2016)",
    "Expansion subclonale sous sélection (contrôle positif)": "Subclonal expansion under positive selection",
    "Gradient spatial de pression sélective (périphérie/coeur)": "Spatial selective gradient (rim vs. hypoxic core)",
    "Croissance gompertzienne (R² de l'ajustement)": "Gompertzian growth curve fit (R²)",
    "Identifiabilité (conditionnement de la sensibilité)": "Parameter identifiability & sensitivity conditioning",
    "Incertitude propagée recouvrant l'expérience": "Monte-Carlo uncertainty covering experimental bars",
    "Fractionnement : hypofractionné plus toxique (α/β=3 Gy)": "Fractionation: large doses hurt healthy tissue more (α/β=3 Gy)",
    "Tolérance clinique d'un schéma standard (30×2 Gy)": "Clinical safety of standard 30×2 Gy protocol (LKB NTCP)",
    "Sélection de la radiorésistance sous fractionnement": "Selection of radioresistance under repeated irradiation",
    "Couronne viable [16.5 mM glc / 0.28 mM O₂]": "Viable rim thickness [Rich medium: 16.5 mM glc / 0.28 mM O₂]",
    "Diamètre d'apparition de la nécrose [16.5 mM glc / 0.28 mM O₂]": "Necrosis onset diameter [Rich medium: 16.5 mM glc / 0.28 mM O₂]",
    "Temps de doublement initial [16.5 mM glc / 0.28 mM O₂]": "Initial doubling time [Rich medium: 16.5 mM glc / 0.28 mM O₂]",
    "Couronne viable [0.8 mM glc / 0.07 mM O₂]": "Viable rim thickness [Poor medium: 0.8 mM glc / 0.07 mM O₂]",
    "Diamètre d'apparition de la nécrose [0.8 mM glc / 0.07 mM O₂]": "Necrosis onset diameter [Poor medium: 0.8 mM glc / 0.07 mM O₂]",
    "Temps de doublement initial [0.8 mM glc / 0.07 mM O₂]": "Initial doubling time [Poor medium: 0.8 mM glc / 0.07 mM O₂]",
    "pO₂ central (Ø 400 µm) [MK82]": "Central oxygen pO₂ at Ø 400 µm (vs. Mueller-Klieser 1982)",
    "pO₂ central (Ø 1000 µm) [MK82]": "Central oxygen pO₂ at Ø 1000 µm (vs. Mueller-Klieser 1982)",
}
