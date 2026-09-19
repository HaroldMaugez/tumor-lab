"""
Vérification numérique et validation expérimentale.

Distinction méthodologique (Roache, "Verification and Validation in
Computational Science and Engineering", 1998) :

    VÉRIFICATION : « résout-on correctement les équations ? »
        -> comparaison à des solutions analytiques, convergence en maillage,
           conservation, exactitude du jacobien, indépendance au pas de temps.

    VALIDATION   : « résout-on les bonnes équations ? »
        -> comparaison aux mesures expérimentales indépendantes (EMT6/Ro).

Chaque test renvoie un enregistrement structuré (nom, valeur, critère, statut)
de façon à produire un rapport reproductible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from . import benchmarks as bm
from . import metabolism as met
from .parameters import Parameters
from .transport import RadialGrid, TransportSolver
from .units import conc_to_po2, po2_to_conc


@dataclass
class Check:
    name: str
    category: str          # "vérification" | "validation"
    value: float
    criterion: float
    passed: bool
    unit: str = ""
    detail: str = ""

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return (f"[{flag}] {self.name}: {self.value:.4g} {self.unit} "
                f"(critère {self.criterion:.3g}) {self.detail}")


# ---------------------------------------------------------------------------
# 1. VÉRIFICATION — solution analytique de la diffusion-réaction
# ---------------------------------------------------------------------------
def analytic_oxygen_profile(r: np.ndarray, R: float, D: float, q: float,
                            c_bulk: float) -> np.ndarray:
    """Solution analytique pour une consommation d'ordre zéro (sans nécrose).

    D (1/r^2) d/dr (r^2 dc/dr) = q   =>   c(r) = c_bulk - (q/6D)(R^2 - r^2)
    """
    return c_bulk - (q / (6.0 * D)) * (R ** 2 - r ** 2)


def check_analytic_solution(tol: float = 2e-3) -> Check:
    """Compare le solveur à la solution analytique en régime d'ordre zéro.

    On se place à saturation (Km -> 0) et sans effet Pasteur ni loi de Pirt,
    régime dans lequel la consommation est constante : la solution exacte est
    alors parabolique.
    """
    p = Parameters(km_o2=1e-9, km_glc=1e-9, pasteur_factor=1.0,
                   f_maint_uptake=1.0, c_o2_bulk=0.28, c_glc_bulk=16.5)
    R = 120.0            # assez petit pour rester loin de l'anoxie
    errs = []
    for n in (80, 160, 320):
        grid = RadialGrid(n, R)
        sol = TransportSolver(p).solve(grid, np.ones(n))
        exact = analytic_oxygen_profile(grid.r, R, p.D_o2, p.q_o2_max_vol,
                                        p.c_o2_bulk)
        err = np.max(np.abs(sol.c_o2 - exact)) / p.c_o2_bulk
        errs.append(err)
    return Check("Solution analytique (profil parabolique d'O₂)", "vérification",
                 float(errs[-1]), tol, errs[-1] < tol, "erreur relative L∞",
                 detail=f"erreurs {['%.2e' % e for e in errs]}")


def check_mesh_convergence(min_order: float = 1.6) -> Check:
    """Ordre de convergence spatial observé (extrapolation de Richardson)."""
    p = Parameters()
    R = 300.0
    ns = [50, 100, 200, 400]
    vals = []
    for n in ns:
        grid = RadialGrid(n, R)
        sol = TransportSolver(p).solve(grid, np.ones(n))
        vals.append(sol.c_o2[0])
    ref = vals[-1]
    e = [abs(v - ref) for v in vals[:-1]]
    orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1) if e[i + 1] > 0]
    order = float(np.mean(orders)) if orders else 0.0
    return Check("Ordre de convergence en maillage", "vérification",
                 order, min_order, order >= min_order, "ordre",
                 detail=f"ordres observés {[round(o,2) for o in orders]}")


def check_mass_conservation(tol: float = 1e-8) -> Check:
    """Le flux d'O2 entrant doit égaler la consommation totale."""
    p = Parameters()
    worst = 0.0
    for R in (100.0, 300.0, 600.0):
        grid = RadialGrid(200, R)
        solver = TransportSolver(p)
        phi = np.ones(200)
        sol = solver.solve(grid, phi)
        influx, consumed = solver.oxygen_flux_balance(grid, sol, phi)
        worst = max(worst, abs(influx - consumed) / max(abs(influx), 1e-30))
    return Check("Conservation de la masse (flux O₂ = consommation)",
                 "vérification", float(worst), tol, worst < tol,
                 "erreur relative")


def check_jacobian(tol: float = 1e-6) -> Check:
    """Exactitude du jacobien analytique (différences finies centrées)."""
    p = Parameters()
    rng = np.random.default_rng(42)
    c_o2 = rng.uniform(1e-4, 0.3, 12)
    c_glc = rng.uniform(1e-3, 16.5, 12)
    h = 1e-8
    worst = 0.0
    pairs = [(met.oxygen_uptake, met.d_oxygen_uptake_do2, met.d_oxygen_uptake_dglc),
             (met.glucose_uptake, met.d_glucose_uptake_do2, met.d_glucose_uptake_dglc)]
    for f, dfo, dfg in pairs:
        num_o = (f(c_o2 + h, c_glc, p) - f(c_o2 - h, c_glc, p)) / (2 * h)
        num_g = (f(c_o2, c_glc + h, p) - f(c_o2, c_glc - h, p)) / (2 * h)
        worst = max(worst,
                    float(np.max(np.abs(num_o - dfo(c_o2, c_glc, p))
                                 / (np.abs(num_o) + 1e-12))),
                    float(np.max(np.abs(num_g - dfg(c_o2, c_glc, p))
                                 / (np.abs(num_g) + 1e-12))))
    return Check("Exactitude du jacobien analytique", "vérification",
                 worst, tol, worst < tol, "erreur relative")


def check_quasi_steady_state(tol: float = 1e-3) -> Check:
    """Vérifie la séparation d'échelles justifiant l'hypothèse quasi-stationnaire."""
    p = Parameters()
    R = 500.0
    tau_diff = R ** 2 / (6.0 * p.D_o2)                  # s
    tau_growth = np.log(2.0) / p.k_prolif_max           # s
    ratio = tau_diff / tau_growth
    return Check("Séparation d'échelles (τ_diffusion / τ_croissance)",
                 "vérification", float(ratio), tol, ratio < tol, "-",
                 detail=f"τ_diff={tau_diff:.1f} s, τ_croiss={tau_growth/3600:.1f} h")


def check_time_step_independence(tol: float = 0.02) -> Check:
    """Indépendance de la solution au pas de temps maximal."""
    from .growth import SpheroidSimulation
    res = []
    for dt in (1800.0, 600.0, 200.0):
        sim = SpheroidSimulation(Parameters(), R0=60.0, n_grid=80)
        sim.run(6.0, dt_max=dt, record_every_h=48.0)
        res.append(sim.R)
    rel = abs(res[-1] - res[0]) / res[-1]
    return Check("Indépendance au pas de temps", "vérification",
                 float(rel), tol, rel < tol, "écart relatif",
                 detail=f"R(dt=1800s)={res[0]:.1f}, R(dt=200s)={res[-1]:.1f} µm")


def check_positivity() -> Check:
    """Les concentrations restent positives même en régime sévèrement déplété."""
    p = Parameters(c_o2_bulk=0.05, c_glc_bulk=0.8)
    grid = RadialGrid(200, 700.0)
    sol = TransportSolver(p).solve(grid, np.ones(200))
    worst = float(min(sol.c_o2.min(), sol.c_glc.min(), sol.c_lac.min()))
    return Check("Positivité des concentrations", "vérification",
                 worst, 0.0, worst >= -1e-12, "mM")


def check_diffusion_limit(tol: float = 0.25) -> Check:
    """Limite de diffusion de l'O2 comparée à la mesure de [Gri14] (233 ± 22 µm).

    On utilise la définition de Grimes : distance de pénétration de l'O2 dans
    un tissu pleinement consommateur, l = sqrt(6 D C / q), avec les paramètres
    de consommation MESURÉS (hors loi de Pirt, qui décrit l'adaptation
    ultérieure des cellules quiescentes).
    """
    p = Parameters()
    # OCR de référence de Grimes : 5.6e-17 mol/cellule/s à pleine activité
    l = p.oxygen_diffusion_limit
    target = 233.0
    rel = abs(l - target) / target
    return Check("Limite de diffusion de l'O₂ (vs Grimes 2014 : 233±22 µm)",
                 "validation", float(l), target * (1 + tol), rel <= tol, "µm",
                 detail=f"écart {100*rel:.0f} %")


def check_ocr_magnitude(tol: float = 0.35) -> Check:
    """Taux de consommation d'O2 volumique, comparé à [Gri14]/[Gri16] (~22-27 mmHg/s)."""
    p = Parameters()
    ocr = conc_to_po2(p.q_o2_max_vol)
    target = 24.5      # moyenne des valeurs publiées 22.1 et 27.4 mmHg/s
    rel = abs(ocr - target) / target
    return Check("OCR volumique (vs 22-27 mmHg/s publiés)", "validation",
                 float(ocr), target * (1 + tol), rel <= tol, "mmHg/s")


# ---------------------------------------------------------------------------
# 2. VALIDATION — comparaison aux données EMT6/Ro
# ---------------------------------------------------------------------------
def check_toxicity_fractionation() -> Check:
    """Le modèle de toxicité doit reproduire la loi fondamentale du fractionnement.

    À DOSE PHYSIQUE ÉGALE (60 Gy), un schéma hypofractionné doit être plus
    toxique pour les tissus sains à réponse tardive qu'un schéma conventionnel,
    parce que α/β = 3 Gy y est faible. C'est le fondement radiobiologique de
    tout le fractionnement clinique [Fowler ; Joiner & van der Kogel].
    """
    from .toxicity import ToxicityModel
    def tox(n, d):
        t = ToxicityModel()
        for i in range(n):
            t.add_fraction(i * 1.4, d)
        return t.eqd2()
    conv, hypo = tox(30, 2.0), tox(12, 5.0)
    ratio = hypo / max(conv, 1e-9)
    return Check("Fractionnement : hypofractionné plus toxique (α/β=3 Gy)",
                 "validation", float(ratio), 1.0, ratio > 1.2, "×",
                 detail=f"EQD2 conventionnel 30×2 Gy = {conv:.1f} Gy ; "
                        f"agressif 12×5 Gy = {hypo:.1f} Gy (dose physique identique)")


def check_toxicity_clinical_tolerance() -> Check:
    """Un schéma clinique standard doit rester sous le seuil de complication.

    30 × 2 Gy est le schéma conventionnel de référence. Si le modèle prédisait
    une forte probabilité de complication pour ce schéma, il serait inutilisable :
    ce traitement est administré quotidiennement en routine.
    """
    from .toxicity import conventional_schedule_check
    s = conventional_schedule_check()
    ntcp = s["ntcp"]
    return Check("Tolérance clinique d'un schéma standard (30×2 Gy)",
                 "validation", float(ntcp), 0.05, ntcp < 0.05, "NTCP",
                 detail=f"EQD2 tissu sain = {s['eqd2_gy']:.1f} Gy, "
                        f"NTCP = {100*ntcp:.2f} % (modèle LKB, Emami/Burman 1991)")


def check_radioresistance_selection() -> Check:
    """La radiothérapie fractionnée doit SÉLECTIONNER les clones résistants.

    Prédiction centrale d'Alfonso & Berk (Radiat Oncol 14:88, 2019) : en
    présence d'une hétérogénéité intratumorale de radiosensibilité, le
    fractionnement standard opère un déplacement de la population vers les
    clones résistants. Contrôle négatif inclus : sans irradiation, aucune
    dérive ne doit apparaître.
    """
    from .parameters import Parameters as P
    from .agents import AgentSimulation, AgentParams
    from .evolution import EvolutionParams
    ev = EvolutionParams(u_driver=1e-3, v_passenger=0.5, s_driver=0.01,
                         s_sigma=0.005, initial_radio_sigma=0.15,
                         p_radioresist=0.1, radioresist_gain=0.3)

    def run(doses):
        a = AgentSimulation(P(), AgentParams(), seed_radius=45.0, seed=4,
                            evolution=ev)
        a.run(2.0, dt=1800.0, record_every_h=24.0, max_cells=6000)
        r0 = a.mean_radioresistance()
        for d in doses:
            a.irradiate(d)
            a.run(1.0, dt=1800.0, record_every_h=24.0, max_cells=6000)
        return r0, a.mean_radioresistance()

    r0_c, r1_c = run([])              # témoin
    r0_t, r1_t = run([2.0] * 5)       # 5 x 2 Gy
    drift_ctrl = abs(r1_c - r0_c)
    gain = (r0_t - r1_t) / max(r0_t, 1e-9)
    ok = gain > 0.02 and drift_ctrl < 0.005
    return Check("Sélection de la radiorésistance sous fractionnement",
                 "validation", float(100 * gain), 2.0, ok, "%",
                 detail=f"5×2 Gy : facteur α {r0_t:.4f} → {r1_t:.4f} "
                        f"({100*gain:+.1f} % de résistance) ; "
                        f"témoin non irradié : dérive {drift_ctrl:.5f}")


def check_sensitivity_identifiability() -> Check:
    """Le modèle doit être numériquement identifiable sur ses observables.

    Un conditionnement élevé de la matrice de sensibilité signifie que
    certaines combinaisons de paramètres sont indiscernables au vu des
    données : le modèle serait alors sur-paramétré. Seuil 100 = usage
    courant en analyse d'identifiabilité.
    """
    from .sensitivity import local_sensitivity, identifiability
    sens = local_sensitivity()
    idn = identifiability(sens)
    cond = idn["condition_number"]
    pairs = idn["collinear_pairs"]
    detail = (f"{len(pairs)} paire(s) colinéaire(s) |r|>0.95"
              + (f" : {pairs[0][0]}/{pairs[0][1]} r={pairs[0][2]:.3f}" if pairs else "")
              + f" ; sans effet : {idn['unidentifiable']}")
    return Check("Identifiabilité (conditionnement de la sensibilité)",
                 "verification", float(cond), 100.0, cond <= 100.0, "-",
                 detail=detail)


def check_uncertainty_covers_experiment(n_draws: int = 60) -> Check:
    """Propagation d'incertitude : l'IC95 du modèle doit recouvrir l'expérience.

    Test plus honnête que la comparaison ponctuelle : il demande que la
    prédiction *avec sa barre d'erreur* soit compatible avec la mesure *avec
    sa barre d'erreur*. Un modèle qui passe ce test avec des barres énormes
    n'est pas pour autant bon — c'est pourquoi le CV est reporté.
    """
    from .sensitivity import propagate_uncertainty
    exp = {"rim_haut": (240., 40.), "necrose_haut": (516., 80.),
           "td_haut": (23., 4.), "rim_bas": (60., 20.),
           "necrose_bas": (152., 40.), "td_bas": (26., 8.)}
    summ = propagate_uncertainty(n_draws=n_draws, seed=0).summary()
    n_ok = sum(1 for o, (e, tol) in exp.items()
               if np.isfinite(summ[o]["lo"])
               and summ[o]["lo"] <= e + tol and summ[o]["hi"] >= e - tol)
    return Check("Incertitude propagée recouvrant l'expérience", "validation",
                 float(n_ok), 6.0, n_ok >= 5, "/6",
                 detail=f"{n_ok}/6 observables compatibles ({n_draws} tirages Monte-Carlo)")


def check_experimental_targets() -> List[Check]:
    """Confronte le modèle aux six observables de [FS86]/[Hunt08]."""
    p = Parameters()
    diag = bm.diagnostics(p)
    checks: List[Check] = []
    labels = {
        "viable_rim_um": ("Couronne viable", "µm", "rim_tol_um"),
        "necrosis_onset_diameter_um": ("Diamètre d'apparition de la nécrose",
                                       "µm", "onset_tol_um"),
        "doubling_time_h": ("Temps de doublement initial", "h", "td_tol_h"),
    }
    cond = {"high": "16.5 mM glc / 0.28 mM O₂", "low": "0.8 mM glc / 0.07 mM O₂"}
    for key, tgt in bm.TARGETS.items():
        for field_, (label, unit, tolk) in labels.items():
            got, exp, tol = diag[key][field_], tgt[field_], tgt[tolk]
            ok = bool(np.isfinite(got) and abs(got - exp) <= tol)
            checks.append(Check(f"{label} [{cond[key]}]", "validation",
                                float(got), float(exp), ok, unit,
                                detail=f"mesure {exp:.0f} ± {tol:.0f} {unit}"))
    return checks


def check_central_po2() -> List[Check]:
    """pO2 central comparé à Mueller-Klieser & Sutherland (1982)."""
    p = Parameters().copy_with(c_o2_bulk=po2_to_conc(155.0), c_glc_bulk=16.5)
    out = []
    for diam, exp in bm.MK82_CENTRAL_PO2.items():
        got = bm.central_po2(p, diam)
        tol = max(0.35 * exp, 8.0)
        out.append(Check(f"pO₂ central (Ø {diam:.0f} µm) [MK82]", "validation",
                         float(got), float(exp), abs(got - exp) <= tol, "mmHg",
                         detail=f"mesure {exp:.0f} mmHg ± 35 %"))
    return out


def check_gompertzian_growth(min_r2: float = 0.95) -> Check:
    """La courbe de croissance doit être gompertzienne (observation de [FS86])."""
    from .growth import SpheroidSimulation
    from scipy.optimize import curve_fit

    sim = SpheroidSimulation(Parameters(), R0=60.0, n_grid=100)
    sim.run(28.0, dt_max=900.0, record_every_h=12.0)
    a = sim.as_arrays()
    t, v = a["t_days"], a["volume_um3"]

    def gompertz(t, v0, vmax, b):
        return vmax * np.exp(np.log(max(v0, 1e-9) / vmax) * np.exp(-b * t))

    try:
        popt, _ = curve_fit(gompertz, t, v, p0=[v[0], max(v[-1] * 1.5, v[0] * 10), 0.15],
                            maxfev=20000)
        pred = gompertz(t, *popt)
        ss_res = float(np.sum((v - pred) ** 2))
        ss_tot = float(np.sum((v - v.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    except Exception:
        r2 = 0.0
    return Check("Croissance gompertzienne (R² de l'ajustement)", "validation",
                 float(r2), min_r2, r2 >= min_r2, "-")


def check_rim_saturation(tol_um: float = 45.0) -> Check:
    """La couronne viable doit se stabiliser quand le sphéroïde grossit [Jia05]."""
    p = Parameters()
    rims = [bm.equilibrium_viable_rim(p, diameter=d) for d in (900.0, 1200.0, 1600.0)]
    spread = float(max(rims) - min(rims))
    return Check("Couronne viable ~constante aux grands diamètres", "validation",
                 spread, tol_um, spread <= tol_um, "µm",
                 detail=f"épaisseurs {[round(r) for r in rims]} µm")


def check_model_cross_validation(tol: float = 0.15, days: float = 5.0) -> Check:
    """Validation croisée : modèle continu vs modèle à agents.

    Les deux descriptions sont numériquement indépendantes (EDP à frontière
    libre en symétrie sphérique d'un côté ; dynamique de N corps sur-amortie
    avec cycles cellulaires stochastiques de l'autre). Elles ne partagent que
    la physique sous-jacente (paramètres et lois de transport/métabolisme).
    Leur accord sur le rayon du sphéroïde constitue donc un test fort de
    cohérence interne, au sens de la vérification par solutions manufacturées
    croisées.
    """
    from .agents import AgentParams, AgentSimulation
    from .growth import SpheroidSimulation

    p = Parameters()
    R0 = 60.0
    abm = AgentSimulation(p, AgentParams(), seed_radius=R0, seed=1)
    abm.run(days, dt=1800.0, record_every_h=24.0, max_cells=40000)
    r_abm = abm.history[-1]["radius_um"]

    sim = SpheroidSimulation(p, R0=R0, n_grid=100)
    sim.run(days, dt_max=900.0, record_every_h=24.0)
    r_cont = sim.as_arrays()["radius_um"][-1]

    rel = abs(r_abm - r_cont) / r_cont
    return Check("Validation croisée continu / agents (rayon)", "vérification",
                 float(rel), tol, rel <= tol, "écart relatif",
                 detail=f"agents {r_abm:.0f} µm, continu {r_cont:.0f} µm à {days:.0f} j")


def check_oer_range() -> Check:
    """L'OER doit varier entre ~1 (anoxie) et ~3 (bien oxygéné) [Alper56]."""
    from .therapy import oer
    p = Parameters()
    lo = float(oer(np.array([0.0]), p)[0])
    hi = float(oer(np.array([100.0]), p)[0])
    ok = abs(lo - 1.0) < 1e-9 and abs(hi - 3.0) < 0.15
    return Check("OER borné entre 1 (anoxie) et 3 (normoxie)", "validation",
                 hi, 3.0, ok, "-", detail=f"OER(0 mmHg)={lo:.2f}, OER(100)={hi:.2f}")


def check_hypoxic_radioresistance() -> Check:
    """Les cellules hypoxiques DOIVENT être plus radiorésistantes [Alper56].

    Test de non-régression sur le sens physique de l'effet oxygène : la
    fraction survivante doit décroître de façon monotone avec la pO2, et le
    rapport des doses isoeffet anoxie/normoxie doit valoir ~OER_max.
    """
    from .therapy import survival_fraction
    p = Parameters()
    po2 = np.array([0.0, 1.0, 3.0, 10.0, 40.0, 155.0])
    sf = survival_fraction(6.0, po2, p)
    monotone = bool(np.all(np.diff(sf) < 0))

    from scipy.optimize import brentq
    def dose_for(po: float) -> float:
        return brentq(lambda x: survival_fraction(x, np.array([po]), p)[0] - 0.1,
                      0.05, 80.0)
    ratio = dose_for(0.0) / dose_for(155.0)
    ok = monotone and abs(ratio - p.oer_max) / p.oer_max < 0.15
    return Check("Radiorésistance hypoxique (dose isoeffet anoxie/normoxie)",
                 "validation", float(ratio), float(p.oer_max), ok, "×",
                 detail=f"survie monotone décroissante en pO₂ : {monotone}")


# ---------------------------------------------------------------------------
# 3. VALIDATION — moteur évolutif
# ---------------------------------------------------------------------------
def check_driver_survival_probability(tol: float = 0.35) -> Check:
    """Probabilité qu'une lignée échappe à l'extinction [Bozic et al. 2010].

    Pour un processus de branchement avec b = 1-d et d = (1-s)^k/2, la
    probabilité de survie d'une lignée portant k drivers vaut 1 - d/b.
    Pour k=1 et s=0.004 : 0.797 %. Bozic et al. en tirent la conclusion que
    plus de 99 % des drivers apparus s'éteignent par simple dérive.
    """
    from .evolution import EvolutionParams, branching_process

    ep = EvolutionParams()
    n_run, escaped = 3000, 0
    for i in range(n_run):
        r = branching_process(ep, n_generations=4000, n_max=400,
                              rng=np.random.default_rng(50_000 + i))
        if r.n_cells >= 400:
            escaped += 1
    obs = escaped / n_run
    d1 = (1.0 - ep.s_driver) / 2.0
    theory = 1.0 - d1 / (1.0 - d1)
    rel = abs(obs - theory) / theory
    return Check("Survie d'une lignée mutante (vs théorie de Bozic 2010)",
                 "validation", obs * 100, theory * 100, rel <= tol, "%",
                 detail=f"théorie {100*theory:.3f} %, simulé {100*obs:.3f} % "
                        f"sur {n_run} réalisations")


def check_neutral_power_law(min_r2: float = 0.98) -> Check:
    """Loi de puissance 1/f de l'évolution neutre [Williams et al. 2016].

    En régime neutre, la distribution cumulée M(f) des mutations subclonales
    doit être proportionnelle à 1/f. Ce comportement n'est PAS imposé : il doit
    émerger de la dynamique stochastique de mutation et de dérive.
    """
    from .evolution import (EvolutionParams, exponential_growth_process,
                            neutrality_test)

    ep = EvolutionParams(s_driver=0.0, s_sigma=0.0, v_passenger=10.0, d_base=0.35)
    r2s: List[float] = []
    i = 0
    while len(r2s) < 3 and i < 12:
        r = exponential_growth_process(ep, n_target=80_000,
                                       rng=np.random.default_rng(600 + i))
        i += 1
        if r.n_cells < 40_000:
            continue          # lignée éteinte par dérive : cas normal
        vaf = r.tree.allele_frequencies(r.counts)
        r2s.append(neutrality_test(vaf)["r2"])
    mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
    return Check("Spectre neutre en 1/f (vs Williams et al. 2016)", "validation",
                 mean_r2, min_r2, bool(mean_r2 >= min_r2), "R²",
                 detail=f"{len(r2s)} réalisations, seuil de neutralité 0.98")


def check_selection_detectable() -> Check:
    """Une expansion subclonale doit FAIRE CHUTER le R² du test 1/f.

    Contrôle positif du test de neutralité : on implante délibérément un
    sous-clone avantagé et on vérifie (i) qu'il balaie une fraction importante
    de la tumeur et (ii) que la signature neutre se dégrade.
    """
    from .evolution import (EvolutionParams, exponential_growth_process,
                            neutrality_test)

    ep = EvolutionParams(s_driver=0.0, s_sigma=0.0, v_passenger=10.0, d_base=0.35)
    fracs, r2_sel = [], []
    i = 0
    while len(fracs) < 3 and i < 12:
        r = exponential_growth_process(
            ep, n_target=80_000, rng=np.random.default_rng(600 + i),
            planted_subclone_at=1000, planted_advantage=0.9)
        i += 1
        if r.n_cells < 40_000:
            continue
        sub = r.tree.subtree_counts(r.counts)
        f = max((n / r.n_cells for cid, n in sub.items()
                 if r.tree.clones[cid].n_drivers >= 2), default=0.0)
        fracs.append(f)
        r2_sel.append(neutrality_test(r.tree.allele_frequencies(r.counts))["r2"])
    mean_f = float(np.mean(fracs)) if fracs else 0.0
    return Check("Expansion subclonale sous sélection (contrôle positif)",
                 "validation", mean_f, 0.10, bool(mean_f >= 0.10), "fraction",
                 detail=f"le sous-clone avantagé occupe {100*mean_f:.0f} % de la "
                        f"tumeur ; R²(1/f) = {np.mean(r2_sel):.3f}")


def check_evolution_reproducibility() -> Check:
    """Même graine -> trajectoire identique ; graine différente -> divergente."""
    from .agents import AgentParams, AgentSimulation
    from .evolution import EvolutionParams

    def run(seed: int) -> tuple:
        ev = EvolutionParams(u_driver=3e-3, v_passenger=1.0, s_driver=0.3)
        a = AgentSimulation(Parameters(), AgentParams(), seed_radius=45.0,
                            seed=seed, evolution=ev)
        a.run(3.0, dt=1800.0, record_every_h=72.0, max_cells=6000)
        h = a.history[-1]
        return h["n_cells"], round(h["mean_fitness"], 12), a.tree.n_clones

    a1, a2, b = run(7), run(7), run(8)
    ok = (a1 == a2) and (a1 != b)
    return Check("Reproductibilité à graine fixée (et divergence sinon)",
                 "vérification", 1.0 if ok else 0.0, 1.0, ok, "-",
                 detail=f"graine 7 : {a1} == {a2} ; graine 8 : {b}")


def check_spatial_selection_gradient() -> Check:
    """Le microenvironnement doit créer un gradient de pression sélective.

    Dans un sphéroïde dépassant la limite de diffusion, les cellules
    périphériques prolifèrent nettement plus vite que celles du coeur : la
    sélection est donc SPATIALEMENT structurée, contrairement à un modèle de
    population bien mélangée.
    """
    from .agents import AgentParams, AgentSimulation
    from .evolution import EvolutionParams

    a = AgentSimulation(Parameters(), AgentParams(), seed_radius=260.0,
                        seed=5, evolution=EvolutionParams())
    k_p, _, po2 = a._local_rates()
    d = np.linalg.norm(a.pos, axis=1)
    R = a.spheroid_radius()
    core, rim = d < 0.4 * R, d > 0.85 * R
    ratio = float(k_p[rim].mean() / max(k_p[core].mean(), 1e-30))
    return Check("Gradient spatial de pression sélective (périphérie/coeur)",
                 "validation", ratio, 1.2, ratio >= 1.2, "×",
                 detail=f"pO₂ coeur {po2[core].mean():.0f} mmHg vs "
                        f"périphérie {po2[rim].mean():.0f} mmHg")


# ---------------------------------------------------------------------------
def run_all(verbose: bool = True) -> List[Check]:
    """Exécute toute la suite et renvoie les résultats."""
    checks: List[Check] = []
    steps: List[Callable] = [
        check_analytic_solution, check_mesh_convergence, check_mass_conservation,
        check_jacobian, check_quasi_steady_state, check_positivity,
        check_time_step_independence, check_diffusion_limit, check_ocr_magnitude,
        check_model_cross_validation,
        check_rim_saturation, check_oer_range, check_hypoxic_radioresistance,
        check_driver_survival_probability, check_neutral_power_law,
        check_selection_detectable, check_evolution_reproducibility,
        check_spatial_selection_gradient,
        check_gompertzian_growth,
        check_sensitivity_identifiability,
        check_uncertainty_covers_experiment,
        check_toxicity_fractionation,
        check_toxicity_clinical_tolerance,
        check_radioresistance_selection,
    ]
    for fn in steps:
        try:
            checks.append(fn())
        except Exception as exc:                                # pragma: no cover
            checks.append(Check(fn.__name__, "vérification", float("nan"),
                                0.0, False, detail=f"exception : {exc}"))
    checks.extend(check_experimental_targets())
    checks.extend(check_central_po2())

    if verbose:
        for c in checks:
            print(c)
        n_ok = sum(c.passed for c in checks)
        print(f"\n{n_ok}/{len(checks)} tests réussis")
    return checks


def to_markdown(checks: List[Check]) -> str:
    """Rapport Markdown des vérifications et validations."""
    lines = []
    for cat in ("vérification", "validation"):
        sub = [c for c in checks if c.category == cat]
        if not sub:
            continue
        lines.append(f"\n### {cat.capitalize()}\n")
        lines.append("| Test | Valeur | Référence / critère | Statut |")
        lines.append("|---|---|---|---|")
        for c in sub:
            flag = "✅" if c.passed else "❌"
            lines.append(f"| {c.name} | {c.value:.4g} {c.unit} | "
                         f"{c.criterion:.4g} {c.unit} | {flag} |")
    n_ok = sum(c.passed for c in checks)
    lines.append(f"\n**Bilan : {n_ok}/{len(checks)} tests réussis.**")
    return "\n".join(lines)


if __name__ == "__main__":       # pragma: no cover
    run_all()
