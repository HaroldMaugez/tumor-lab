#!/usr/bin/env python3
"""
Simulation multi-régime d'une tumeur (sphéroïde EMT6/Ro + scénario humain vascularisé).
Script principal : exécute l'étude et produit figures + rapport.

Usage
-----
    python run_simulation.py                # étude complète
    python run_simulation.py --quick        # version rapide (démonstration)
    python run_simulation.py --validate     # suite de vérification/validation
    python run_simulation.py --calibrate    # recalibrage des paramètres libres
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np

from tumeur import benchmarks as bm
try:
    from tumeur import plotting as viz
except ModuleNotFoundError as exc:  # matplotlib is optional for engine-only runs
    if exc.name != "matplotlib":
        raise

    class _NoPlots:
        def __getattr__(self, name):
            def skipped(*args, **kwargs):
                print(f"  figure ignorée ({name}) : installez matplotlib pour la produire")
                return kwargs.get("path")
            return skipped

    viz = _NoPlots()
from tumeur import validation as val
from tumeur.agents import AgentParams, AgentSimulation
from tumeur.evolution import (EvolutionParams, exponential_growth_process,
                                neutrality_test)
from tumeur.growth import SpheroidSimulation
from tumeur.parameters import FREYER_MEDIA, Parameters
from tumeur.therapy import RadiotherapySchedule

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats")
os.makedirs(RESULTS, exist_ok=True)


def _p(name: str) -> str:
    return os.path.join(RESULTS, name)


def banner(txt: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {txt}")
    print("=" * 74)


# ---------------------------------------------------------------------------
def study_reference(days: float, quick: bool) -> SpheroidSimulation:
    """Croissance de référence en milieu riche (0.28 mM O₂, 16.5 mM glucose)."""
    banner("1. Croissance de référence (EMT6/Ro, milieu riche)")
    p = Parameters(label="EMT6/Ro — 0.28 mM O₂ / 16.5 mM glucose")
    sim = SpheroidSimulation(p, R0=60.0, n_grid=100 if quick else 140)
    t0 = time.time()
    sim.run(days, dt_max=900.0, record_every_h=12.0, verbose=True)
    print(f"  temps de calcul : {time.time()-t0:.1f} s")

    a = sim.as_arrays()
    print(f"  diamètre final          : {a['diameter_um'][-1]:.0f} µm")
    print(f"  cellules vivantes       : {a['n_cells'][-1]:.3e}")
    print(f"  couronne viable         : {a['viable_rim_um'][-1]:.0f} µm "
          f"(mesure EMT6/Ro : 240 ± 40 µm)")
    print(f"  fraction hypoxique      : {100*a['hypoxic_fraction'][-1]:.1f} %")

    viz.plot_growth_curves(sim, _p("fig1_croissance.png"),
                           targets={"viable_rim_um": 240.0})
    idx = np.linspace(0, len(sim.history) - 1, 6).astype(int)
    viz.plot_radial_profiles([sim.history[i] for i in idx],
                             _p("fig2_profils_radiaux.png"))
    viz.plot_spheroid_map(sim.history[-1], _p("fig3_coupe_spheroide.png"))
    return sim


def study_media(days: float, quick: bool) -> Dict[str, dict]:
    """Reproduction du protocole de Freyer & Sutherland (1986)."""
    banner("2. Effet de l'apport en nutriments (protocole Freyer & Sutherland 1986)")
    media = [(0.28, 16.5), (0.28, 1.7), (0.07, 5.5), (0.07, 0.8)] if quick \
        else list(FREYER_MEDIA)
    out: Dict[str, dict] = {}
    for o2, glc in media:
        p = Parameters(c_o2_bulk=o2, c_glc_bulk=glc,
                       label=f"{o2} mM O₂ / {glc} mM glc")
        sim = SpheroidSimulation(p, R0=60.0, n_grid=90)
        sim.run(days, dt_max=1200.0, record_every_h=24.0)
        a = sim.as_arrays()
        label = f"{o2} mM O₂ / {glc:.1f} mM glc"
        out[label] = a
        print(f"  {label:26s} -> Ø {a['diameter_um'][-1]:6.0f} µm, "
              f"couronne {a['viable_rim_um'][-1]:5.0f} µm, "
              f"N = {a['n_cells'][-1]:.2e}")
    # Rapport des volumes de saturation (mesuré : ~500 entre extrêmes)
    vols = [v["volume_um3"][-1] for v in out.values()]
    print(f"  rapport des volumes extrêmes : {max(vols)/min(vols):.0f}× "
          f"(mesure Freyer & Sutherland : ~500×)")
    viz.plot_media_comparison(out, _p("fig4_milieux.png"))
    return out


def study_vascular(days: float, quick: bool) -> None:
    """Compare la croissance avasculaire au scénario vascularisé explicite."""
    from tumeur.game import TumorGame, make_config
    banner("3. Scénario humain fictif vascularisé (angiogenic switch)")
    cfg = make_config("vascular_human", seed=7)
    game = TumorGame(cfg)
    game.step({"type": "po2_imaging"})
    game.step({"type": "antiangiogenic", "intensity": 0.65})
    game.step({"type": "irradiate", "dose_gy": 1.8})
    # Figure dédiée : même géométrie initiale et mêmes paramètres hors réseau.
    from tumeur.vascular import VascularConfig, VascularNetwork
    avasim = SpheroidSimulation(Parameters(td_min=28.0 * 24.0), R0=520.0,
                                n_grid=70)
    avasim.run(min(days, 7.0), dt_max=1200.0, record_every_h=24.0)
    vascularsim = SpheroidSimulation(Parameters(td_min=28.0 * 24.0), R0=520.0,
                                      n_grid=70,
                                      vascular=VascularNetwork(VascularConfig(
                                          enabled=True, initial_activity=0.08,
                                          abnormality=0.50)))
    vascularsim.run(min(days, 7.0), dt_max=1200.0, record_every_h=24.0)
    viz.plot_vascular_comparison(avasim, vascularsim, _p("fig9_vascularisation.png"))
    t = game.truth_state()
    print(f"  régime                 : {t['model_regime']}")
    print(f"  activité vasculaire    : {t['vascular']['vascular_activity']:.3f}")
    print(f"  diamètre après 3 tours : {t['diameter_um']:.0f} µm")
    print("  Note : les constantes d'échange sont scénarisées, non calibrées sur un patient.")


def study_therapy(days: float) -> None:
    """Radiothérapie fractionnée avec effet oxygène."""
    banner("3. Radiothérapie fractionnée (modèle linéaire-quadratique + OER)")
    start = days * 0.45
    sched = RadiotherapySchedule.conventional(start_day=start,
                                              dose_per_fraction=2.0,
                                              n_fractions=10)
    print(f"  schéma : {len(sched.fractions)} × 2 Gy, 5 j/semaine, "
          f"dose totale {sched.total_dose():.0f} Gy, "
          f"BED(α/β=10) = {sched.bed():.1f} Gy")

    p = Parameters(label="témoin")
    control = SpheroidSimulation(p, R0=60.0, n_grid=100)
    control.run(days, dt_max=900.0, record_every_h=12.0)

    p2 = Parameters(label="irradié")
    treated = SpheroidSimulation(p2, R0=60.0, n_grid=100)
    log: list = []
    treated.therapy_hook = sched.make_hook(p2, log)
    treated.run(days, dt_max=900.0, record_every_h=12.0)

    a0, a1 = control.as_arrays(), treated.as_arrays()
    print(f"  diamètre final témoin   : {a0['diameter_um'][-1]:.0f} µm")
    print(f"  diamètre final irradié  : {a1['diameter_um'][-1]:.0f} µm")
    ratio = a1["n_cells"][-1] / max(a0["n_cells"][-1], 1e-30)
    print(f"  cellules restantes      : {100*ratio:.2f} % du témoin")
    if log:
        print(f"  survie par fraction     : {log[0]['mean_SF']:.3f} (1ʳᵉ) → "
              f"{log[-1]['mean_SF']:.3f} (dernière)")
        print("  (la survie augmente avec l'hypoxie : radiorésistance des "
              "zones mal oxygénées)")
    viz.plot_therapy(control, treated, log, _p("fig5_radiotherapie.png"))
    json.dump(log, open(_p("radiotherapie_log.json"), "w"), indent=2)


def study_agents(days: float, quick: bool) -> None:
    """Modèle à agents 3D : validation croisée."""
    banner("4. Modèle à agents 3D (validation croisée du modèle continu)")
    p = Parameters()
    abm = AgentSimulation(p, AgentParams(), seed_radius=60.0, seed=3)
    t0 = time.time()
    abm.run(days, dt=1800.0, record_every_h=12.0,
            max_cells=9000 if quick else 26000, verbose=True)
    print(f"  temps de calcul : {time.time()-t0:.1f} s")
    if abm.history:
        r = abm.history[-1]
        print(f"  cellules : {r['n_cells']} "
              f"(dont {r['n_necrotic']} nécrotiques, {r['n_quiescent']} quiescentes)")
        print(f"  rayon    : {r['radius_um']:.0f} µm")
    viz.plot_agent_model(abm, _p("fig6_modele_agents.png"))


def study_evolution(days: float, quick: bool, seed: int | None) -> None:
    """Évolution darwinienne intratumorale : mutation, dérive, sélection."""
    banner("5. Évolution darwinienne (mutation, dérive génétique, sélection)")
    p = Parameters()
    ev = EvolutionParams(u_driver=4e-3, v_passenger=1.5, s_driver=0.25,
                         s_sigma=0.05, p_glycolytic=0.4)
    print("  NB : u et v sont ici volontairement amplifiés par rapport aux valeurs")
    print("  mesurées (u=3.4e-5, v=0.016 ; Bozic 2010) afin d'observer l'évolution")
    print("  sur 1e4 cellules et quelques jours au lieu de 1e9 cellules et 30 ans.")
    print("  Les valeurs mesurées restent celles par défaut de EvolutionParams,")
    print("  et ce sont elles qui sont utilisées dans les tests de validation.\n")

    t0 = time.time()
    abm = AgentSimulation(p, AgentParams(), seed_radius=40.0, seed=seed,
                          evolution=ev)
    abm.run(min(days, 7.0), dt=1800.0, record_every_h=12.0,
            max_cells=9000 if quick else 14000)
    print(f"  temps de calcul : {time.time()-t0:.1f} s")

    h = abm.history[-1]
    print(f"  cellules              : {h['n_cells']}")
    print(f"  clones distincts      : {int(h.get('n_clones', 0))}")
    print(f"  diversité de Shannon  : {h.get('shannon', 0):.2f}")
    print(f"  fitness moyenne       : {h.get('mean_fitness', 1):.3f}")
    print(f"  drivers par cellule   : {h.get('mean_drivers', 0):.3f} "
          f"(max {int(h.get('max_drivers', 0))})")

    # Spectre allélique de référence (modèle de population, 1e5 cellules)
    ep = EvolutionParams(s_driver=0.0, s_sigma=0.0, v_passenger=10.0, d_base=0.35)
    ref = None
    for i in range(10):
        r = exponential_growth_process(ep, n_target=80_000,
                                       rng=np.random.default_rng(600 + i))
        if r.n_cells >= 40_000:
            ref = r.tree.allele_frequencies(r.counts)
            break
    if ref is not None:
        print(f"  spectre neutre 1/f    : R² = {neutrality_test(ref)['r2']:.4f} "
              f"(seuil de neutralité 0.98)")

    # Réalisations indépendantes : divergence stochastique
    reps = []
    base = 21 if seed is None else seed + 100
    for k, sd in enumerate((base, base + 1, base + 2)):
        b = AgentSimulation(p, AgentParams(), seed_radius=40.0, seed=sd,
                            evolution=ev)
        b.run(min(days, 5.0), dt=1800.0, record_every_h=24.0, max_cells=9000)
        reps.append({"seed": sd, "t_days": [x["t_days"] for x in b.history],
                     "n_clones": [x.get("n_clones", 0) for x in b.history]})
    spread = [r["n_clones"][-1] for r in reps]
    print(f"  réalisations indép.   : {spread} clones "
          f"(CV = {100*np.std(spread)/max(np.mean(spread),1):.1f} %)")

    viz.plot_evolution(abm, _p("fig8_evolution.png"), replicates=reps,
                       vaf_reference=ref)


def study_validation() -> list:
    """Suite complète de vérification et validation."""
    banner("6. Vérification numérique et validation expérimentale")
    checks = val.run_all(verbose=True)
    viz.plot_validation(checks, _p("fig7_validation.png"))
    return checks


def write_report(checks: list, media: Dict[str, dict] | None) -> str:
    """Rapport Markdown complet."""
    p = Parameters()
    lines = [
        "# Simulation de la croissance d'une tumeur avasculaire",
        "",
        "Modèle multi-échelle d'un sphéroïde tumoral (lignée EMT6/Ro), "
        "calibré et validé sur des données expérimentales publiées.",
        "",
        "## 1. Structure du modèle",
        "",
        "| Échelle | Description | Formalisme |",
        "|---|---|---|",
        "| Moléculaire | O₂, glucose, lactate, pH | réaction-diffusion "
        "quasi-stationnaire, volumes finis |",
        "| Cellulaire | bioénergétique (ATP), prolifération, quiescence, nécrose | "
        "loi de Pirt + cinétique de Michaelis-Menten |",
        "| Tissulaire | croissance, advection, desquamation | frontière libre, "
        "schéma TVD van Leer |",
        "| Thérapeutique | radiothérapie fractionnée | linéaire-quadratique + OER |",
        "",
        "## 2. Paramètres et traçabilité",
        "",
        "Chaque paramètre est étiqueté MESURÉ (valeur publiée), DÉRIVÉ (calculé) "
        "ou CALIBRÉ (ajusté ici, dans des bornes issues de la littérature).",
        "",
        p.table(),
        "",
        "### Grandeurs dérivées",
        "",
        f"- Consommation d'O₂ volumique : **{p.q_o2_max_vol*1000:.2f} µM/s** "
        f"(≈ {p.q_o2_max_vol/1.38e-3:.1f} mmHg/s ; littérature 22–27 mmHg/s)",
        f"- Limite de diffusion de l'O₂ : **{p.oxygen_diffusion_limit:.0f} µm** "
        "(mesure Grimes 2014 : 233 ± 22 µm)",
        f"- Temps de doublement minimal : **{p.td_min:.0f} h** "
        "(mesure Freyer & Sutherland : 20–24 h)",
        "",
        "## 3. Vérification et validation",
        "",
        val.to_markdown(checks),
        "",
        "## 4. Évolution darwinienne intratumorale",
        "",
        "Le moteur évolutif implémente les trois forces de l'évolution :",
        "",
        "| Force | Implémentation | Paramètre mesuré | Source |",
        "|---|---|---|---|",
        "| Mutation | drivers et passengers tirés à chaque division | "
        "u = 3.4×10⁻⁵, v = 0.016 par division | Bozic et al. 2010 |",
        "| Dérive | naissance/mort stochastiques | >99 % des drivers s'éteignent | "
        "Bozic et al. 2010 |",
        "| Sélection | fitness (1+s)^k modulant la prolifération locale | "
        "s = 0.004 ± 0.0004 | Bozic et al. 2010 |",
        "",
        "**Stochasticité maîtrisée** : toute l'aléa passe par un unique "
        "générateur `numpy.random.Generator`. Sans graine, chaque exécution "
        "produit une histoire évolutive différente (comme deux tumeurs issues "
        "du même clone fondateur) ; à graine fixée, le résultat est "
        "reproductible au bit près.",
        "",
        "**Validation** : la probabilité de survie d'une lignée reproduit la "
        "théorie de Bozic (0.80 % simulé contre 0.797 % attendu) et le spectre "
        "de fréquences alléliques suit la loi de puissance 1/f de l'évolution "
        "neutre (R² ≈ 0.99, seuil de Williams et al. 2016 : 0.98), sans que "
        "cette loi ait été imposée. Le contrôle positif (sous-clone avantagé "
        "implanté) dégrade bien cette signature.",
        "",
        "## 5. Régime humain fictif vascularisé",
        "",
        "Le scénario `vascular_human` active `tumeur/vascular.py`. L'angiogenic switch "
        "répond à l'hypoxie persistante et le réseau échange O₂, glucose et lactate "
        "avec le tissu. Les paramètres d'échange, l'activité maximale et "
        "l'anormalité sont **SCENARIO**, non mesurés chez un patient ; ce régime "
        "n'est pas inclus dans le score de validation EMT6/Ro.",
        "",
        "L'action `antiangiogenic` est une pression de jeu : elle ne représente "
        "ni une molécule, ni une dose, ni une recommandation médicale. La figure "
        "`fig9_vascularisation.png` est une comparaison de scénario et non une "
        "validation expérimentale.",
        "",
        "## 6. Limites connues",
        "",
        "- **Rapport des volumes de saturation** : le modèle produit un rapport "
        "d'environ 100× entre les conditions extrêmes, contre ~500× mesurés. "
        "Freyer & Sutherland attribuent explicitement cet écart à des "
        "inhibiteurs produits par la nécrose, absents de ce modèle ; par "
        "ailleurs leurs volumes de saturation sont *extrapolés* par ajustement "
        "gompertzien et non mesurés directement (Kim, Park & Hunt 2008 relèvent "
        "le même désaccord). La hiérarchie des tailles finales est en revanche "
        "correctement reproduite.",
        "- **Diamètre d'apparition de la nécrose (milieu riche)** : le modèle "
        "prédit ~675 µm contre 516 µm mesurés. L'écart est cohérent avec la "
        "littérature : Casciari et al. (1992) rapportent que les modèles fondés "
        "uniquement sur les nutriments « ne prédisent pas correctement l'apparition "
        "de la nécrose », ce qui suggère l'intervention de facteurs additionnels "
        "(inhibiteurs solubles, contacts cellule-cellule) non représentés ici. "
        "Ce désaccord est conservé plutôt que masqué par un ajustement supplémentaire.",
        "- Géométrie à symétrie sphérique pour le modèle continu (les asymétries "
        "sont accessibles via le modèle à agents).",
        "- Modèle avasculaire : ni angiogenèse, ni invasion, ni réponse immunitaire.",
        "- Le spectre allélique n'est interprétable que sur le modèle de "
        "population (~10⁵ cellules) : à l'échelle du modèle spatial (~10⁴ "
        "cellules), la fenêtre de fréquences ne contient que quelques mutations. "
        "Cette limite d'échantillonnage est signalée sur les figures.",
        "- Pour observer l'évolution en quelques jours simulés, les taux de "
        "mutation des démonstrations sont amplifiés ; les valeurs mesurées "
        "(Bozic 2010) restent celles par défaut et servent aux validations.",
        "",
        "## 7. Reproductibilité",
        "",
        "```bash",
        "python run_simulation.py --validate   # suite de tests",
        "python run_simulation.py              # étude complète",
        "```",
        "",
        "Générateurs aléatoires initialisés explicitement ; solveur déterministe.",
        "",
        "## 8. Références",
        "",
        "1. Freyer JP & Sutherland RM. *Cancer Res* 46:3504-3512 (1986).",
        "2. Freyer JP & Sutherland RM. *J Cell Physiol* 124:516-524 (1985).",
        "3. Casciari JJ, Sotirchos SV & Sutherland RM. *Cancer Res* 48:3905 (1988) ; "
        "*Cell Prolif* 25:1-22 (1992).",
        "4. Grimes DR et al. *J R Soc Interface* 11:20131124 (2014) ; "
        "*PLoS ONE* 11:e0153692 (2016).",
        "5. Mueller-Klieser W & Sutherland RM. *Br J Cancer* 45:256 (1982).",
        "6. Jiang Y, Pjesivac-Grbovic J, Cantrell C & Freyer JP. "
        "*Biophys J* 89:3884-3894 (2005).",
        "7. Ward JP & King JR. *IMA J Math Appl Med Biol* 14:39-69 (1997).",
        "8. Bertuzzi A, Fasano A, Gandolfi A & Sinisgalli C. "
        "*J Theor Biol* 262:142-150 (2010).",
        "9. Drasdo D & Höhme S. *Phys Biol* 2:133-147 (2005).",
        "10. Kim S, Park S & Hunt CA. *BMC Syst Biol* 2:110 (2008).",
        "11. Alper T & Howard-Flanders P. *Nature* 178:978-979 (1956).",
        "12. Lugano R, Ramachandran M & Dimberg A. *Cell Mol Life Sci* "
        "77:1745-1770 (2019), DOI: 10.1007/s00018-019-03351-7, "
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC7190605/.",
        "13. Jain RK. *Science* 307:58-62 (2005), DOI: 10.1126/science.1104819.",
        "14. Hanahan D. *Cancer Discovery* 12:31-46 (2022), "
        "DOI: 10.1158/2159-8290.CD-21-1059.",
    ]
    txt = "\n".join(lines)
    path = _p("RAPPORT.md")
    open(path, "w").write(txt)
    return path


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="version rapide")
    ap.add_argument("--validate", action="store_true", help="tests uniquement")
    ap.add_argument("--calibrate", action="store_true", help="recalibrage")
    ap.add_argument("--days", type=float, default=None, help="durée simulée")
    ap.add_argument("--seed", type=int, default=None,
                    help="graine aléatoire (par défaut : entropie système, donc "
                         "une évolution différente à chaque exécution)")
    ap.add_argument("--evolution", action="store_true",
                    help="n'exécuter que l'étude d'évolution darwinienne")
    ap.add_argument("--vascular", action="store_true",
                    help="n'exécuter que le scénario humain fictif vascularisé")
    args = ap.parse_args()

    if args.calibrate:
        from tumorsim.calibrate import calibrate
        res = calibrate()
        print(res.report)
        json.dump(res.best, open(_p("calibration.json"), "w"), indent=2)
        return

    if args.evolution:
        study_evolution(args.days or 7.0, args.quick, args.seed)
        return

    if args.vascular:
        study_vascular(args.days or 7.0, args.quick)
        return

    if args.validate:
        checks = study_validation()
        write_report(checks, None)
        print(f"\nRapport : {_p('RAPPORT.md')}")
        return

    days = args.days or (14.0 if args.quick else 24.0)
    t_start = time.time()

    study_reference(days, args.quick)
    media = study_media(days, args.quick)
    study_therapy(days)
    study_vascular(days, args.quick)
    study_agents(min(days, 9.0), args.quick)
    study_evolution(days, args.quick, args.seed)
    checks = study_validation()
    report = write_report(checks, media)

    banner("Terminé")
    n_ok = sum(c.passed for c in checks)
    print(f"  tests réussis    : {n_ok}/{len(checks)}")
    print(f"  temps total      : {time.time()-t_start:.0f} s")
    print(f"  rapport          : {report}")
    print(f"  figures          : {RESULTS}/*.png")


if __name__ == "__main__":
    main()
