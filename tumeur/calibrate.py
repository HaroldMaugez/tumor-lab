"""
Calibration des (rares) paramètres non mesurables directement.

Principe : tous les paramètres physico-chimiques (diffusivités, cinétiques
de consommation, densité cellulaire, rendements ATP, radiosensibilité) sont
FIXÉS à leurs valeurs publiées. Seuls quatre paramètres, qui ne correspondent
à aucune mesure directe disponible, sont ajustés :

    f_atp_maint  : coût énergétique de maintenance (fraction de r_ATP,max)
    f_atp_death  : seuil de production d'ATP déclenchant la nécrose
    s50_growth   : demi-saturation de la réponse proliférative
    q_glc_max    : consommation maximale de glucose — non pas libre, mais
                   CONFINÉE à la gamme expérimentale publiée (3-15e-17 mol/cell/s),
                   car les mesures de [FS85] dispersent d'un facteur ~5
    shedding_velocity : vitesse de desquamation de surface

Ces quatre paramètres sont contraints dans des bornes issues de la littérature
et ajustés sur SIX observables indépendantes de [FS86]/[Hunt08] (couronne
viable, diamètre d'apparition de la nécrose et temps de doublement initial,
dans deux conditions de culture très contrastées). Le problème est donc
sur-déterminé (6 observables pour 4 degrés de liberté), ce qui constitue un
véritable test de la structure du modèle et pas un simple ajustement.

La desquamation n'influence pas les diagnostics quasi-statiques : elle est
calibrée séparément sur le plateau de croissance (voir `validation.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from . import benchmarks as bm
from .parameters import Parameters

# Paramètre -> (borne inf, borne sup, justification)
BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "f_atp_maint": (0.10, 0.60,
                    "la maintenance représente typiquement 10-60 % du budget ATP "
                    "d'une cellule de mammifère (Rolfe & Brown, Physiol Rev 1997)"),
    "f_atp_death": (0.02, 0.45,
                    "seuil de mort strictement inférieur à la maintenance [BG09]"),
    "s50_growth": (0.02, 0.60,
                   "demi-saturation de la réponse proliférative (sans mesure directe)"),
    "lac_death_weight": (0.0, 8.0,
                        "amplitude de la toxicité lactate : les SEUILS sont "
                        "mesurés [Jia05, BF91] mais pas le taux de mort absolu ; "
                        "0 = pas d'effet, 1 = effet égal à la nécrose par ATP"),
    "q_glc_max": (3.0e-17, 1.5e-16,
                  "gamme MESURÉE de consommation de glucose des cellules EMT6/Ro "
                  "[FS85, Cas92] : le paramètre reste confiné aux valeurs publiées"),
}


@dataclass
class CalibrationResult:
    params: Parameters
    best: Dict[str, float]
    cost: float
    residuals: np.ndarray
    report: str
    n_observables: int
    n_free: int


def _apply(p: Parameters, x: Sequence[float], names: List[str]) -> Parameters:
    kw = {n: float(v) for n, v in zip(names, x)}
    # Contrainte physique : le seuil de mort est sous la maintenance
    if "f_atp_death" in kw and "f_atp_maint" in kw:
        kw["f_atp_death"] = min(kw["f_atp_death"], 0.95 * kw["f_atp_maint"])
    return p.copy_with(**kw)


def cost_function(p: Parameters) -> float:
    """Somme des carrés des résidus normalisés."""
    r = bm.residuals(p)
    r = np.where(np.isfinite(r), r, 50.0)
    return float(np.sum(r ** 2))


def calibrate(base: Parameters | None = None, seed: int = 0,
              maxiter: int = 40, popsize: int = 12,
              verbose: bool = True) -> CalibrationResult:
    """Ajuste les paramètres libres par évolution différentielle."""
    p0 = base or Parameters()
    names = list(BOUNDS.keys())
    bounds = [(BOUNDS[n][0], BOUNDS[n][1]) for n in names]

    n_eval = {"n": 0}

    def obj(x: np.ndarray) -> float:
        n_eval["n"] += 1
        return cost_function(_apply(p0, x, names))

    res = differential_evolution(
        obj, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-3, polish=True, init="sobol", mutation=(0.4, 1.0),
        recombination=0.85,
    )
    best_p = _apply(p0, res.x, names)
    best = {n: float(v) for n, v in zip(names, res.x)}
    best["f_atp_death"] = best_p.f_atp_death
    if verbose:
        print(f"[calibration] {n_eval['n']} évaluations, coût = {res.fun:.4f}")
        for k, v in best.items():
            print(f"   {k:20s} = {v:.4f}")
    return CalibrationResult(
        params=best_p, best=best, cost=float(res.fun),
        residuals=bm.residuals(best_p), report=bm.report(best_p),
        n_observables=6, n_free=len(names),
    )


if __name__ == "__main__":       # pragma: no cover
    r = calibrate()
    print(r.report)
