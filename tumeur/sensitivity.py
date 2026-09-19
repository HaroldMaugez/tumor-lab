"""Analyse de sensibilité et propagation d'incertitude.

Un modèle n'est pas « irréprochable » parce qu'il tombe juste : il l'est quand
on sait **pourquoi** il tombe juste, **quels** paramètres portent le résultat, et
**quelle** est la barre d'erreur autour de la prédiction.

Ce module fournit les trois outils qui manquaient :

1. `local_sensitivity`  — indices de sensibilité normalisés S = (∂y/∂p)·(p/y),
   calculés par différences finies centrées. Sans dimension : S = 2 signifie
   « +1 % sur le paramètre → +2 % sur l'observable ».

2. `propagate_uncertainty` — Monte-Carlo sur les incertitudes **publiées** des
   paramètres mesurés (pas des incertitudes inventées : chaque écart-type est
   tiré de la source, cf. `PARAM_UNCERTAINTY`). Donne un intervalle de crédibilité
   sur chaque observable, à comparer à la barre d'erreur expérimentale.

3. `identifiability` — matrice de corrélation des sensibilités. Deux paramètres
   dont les vecteurs de sensibilité sont colinéaires ne sont **pas** identifiables
   séparément : c'est une limite structurelle qu'il faut déclarer, pas cacher.

Références des incertitudes : voir `PARAM_UNCERTAINTY` ci-dessous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .parameters import Parameters
from . import benchmarks as bm


# ---------------------------------------------------------------------------
# Incertitudes publiées des paramètres mesurés
# ---------------------------------------------------------------------------
# Chaque entrée : (écart-type relatif, source, nature de l'incertitude).
# Les paramètres CALIBRÉS n'ont pas d'incertitude publiée : leur incertitude est
# estimée a posteriori par le profil de coût (voir `calibrate.py`), on utilise
# ici une valeur conservatrice de 15 % explicitement signalée comme telle.
PARAM_UNCERTAINTY: Dict[str, tuple] = {
    # --- paramètres MESURÉS : incertitude tirée de la publication ---
    "D_o2": (0.15, "[Kim07]/[Gri14]",
             "dispersion inter-études 1.34-1.82e-5 cm²/s selon l'état cellulaire"),
    "D_glc": (0.20, "[Cas88]",
              "méthode d'efflux de L-glucose tritié ; dispersion inter-lignées"),
    "q_o2_max": (0.12, "[Gri14]",
                 "OCR 22.1-27.4 mmHg/s selon la méthode"),
    "cell_radius": (0.07, "[FS86]",
                    "dispersion de taille cellulaire EMT6/Ro"),
    # --- paramètres CALIBRÉS : incertitude conservatrice, NON publiée ---
    "f_atp_maint": (0.15, "[CALIBRÉ]", "incertitude conservatrice, non publiée"),
    "f_atp_death": (0.15, "[CALIBRÉ]", "incertitude conservatrice, non publiée"),
    "s50_growth": (0.15, "[CALIBRÉ]", "incertitude conservatrice, non publiée"),
    "q_glc_max": (0.15, "[CALIBRÉ]", "incertitude conservatrice, non publiée"),
}

# Observables confrontées à l'expérience.
OBSERVABLES: Dict[str, Callable[[Parameters], float]] = {
    "rim_haut": lambda p: bm.diagnostics(p)["high"]["viable_rim_um"],
    "necrose_haut": lambda p: bm.diagnostics(p)["high"]["necrosis_onset_diameter_um"],
    "td_haut": lambda p: bm.diagnostics(p)["high"]["doubling_time_h"],
    "rim_bas": lambda p: bm.diagnostics(p)["low"]["viable_rim_um"],
    "necrose_bas": lambda p: bm.diagnostics(p)["low"]["necrosis_onset_diameter_um"],
    "td_bas": lambda p: bm.diagnostics(p)["low"]["doubling_time_h"],
}


@dataclass
class SensitivityResult:
    """Indices de sensibilité normalisés : observable × paramètre."""
    params: List[str]
    observables: List[str]
    S: np.ndarray            # (n_obs, n_par), sans dimension

    def top(self, observable: str, k: int = 3) -> List[tuple]:
        """Les k paramètres les plus influents sur une observable."""
        i = self.observables.index(observable)
        order = np.argsort(-np.abs(self.S[i]))[:k]
        return [(self.params[j], float(self.S[i, j])) for j in order]

    def table(self) -> str:
        w = max(len(o) for o in self.observables) + 2
        head = "observable".ljust(w) + "".join(f"{p:>14s}" for p in self.params)
        lines = [head, "-" * len(head)]
        for i, obs in enumerate(self.observables):
            row = obs.ljust(w) + "".join(f"{self.S[i, j]:>14.3f}"
                                         for j in range(len(self.params)))
            lines.append(row)
        return "\n".join(lines)


def local_sensitivity(params: Sequence[str] | None = None,
                      observables: Sequence[str] | None = None,
                      rel_step: float = 0.02,
                      base: Parameters | None = None) -> SensitivityResult:
    """Sensibilités normalisées par différences finies centrées.

    S_ij = (∂y_i/∂p_j)·(p_j/y_i) : variation relative de l'observable par
    variation relative du paramètre. `rel_step` = 2 % est un compromis entre
    erreur de troncature (grand pas) et bruit numérique (petit pas) ; les
    observables passent par une bissection de tolérance ~1e-9, donc un pas de
    2 % reste très au-dessus du bruit.
    """
    base = base or Parameters()
    params = list(params or PARAM_UNCERTAINTY.keys())
    observables = list(observables or OBSERVABLES.keys())

    y0 = np.array([OBSERVABLES[o](base) for o in observables])
    S = np.zeros((len(observables), len(params)))

    for j, name in enumerate(params):
        p0 = getattr(base, name)
        h = abs(p0) * rel_step
        y_plus = np.array([OBSERVABLES[o](base.copy_with(**{name: p0 + h}))
                           for o in observables])
        y_minus = np.array([OBSERVABLES[o](base.copy_with(**{name: p0 - h}))
                            for o in observables])
        with np.errstate(divide="ignore", invalid="ignore"):
            S[:, j] = np.where(np.abs(y0) > 0,
                               (y_plus - y_minus) / (2 * h) * p0 / y0, np.nan)
    return SensitivityResult(params, observables, S)


@dataclass
class UncertaintyResult:
    """Distribution des observables sous l'incertitude des paramètres."""
    observables: List[str]
    samples: np.ndarray      # (n_draws, n_obs)
    n_draws: int

    def summary(self) -> Dict[str, dict]:
        out = {}
        for i, o in enumerate(self.observables):
            col = self.samples[:, i]
            col = col[np.isfinite(col)]
            if col.size == 0:
                out[o] = {"median": np.nan, "lo": np.nan, "hi": np.nan,
                          "cv_pct": np.nan, "n_valid": 0}
                continue
            out[o] = {
                "median": float(np.median(col)),
                "lo": float(np.percentile(col, 2.5)),
                "hi": float(np.percentile(col, 97.5)),
                "cv_pct": float(100 * np.std(col) / abs(np.mean(col))),
                "n_valid": int(col.size),
            }
        return out


def propagate_uncertainty(n_draws: int = 200, seed: int | None = 0,
                          base: Parameters | None = None) -> UncertaintyResult:
    """Monte-Carlo log-normal sur les incertitudes publiées des paramètres.

    Loi log-normale (et non normale) : ces paramètres sont des grandeurs
    strictement positives dont l'incertitude est multiplicative — un
    coefficient de diffusion ne peut pas devenir négatif.
    """
    base = base or Parameters()
    rng = np.random.default_rng(seed)
    names = list(PARAM_UNCERTAINTY.keys())
    observables = list(OBSERVABLES.keys())

    rows = []
    for _ in range(n_draws):
        kw = {}
        for nm in names:
            sd_rel = PARAM_UNCERTAINTY[nm][0]
            sigma = np.sqrt(np.log(1 + sd_rel ** 2))     # log-normale de CV=sd_rel
            kw[nm] = getattr(base, nm) * float(rng.lognormal(-sigma**2 / 2, sigma))
        # contrainte physique du modèle : le seuil de mort est sous la maintenance
        kw["f_atp_death"] = min(kw["f_atp_death"], 0.95 * kw["f_atp_maint"])
        try:
            p = base.copy_with(**kw)
            rows.append([OBSERVABLES[o](p) for o in observables])
        except Exception:
            rows.append([np.nan] * len(observables))

    return UncertaintyResult(observables, np.asarray(rows), n_draws)


def identifiability(sens: SensitivityResult) -> Dict[str, object]:
    """Détecte les paramètres non identifiables séparément.

    Deux paramètres dont les colonnes de sensibilité sont colinéaires produisent
    le même effet sur toutes les observables : les données ne peuvent pas les
    séparer. On renvoie la matrice de corrélation et les paires problématiques
    (|r| > 0.95), ainsi que le conditionnement de la matrice de sensibilité.
    """
    S = np.nan_to_num(sens.S)
    norms = np.linalg.norm(S, axis=0)
    keep = norms > 1e-12
    Sn = S[:, keep] / norms[keep]
    corr = Sn.T @ Sn
    names = [n for n, k in zip(sens.params, keep) if k]

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > 0.95:
                pairs.append((names[i], names[j], float(corr[i, j])))

    sv = np.linalg.svd(Sn, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else np.inf
    return {"names": names, "corr": corr, "collinear_pairs": pairs,
            "condition_number": cond,
            "unidentifiable": [n for n, k in zip(sens.params, keep) if not k]}
