"""
Thérapie : radiothérapie (modèle linéaire-quadratique avec effet oxygène).

Modèle LQ
---------
Fraction survivante après une dose d :
    SF = exp(-alpha_eff d - beta_eff d^2)

Effet oxygène (OER), Alper & Howard-Flanders (1956) :
la radiosensibilité dépend de la pression partielle locale d'O2 selon une
cinétique de type Michaelis-Menten

    OER(pO2) = (OER_max * pO2 + K) / (pO2 + K)      avec K ~ 3 mmHg

qui vaut 1 en anoxie (pO2 = 0) et tend vers OER_max ~ 3 en normoxie.

SENS PHYSIQUE (essentiel) : les cellules HYPOXIQUES sont RADIORÉSISTANTES.
L'oxygène « fixe » les lésions radicalaires de l'ADN ; en son absence, elles
sont chimiquement restaurées. Il faut donc une dose OER fois plus grande en
anoxie pour obtenir le même effet qu'en normoxie.

Les paramètres alpha et beta étant tabulés en conditions AÉROBIES, le facteur
modificateur de dose local est le rapport de la sensibilité locale à la
sensibilité maximale (oxique) :

    f(pO2) = OER(pO2) / OER_max        (f = 1/3 en anoxie, f -> 1 en normoxie)
    d_eff  = f(pO2) * d

d'où

    SF = exp(-alpha * d_eff - beta * d_eff^2)

Ainsi le coeur hypoxique survit préférentiellement, ce qui reproduit la
radiorésistance des tumeurs mal oxygénées observée en clinique.

Les cellules quiescentes sont moins radiosensibles (facteur documenté ~0.5-0.8
sur alpha) : le paramètre `quiescent_radioresistance` module alpha et beta.

Fractionnement
--------------
Un schéma est une liste d'instants (jours) et de doses (Gy) — par exemple
5 x 2 Gy/semaine. À chaque fraction, la fraction survivante est appliquée à la
fraction volumique de cellules vivantes ; les cellules tuées basculent dans le
compartiment nécrotique (mort mitotique, résorbée par la lyse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .parameters import Parameters
from .units import SECONDS_PER_DAY, conc_to_po2


def oer(po2_mmHg: np.ndarray, p: Parameters) -> np.ndarray:
    """Rapport d'amélioration par l'oxygène (Oxygen Enhancement Ratio)."""
    po2 = np.maximum(po2_mmHg, 0.0)
    return (p.oer_max * po2 + p.k_oer) / (po2 + p.k_oer)


def survival_fraction(dose_gy: float, po2_mmHg: np.ndarray,
                      p: Parameters, quiescent_frac: np.ndarray | None = None
                      ) -> np.ndarray:
    """Fraction survivante locale au modèle LQ corrigé de l'OER."""
    # Facteur modificateur de dose : 1/OER_max en anoxie, 1 en normoxie.
    d_eff = dose_gy * oer(po2_mmHg, p) / p.oer_max
    a, b = p.alpha_lq, p.beta_lq
    if quiescent_frac is not None:
        f = np.clip(quiescent_frac, 0.0, 1.0)
        scale = 1.0 - f * (1.0 - p.quiescent_radioresistance)
        a = a * scale
        b = b * scale
    return np.exp(-a * d_eff - b * d_eff ** 2)


@dataclass
class RadiotherapySchedule:
    """Schéma de fractionnement : (instant en jours, dose en Gy)."""
    fractions: List[Tuple[float, float]] = field(default_factory=list)
    _done: set = field(default_factory=set, repr=False)

    @classmethod
    def conventional(cls, start_day: float, dose_per_fraction: float = 2.0,
                     n_fractions: int = 10, days_per_week: int = 5
                     ) -> "RadiotherapySchedule":
        """Schéma conventionnel : n fractions, 5 jours/semaine, week-end libre."""
        frac: List[Tuple[float, float]] = []
        day = start_day
        given = 0
        while given < n_fractions:
            weekday = int(day - start_day) % 7
            if weekday < days_per_week:
                frac.append((day, dose_per_fraction))
                given += 1
            day += 1.0
        return cls(frac)

    @classmethod
    def single(cls, day: float, dose: float) -> "RadiotherapySchedule":
        return cls([(day, dose)])

    def total_dose(self) -> float:
        return sum(d for _, d in self.fractions)

    def bed(self, alpha_beta: float = 10.0) -> float:
        """Dose biologiquement efficace (BED) du schéma."""
        return sum(d * (1.0 + d / alpha_beta) for _, d in self.fractions)

    # ------------------------------------------------------------------
    def make_hook(self, params: Parameters, log: list | None = None):
        """Renvoie un hook à brancher sur SpheroidSimulation.therapy_hook."""
        def hook(sim) -> None:
            t_day = sim.t / SECONDS_PER_DAY
            for i, (day, dose) in enumerate(self.fractions):
                if i in self._done or t_day < day:
                    continue
                self._done.add(i)
                sol, k_p, k_d = sim._rates()
                po2 = conc_to_po2(sol.c_o2)
                quiescent = 1.0 - np.clip(k_p / max(params.k_prolif_max, 1e-30), 0.0, 1.0)
                sf = survival_fraction(dose, po2, params, quiescent)
                before = float(np.sum(sim.phi))
                sim.phi = np.clip(sim.phi * sf, 0.0, 1.0)
                if log is not None:
                    log.append({
                        "day": t_day, "dose_gy": dose,
                        "mean_SF": float(np.mean(sf)),
                        "min_SF": float(np.min(sf)), "max_SF": float(np.max(sf)),
                        "phi_before": before, "phi_after": float(np.sum(sim.phi)),
                        "radius_um": sim.R,
                    })
        return hook
