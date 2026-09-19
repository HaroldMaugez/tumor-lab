"""
Module de vascularisation tumorale — régime humain fictif, explicitement séparé
 du sphéroïde avasculaire validé.

Ce module ne prétend pas reconstruire un réseau vasculaire 3D patient-spécifique.
Il représente le minimum mécaniste nécessaire pour quitter le régime avasculaire :

* un signal VEGF-like augmente quand l'hypoxie persiste ;
* des vaisseaux anormaux apportent O2/glucose et évacuent le lactate ;
* le recrutement est retardé, plafonné et soumis à un élagage ;
* un traitement anti-angiogénique peut réduire transitoirement l'activité.

Les formes radiales sont une fermeture phénoménologique, pas une mesure clinique.
Les niveaux et constantes sont donc marqués CALIBRÉ/SCÉNARIO dans l'API et ne
participent pas aux 32 tests EMT6/Ro du régime avasculaire.

Références biologiques : Lugano et al. 2019, doi:10.1007/s00018-019-03351-7,
PMC7190605 ; Jain 2005, doi:10.1126/science.1104819. Ces travaux justifient
l'angiogenic switch, l'hétérogénéité, la fuite et la perfusion imparfaite ; ils
ne fournissent pas un jeu universel de paramètres numériques.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .units import po2_to_conc


VASCULAR_PROVENANCE = {
    "recruitment_per_day": ("SCENARIO", "Lugano et al. 2019", "mechanism-supported rate closure"),
    "pruning_per_day": ("SCENARIO", "Lugano et al. 2019; Jain 2005", "mechanism-supported rate closure"),
    "exchange_o2_per_s": ("SCENARIO", "Lugano et al. 2019; Jain 2005", "effective exchange, not patient data"),
    "exchange_glucose_per_s": ("SCENARIO", "Lugano et al. 2019", "effective exchange, not patient data"),
    "exchange_lactate_per_s": ("SCENARIO", "Lugano et al. 2019; Jain 2005", "effective waste removal closure"),
    "vessel_po2_mmHg": ("SCENARIO", "Jain 2005", "abnormal perfusion thought experiment"),
    "vessel_glucose_mM": ("SCENARIO", "Lugano et al. 2019", "not a universal human value"),
    "vessel_lactate_mM": ("SCENARIO", "Lugano et al. 2019", "not a universal human value"),
    "abnormality": ("SCENARIO", "Jain 2005", "qualitative abnormality axis"),
}


@dataclass
class VascularConfig:
    """Paramètres de scénario, en unités SI internes du moteur.

    `exchange_*` sont des constantes d'échange effectives (/s), pas des débits
    mesurés chez un patient. Elles sont volontairement exposées pour permettre
    une analyse de sensibilité et empêcher qu'un réglage de gameplay soit pris
    pour une constante biologique.
    """

    enabled: bool = False
    initial_activity: float = 0.0
    max_activity: float = 0.78
    recruitment_per_day: float = 0.055
    pruning_per_day: float = 0.012
    hypoxia_trigger_fraction: float = 0.18
    hypoxia_softness: float = 0.08
    exchange_o2_per_s: float = 2.0e-5
    exchange_glucose_per_s: float = 6.0e-6
    exchange_lactate_per_s: float = 2.5e-6
    vessel_po2_mmHg: float = 55.0
    vessel_glucose_mM: float = 5.0
    vessel_lactate_mM: float = 1.0
    abnormality: float = 0.45


class VascularNetwork:
    """État lent d'un réseau vasculaire radial et anormal."""

    def __init__(self, config: VascularConfig | None = None):
        self.config = config or VascularConfig()
        self.activity = float(np.clip(self.config.initial_activity, 0.0, 1.0))
        self.antiangiogenic_pressure = 0.0
        self.history: list[Dict[str, float]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(self) -> None:
        self.activity = float(np.clip(self.config.initial_activity, 0.0, 1.0))
        self.antiangiogenic_pressure = 0.0
        self.history = []

    def step(self, days: float, hypoxic_fraction: float,
             mean_po2_mmHg: float) -> Dict[str, float]:
        """Fait évoluer le réseau après une période de croissance.

        La fonction logistique encode un angiogenic switch graduel, plutôt qu'un
        seuil discontinu. Le terme anti-angiogénique réduit le recrutement et
        augmente l'élagage ; ce n'est pas une simulation d'un médicament réel.
        """
        if not self.enabled:
            return self.summary()
        c = self.config
        signal = 1.0 / (1.0 + np.exp(
            -(hypoxic_fraction - c.hypoxia_trigger_fraction)
            / max(c.hypoxia_softness, 1e-6)))
        # Une pO2 moyenne très basse renforce le signal, sans le rendre infini.
        signal *= 0.65 + 0.35 * float(np.clip(1.0 - mean_po2_mmHg / 30.0, 0.0, 1.0))
        recruit = c.recruitment_per_day * signal * (1.0 - self.activity)
        prune = c.pruning_per_day * (0.35 + self.antiangiogenic_pressure) * self.activity
        self.activity = float(np.clip(self.activity + days * (recruit - prune),
                                      0.0, c.max_activity))
        self.antiangiogenic_pressure *= float(np.exp(-0.25 * days))
        self.history.append({"day_step": float(days), "signal": float(signal),
                             "activity": self.activity,
                             "antiangiogenic_pressure": self.antiangiogenic_pressure})
        return self.summary()

    def apply_antiangiogenic(self, intensity: float = 1.0) -> Dict[str, float]:
        """Applique une pression anti-angiogénique scénarisée pour un tour."""
        if self.enabled:
            self.antiangiogenic_pressure = float(np.clip(
                self.antiangiogenic_pressure + max(0.0, intensity), 0.0, 3.0))
            self.activity = float(max(0.0, self.activity - 0.035 * intensity))
        return self.summary()

    def _density(self, xi: np.ndarray) -> np.ndarray:
        """Densité relative : réseau surtout périphérique, pénétration imparfaite."""
        c = self.config
        rim = np.exp(-((1.0 - xi) / 0.30) ** 2)
        core = 0.25 * np.exp(-(xi / 0.42) ** 2)
        density = (0.10 + 0.90 * rim + core) * (1.0 - 0.35 * c.abnormality)
        return np.clip(density, 0.0, 1.0)

    def profiles(self, xi: np.ndarray) -> Dict[str, np.ndarray | float]:
        """Profils d'échange à injecter dans le solveur réaction-diffusion."""
        xi = np.asarray(xi, dtype=float)
        c = self.config
        density = self._density(xi) * self.activity
        return {
            "exchange_o2_per_s": c.exchange_o2_per_s * density,
            "exchange_glucose_per_s": c.exchange_glucose_per_s * density,
            "exchange_lactate_per_s": c.exchange_lactate_per_s * density,
            "o2_vessel_mM": float(po2_to_conc(c.vessel_po2_mmHg)),
            "glucose_vessel_mM": float(c.vessel_glucose_mM),
            "lactate_vessel_mM": float(c.vessel_lactate_mM),
            "activity": float(self.activity),
            "relative_density": density,
        }

    def provenance(self) -> Dict[str, Dict[str, str]]:
        """Retourne le registre d'évidence sans confondre source et calibration."""
        return {name: {"status": status, "source": source, "note": note}
                for name, (status, source, note) in VASCULAR_PROVENANCE.items()}

    def summary(self) -> Dict[str, float]:
        c = self.config
        return {
            "vascular_enabled": float(self.enabled),
            "vascular_activity": float(self.activity),
            "vascular_max_activity": float(c.max_activity),
            "antiangiogenic_pressure": float(self.antiangiogenic_pressure),
            "abnormality": float(c.abnormality),
        }
