"""Adversaire IA explicable pour les modes Unity.

La politique est volontairement déterministe et lisible : elle ne consulte jamais
``truth_state``. Elle est donc une référence honnête avant l'entraînement d'une
politique ML-Agents/Sentis. Le moteur expose ses recommandations sous forme
JSON ; Unity peut les appliquer ou les afficher au joueur.
"""

from __future__ import annotations

from typing import Dict


def recommend_action(game) -> Dict:
    """Choisit une action à partir de l'observation publique uniquement.

    Heuristique : préserver le patient, réoxygéner avant d'irradier une tumeur
    large, mesurer l'incertitude périodiquement, et réduire l'angiogenèse dans
    le mode vascularisé. Les priorités sont auditablement visibles dans la
    réponse ``reason``.
    """
    obs = game.observe()
    tox = float(obs.get("toxicity_score", 0.0))
    diameter = float(obs.get("diameter_um_observed", 0.0))
    budget = float(obs.get("dose_budget_remaining_gy", 0.0))
    vascular = bool(game.cfg.vascular_enabled)
    last_biopsy = obs.get("last_biopsy")
    last_imaging = obs.get("last_po2_imaging")

    if tox >= 30.0:
        return {"type": "wait", "reason": "marge de toxicité prioritaire"}
    if vascular and diameter > 750.0 and float(obs.get("day", 0.0)) % 7.0 < 1.0:
        return {"type": "antiangiogenic", "intensity": 0.65,
                "reason": "réduire le signal vasculaire avant progression"}
    if last_imaging is None and diameter > 600.0:
        return {"type": "po2_imaging", "reason": "mesurer l'hypoxie avant la fraction"}
    if last_biopsy is None and diameter > 850.0:
        return {"type": "biopsy", "reason": "estimer l'hétérogénéité de radiosensibilité"}
    if budget >= 1.8:
        return {"type": "irradiate", "dose_gy": 1.8,
                "reason": "fraction modérée et épargne du tissu sain"}
    return {"type": "wait", "reason": "budget de dose épuisé"}
