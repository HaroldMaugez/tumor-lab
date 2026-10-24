"""Adversaire IA explicable pour les modes Unity et assistant pédagogique web.

La politique est volontairement déterministe et lisible : elle ne consulte jamais
``truth_state``. Elle est donc une référence honnête avant l'entraînement d'une
politique ML-Agents/Sentis. Le moteur expose ses recommandations sous forme
JSON ; Unity et le site web peuvent les appliquer ou les expliquer au joueur.
"""

from __future__ import annotations

from typing import Dict


def recommend_action(game) -> Dict:
    """Choisit une action à partir de l'observation publique uniquement.

    Heuristique : préserver le patient, réoxygéner avant d'irradier une tumeur
    large, mesurer l'incertitude périodiquement, et réduire l'angiogenèse dans
    le mode vascularisé. Les priorités sont auditablement visibles dans la
    réponse ``reason`` (et ``reason_en`` / ``tip_*`` pour le mode débutant).
    """
    obs = game.observe()
    tox = float(obs.get("toxicity_score", 0.0))
    diameter = float(obs.get("diameter_um_observed", 0.0))
    budget = float(obs.get("dose_budget_remaining_gy", 0.0))
    vascular = bool(game.cfg.vascular_enabled)
    last_biopsy = obs.get("last_biopsy")
    last_imaging = obs.get("last_po2_imaging")

    if tox >= 30.0:
        return {
            "type": "wait",
            "reason": "marge de toxicité prioritaire",
            "reason_en": "healthy organ safety comes first (high toxicity)",
            "tip_fr": "L'organe sain commence à souffrir (toxicité ≥ 30/50). Faire une pause d'un jour évite de franchir le seuil critique de 50.",
            "tip_en": "The surrounding healthy organ is stressed (toxicity ≥ 30/50). Resting for a day avoids crossing the lethal limit of 50.",
        }
    if vascular and diameter > 750.0 and float(obs.get("day", 0.0)) % 7.0 < 1.0:
        return {
            "type": "antiangiogenic",
            "intensity": 0.65,
            "reason": "réduire le signal vasculaire avant progression",
            "reason_en": "calm abnormal blood vessels before irradiating",
            "tip_fr": "La tumeur dépasse 750 µm avec des vaisseaux actifs : une dose modérée d'anti-angiogénique freine son alimentation sanguine.",
            "tip_en": "The tumor is above 750 µm with active blood vessels: a moderate anti-angiogenic step slows down its blood supply.",
        }
    if last_imaging is None and diameter > 600.0:
        return {
            "type": "po2_imaging",
            "reason": "mesurer l'hypoxie avant la fraction",
            "reason_en": "run an oxygen scan (pO₂) to check for hidden hypoxia",
            "tip_fr": "La tumeur est assez grosse (> 600 µm) pour manquer d'oxygène au centre. L'imagerie pO₂ permet de diagnostiquer cette hypoxie.",
            "tip_en": "At > 600 µm, the tumor center often lacks oxygen (hypoxia), making radiation less effective. An oxygen scan diagnoses this first.",
        }
    if last_biopsy is None and diameter > 850.0:
        return {
            "type": "biopsy",
            "reason": "estimer l'hétérogénéité de radiosensibilité",
            "reason_en": "perform a biopsy to detect radiation-resistant cells",
            "tip_fr": "La tumeur est volumineuse (> 850 µm) et aucune biopsie n'a été faite : prélever 60 cellules permet d'estimer le % de résistance.",
            "tip_en": "The tumor is large (> 850 µm) with no biopsy yet: sampling 60 cells estimates what % of cells resist radiation.",
        }
    if budget >= 1.8:
        return {
            "type": "irradiate",
            "dose_gy": 1.8,
            "reason": "fraction modérée et épargne du tissu sain",
            "reason_en": "apply a gentle 1.8 Gy radiation fraction to shrink the tumor safely",
            "tip_fr": "Une petite dose régulière (1.8 Gy, proche des 2 Gy utilisés à l'hôpital) attaque les cellules tumorales tout en ménageant l'organe sain.",
            "tip_en": "A gentle daily dose (1.8 Gy, close to standard 2 Gy hospital sessions) kills dividing tumor cells while sparing healthy tissue.",
        }
    return {
        "type": "wait",
        "reason": "budget de dose épuisé",
        "reason_en": "radiation dose budget is exhausted — observe evolution",
        "tip_fr": "Il ne reste plus assez de budget de radiation : on observe l'évolution de la tumeur.",
        "tip_en": "Your total radiation budget is used up: wait and monitor whether the tumor stays controlled.",
    }
