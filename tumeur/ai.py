"""
Politique par Réseau de Neurones (Actor-Critic MLP) pour Tumor Lab.

Contrairement à une heuristique figée, ce module exécute un véritable réseau de
neurones multi-têtes (`PolicyValueNet`, 14 -> 64 -> 64 -> [pi(a|s), mu_dose(s), V(s)],
5 575 paramètres entraînés avec Adam sur rollouts PDE + Actor-Critic A2C ;
voir `tumeur/neural_net.py` et `train_ai.py`).

Garantie d'honnêteté scientifique (POMDP) :
- Le réseau ne lit JAMAIS `truth_state`, `n_cells` ni la vraie résistance clonale.
- Son entrée `x` (14 variables) est extraite exclusivement de `game.observe()` et
  des paramètres publics du scénario (`GameConfig`).
- L'explicabilité repose sur la rétropropagation exacte du gradient d'entrée
  (|d logit_a* / dx_i|) qui identifie les signaux diagnostiques déterminants.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .neural_net import (
    ACTIONS,
    ACTION_TO_IDX,
    FEATURE_NAMES_EN,
    FEATURE_NAMES_FR,
    PolicyValueNet,
    encode_observation,
)

WEIGHTS_PATH = Path(__file__).resolve().parent / "ai_weights.npz"

_CACHED_NET: PolicyValueNet | None = None
_CACHED_META: Dict = {}
_CACHED_MTIME: float = -1.0


def load_policy_net() -> Tuple[PolicyValueNet, Dict]:
    """Charge (et met en cache) les poids entraînés du réseau de neurones."""
    global _CACHED_NET, _CACHED_META, _CACHED_MTIME
    if WEIGHTS_PATH.exists():
        mtime = os.path.getmtime(WEIGHTS_PATH)
        if _CACHED_NET is None or mtime != _CACHED_MTIME:
            _CACHED_NET, _CACHED_META = PolicyValueNet.load(str(WEIGHTS_PATH))
            _CACHED_MTIME = mtime
        return _CACHED_NET, _CACHED_META
    if _CACHED_NET is None:
        _CACHED_NET = PolicyValueNet(seed=42)
        _CACHED_META = {"trained": False}
    return _CACHED_NET, _CACHED_META


def _build_explanations(
    chosen: str,
    dose_gy: float,
    intensity: float,
    conf: float,
    top_fr: list[str],
    top_en: list[str],
    obs: Dict,
) -> Dict[str, str]:
    sig_fr = ", ".join(s.strip() for s in top_fr[:2])
    sig_en = ", ".join(s.strip() for s in top_en[:2])
    pct = int(round(conf * 100))

    if chosen == "irradiate":
        return {
            "reason": f"réseau MLP ({pct}%) : irradier à {dose_gy:.1f} Gy [signaux : {sig_fr}]",
            "reason_en": f"neural net ({pct}%): irradiate {dose_gy:.1f} Gy [signals: {sig_en}]",
            "tip_fr": (
                f"Le réseau de neurones recommande une dose de {dose_gy:.1f} Gy "
                f"(confiance {pct}%). Ses neurones ont surtout réagi à : {sig_fr}. "
                f"Cette dose réduit la tumeur tout en gardant la toxicité sous contrôle."
            ),
            "tip_en": (
                f"The trained neural network recommends {dose_gy:.1f} Gy of radiation "
                f"({pct}% probability). Top input signals via gradient saliency: {sig_en}. "
                f"This shrinks dividing cells while protecting healthy tissue."
            ),
        }
    if chosen == "po2_imaging":
        return {
            "reason": f"réseau MLP ({pct}%) : imagerie pO₂ diagnostique [signaux : {sig_fr}]",
            "reason_en": f"neural net ({pct}%): run pO₂ oxygen scan [signals: {sig_en}]",
            "tip_fr": (
                f"Le réseau de neurones choisit une Imagerie d'oxygène pO₂ ({pct}%). "
                f"Signaux principaux : {sig_fr}. Mesurer l'hypoxie centrale permet "
                f"d'adapter les prochaines doses de rayons."
            ),
            "tip_en": (
                f"The neural network selects an Oxygen Scan (pO₂) ({pct}% probability). "
                f"Top input signals: {sig_en}. Checking whether the core lacks oxygen "
                f"(hypoxia) helps calibrate upcoming radiation doses."
            ),
        }
    if chosen == "biopsy":
        return {
            "reason": f"réseau MLP ({pct}%) : biopsie de résistance [signaux : {sig_fr}]",
            "reason_en": f"neural net ({pct}%): needle biopsy for resistance [signals: {sig_en}]",
            "tip_fr": (
                f"Le réseau de neurones demande une Biopsie ({pct}%). "
                f"Signaux principaux : {sig_fr}. Prélever un échantillon estime "
                f"la proportion de cellules devenues résistantes aux rayons."
            ),
            "tip_en": (
                f"The neural network requests a Needle Biopsy ({pct}% probability). "
                f"Top input signals: {sig_en}. Sampling cells estimates how much "
                f"radiation resistance has evolved inside the tumor."
            ),
        }
    if chosen == "antiangiogenic":
        return {
            "reason": f"réseau MLP ({pct}%) : anti-angiogénique ({intensity:.2f}) [signaux : {sig_fr}]",
            "reason_en": f"neural net ({pct}%): anti-angiogenic ({intensity:.2f}) [signals: {sig_en}]",
            "tip_fr": (
                f"Le réseau de neurones recommande un traitement anti-angiogénique "
                f"(intensité {intensity:.2f}, confiance {pct}%) suite aux signaux : {sig_fr}."
            ),
            "tip_en": (
                f"The neural network recommends anti-angiogenic therapy "
                f"(intensity {intensity:.2f}, {pct}% probability) driven by: {sig_en}."
            ),
        }
    return {
        "reason": f"réseau MLP ({pct}%) : repos thérapeutique (1 jour) [signaux : {sig_fr}]",
        "reason_en": f"neural net ({pct}%): wait 1 day (tissue recovery) [signals: {sig_en}]",
        "tip_fr": (
            f"Le réseau de neurones conseille d'attendre 1 jour ({pct}%). "
            f"Signaux principaux : {sig_fr}. Faire une pause préserve l'organe sain "
            f"ou économise le budget de dose lorsque la tumeur est déjà contenue."
        ),
        "tip_en": (
            f"The neural network advises waiting 1 day ({pct}% probability). "
            f"Top input signals: {sig_en}. Pausing spares healthy tissue or saves "
            f"radiation budget when toxicity rises or the tumor is already small."
        ),
    }


def recommend_action(game, obs: Dict | None = None) -> Dict:
    """Inférence du réseau de neurones uniquement sur l'observation publique."""
    public_obs = obs if obs is not None else game.observe()
    net, meta = load_policy_net()
    x, mask = encode_observation(public_obs, game.cfg)
    out = net.forward(x, mask)

    probs = out["probs"][0]
    best_idx = int(np.argmax(probs))
    chosen = ACTIONS[best_idx]
    conf = float(probs[best_idx])

    raw_dose = float(out["dose_gy"][0])
    max_allowed = min(
        float(game.cfg.dose_max_per_turn),
        max(0.0, float(public_obs.get("dose_budget_remaining_gy", 0.0))),
    )
    dose_gy = round(float(np.clip(raw_dose, 0.5, max(0.5, max_allowed))), 1)
    intensity = round(float(np.clip(0.35 + 0.55 * float(out["u_dose"][0]), 0.2, 1.0)), 2)
    value_est = round(float(out["value"][0]), 3)

    # Attribution par gradient d'entrée exact |d logit_a* / dx_i * x_i|
    saliency = net.input_saliency(x, mask, best_idx)
    top_indices = np.argsort(saliency)[::-1][:3]
    top_fr = [FEATURE_NAMES_FR[int(i)].strip() for i in top_indices]
    top_en = [FEATURE_NAMES_EN[int(i)].strip() for i in top_indices]

    expl = _build_explanations(chosen, dose_gy, intensity, conf, top_fr, top_en, public_obs)

    prob_dict = {ACTIONS[i]: round(float(probs[i]), 4) for i in range(len(ACTIONS))}
    result: Dict = {
        "type": chosen,
        "reason": expl["reason"],
        "reason_en": expl["reason_en"],
        "tip_fr": expl["tip_fr"],
        "tip_en": expl["tip_en"],
        "probabilities": prob_dict,
        "confidence": round(conf, 4),
        "predicted_dose_gy": dose_gy,
        "value_estimate": value_est,
        "top_signals_fr": top_fr,
        "top_signals_en": top_en,
        "model": f"PolicyValueNet ({net.in_dim}->64->64, {net.num_parameters()} params)",
        "trained": bool(meta.get("trained", False)),
    }
    if chosen == "irradiate":
        result["dose_gy"] = dose_gy
    elif chosen == "antiangiogenic":
        result["intensity"] = intensity
    return result
