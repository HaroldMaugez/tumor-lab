"""
Réseau de neurones multi-têtes (Actor-Critic / Policy-Value MLP) en NumPy pur.

Architecture
------------
Entrée  : vecteur d'observation clinique publique x in R^14 (aucun accès à truth_state)
Couche 1: Linear(14 -> 64) + tanh
Couche 2: Linear(64 -> 64) + tanh
Têtes   :
  1. Politique discrète pi(a | x) : Linear(64 -> 5) + masque d'actions + Softmax
     Actions : ("wait", "irradiate", "biopsy", "po2_imaging", "antiangiogenic")
  2. Dose continue mu_dose(x)     : Linear(64 -> 1) + sigmoïde -> [0.5, 5.0] Gy
  3. Critique V(x)                : Linear(64 -> 1) (espérance du retour normalisé)

Entraînement en deux phases (voir `train_ai.py`) :
  - Phase 1 : Apprentissage supervisé sur rollouts contrefactuels du simulateur PDE
              (Cross-Entropy + MSE dose + MSE valeur + régularisation L2, optimiseur Adam).
  - Phase 2 : Raffinement par apprentissage par renforcement Actor-Critic (A2C)
              avec avantage standardisé, régularisation entropique et gradient clipping.

Vérification numérique : `check_gradients()` compare la rétropropagation analytique
aux différences finies centrées (erreur relative < 1e-6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

ACTIONS: Tuple[str, ...] = (
    "wait",
    "irradiate",
    "biopsy",
    "po2_imaging",
    "antiangiogenic",
)
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}

DOSE_MIN_GY: float = 0.5
DOSE_MAX_GY: float = 5.0
N_FEATURES: int = 14

FEATURE_NAMES_FR: Tuple[str, ...] = (
    " avancement du protocole (jour / max)",
    "jours restants normalisés",
    "diamètre observé au scanner (mm)",
    "écart au seuil de contrôle (400 µm)",
    "toxicité tissu sain (/ 50)",
    "alerte clinique de toxicité (/ 5)",
    "budget de dose restant",
    "sensibilité de l'organe (sparing)",
    "mode tumeur vascularisée",
    "activité vasculaire observée",
    "imagerie pO₂ récente disponible",
    "fraction hypoxique mesurée",
    "biopsie récente disponible",
    "résistance clonale estimée (%)",
)

FEATURE_NAMES_EN: Tuple[str, ...] = (
    "protocol time progress",
    "remaining days ratio",
    "observed scan diameter (mm)",
    "distance above 400 µm control goal",
    "healthy organ toxicity (/ 50)",
    "early clinical toxicity warning",
    "remaining radiation budget",
    "organ radiation sensitivity",
    "vascular tumor regime",
    "observed vascular activity",
    "recent pO₂ oxygen scan available",
    "measured hypoxic core fraction",
    "recent needle biopsy available",
    "estimated clonal resistance (%)",
)


def encode_observation(obs: Dict, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """Encode uniquement l'observation PUBLIQUE bruitée et les règles connues.

    Ne lit JAMAIS `truth_state`, `n_cells` ni `resistance_pct` réel.
    Retourne `(x, valid_action_mask)` de tailles `(14,)` et `(5,)`.
    """
    max_days = max(float(getattr(cfg, "max_days", 70.0)), 1.0)
    day = float(obs.get("day", 0.0))
    days_rem = float(obs.get("days_remaining", max_days - day))
    diam_um = float(obs.get("diameter_um_observed", 500.0))
    ctrl_um = float(getattr(cfg, "control_diameter_um", 400.0))
    lose_tox = max(float(getattr(cfg, "lose_toxicity", 50.0)), 1.0)
    warn_tox = max(float(getattr(cfg, "warn_toxicity", 5.0)), 1.0)
    tox = float(obs.get("toxicity_score", 0.0))
    budget_max = max(float(getattr(cfg, "dose_budget_gy", 60.0)), 1.0)
    budget_rem = max(float(obs.get("dose_budget_remaining_gy", 0.0)), 0.0)
    sparing = float(getattr(cfg, "sparing_factor", 0.6))
    vasc_enabled = 1.0 if bool(getattr(cfg, "vascular_enabled", False)) else 0.0
    vasc_act = float(obs.get("vascular_activity", 0.0))

    last_po2 = obs.get("last_po2_imaging")
    if isinstance(last_po2, dict):
        age_po2 = max(0.0, day - float(last_po2.get("day", day)))
        has_po2 = float(np.exp(-age_po2 / 14.0))
        hyp_frac = float(np.clip(last_po2.get("hypoxic_fraction", 0.0), 0.0, 1.0))
    else:
        has_po2 = 0.0
        hyp_frac = 0.0

    last_bio = obs.get("last_biopsy")
    if isinstance(last_bio, dict):
        age_bio = max(0.0, day - float(last_bio.get("day", day)))
        has_bio = float(np.exp(-age_bio / 21.0))
        res_est = float(np.clip(last_bio.get("resistance_pct_estimated", 0.0) / 100.0, -0.2, 1.0))
    else:
        has_bio = 0.0
        res_est = 0.0

    x = np.array([
        np.clip(day / max_days, 0.0, 1.5),
        np.clip(days_rem / max_days, 0.0, 1.0),
        np.clip(diam_um / 1000.0, 0.0, 2.5),
        np.clip((diam_um - ctrl_um) / 1000.0, -0.5, 2.0),
        np.clip(tox / lose_tox, 0.0, 1.5),
        np.clip(tox / (3.0 * warn_tox), 0.0, 1.5),
        np.clip(budget_rem / budget_max, 0.0, 1.0),
        np.clip(sparing, 0.0, 1.5),
        vasc_enabled,
        np.clip(vasc_act, 0.0, 1.0),
        has_po2,
        hyp_frac,
        has_bio,
        res_est,
    ], dtype=np.float64)

    mask = np.ones(len(ACTIONS), dtype=np.float64)
    if budget_rem < 0.25:
        mask[ACTION_TO_IDX["irradiate"]] = 0.0
    if not bool(getattr(cfg, "vascular_enabled", False)):
        mask[ACTION_TO_IDX["antiangiogenic"]] = 0.0

    return x, mask


def unit_to_dose_gy(u: float | np.ndarray) -> float | np.ndarray:
    return DOSE_MIN_GY + (DOSE_MAX_GY - DOSE_MIN_GY) * u


def dose_gy_to_unit(d: float | np.ndarray) -> float | np.ndarray:
    return np.clip((d - DOSE_MIN_GY) / (DOSE_MAX_GY - DOSE_MIN_GY), 0.0, 1.0)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z_clip))


def _masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Log-softmax et probabilités numériquement stables sous masque binaire."""
    masked = np.where(mask > 0.5, logits, -1e9)
    m = np.max(masked, axis=-1, keepdims=True)
    exp_z = np.where(mask > 0.5, np.exp(masked - m), 0.0)
    sum_exp = np.maximum(np.sum(exp_z, axis=-1, keepdims=True), 1e-30)
    probs = exp_z / sum_exp
    log_probs = np.where(mask > 0.5, (masked - m) - np.log(sum_exp), -1e9)
    return log_probs, probs


class PolicyValueNet:
    """MLP Actor-Critic à 2 couches cachées avec rétropropagation analytique exacte."""

    PARAM_KEYS = (
        "W1", "b1",
        "W2", "b2",
        "W_act", "b_act",
        "W_dose", "b_dose",
        "W_val", "b_val",
    )

    def __init__(self, in_dim: int = N_FEATURES, hidden1: int = 64,
                 hidden2: int = 64, n_actions: int = len(ACTIONS),
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.n_actions = n_actions

        # Initialisation Xavier / Glorot adaptée à tanh
        self.params: Dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / (in_dim + hidden1)), (in_dim, hidden1)),
            "b1": np.zeros(hidden1, dtype=np.float64),
            "W2": rng.normal(0.0, np.sqrt(2.0 / (hidden1 + hidden2)), (hidden1, hidden2)),
            "b2": np.zeros(hidden2, dtype=np.float64),
            "W_act": rng.normal(0.0, np.sqrt(1.0 / hidden2), (hidden2, n_actions)),
            "b_act": np.zeros(n_actions, dtype=np.float64),
            "W_dose": rng.normal(0.0, np.sqrt(1.0 / hidden2), (hidden2, 1)),
            "b_dose": np.zeros(1, dtype=np.float64),
            "W_val": rng.normal(0.0, np.sqrt(1.0 / hidden2), (hidden2, 1)),
            "b_val": np.zeros(1, dtype=np.float64),
        }

    def num_parameters(self) -> int:
        return int(sum(arr.size for arr in self.params.values()))

    def forward(self, X: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, np.ndarray]:
        """Passe avant vectorisée. X: (B, 14) ou (14,)."""
        single = (X.ndim == 1)
        Xb = X[None, :] if single else X
        B = Xb.shape[0]
        if mask is None:
            mb = np.ones((B, self.n_actions), dtype=np.float64)
        else:
            mb = mask[None, :] if mask.ndim == 1 else mask

        p = self.params
        z1 = Xb @ p["W1"] + p["b1"]
        h1 = np.tanh(z1)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.tanh(z2)

        logits = h2 @ p["W_act"] + p["b_act"]
        log_probs, probs = _masked_log_softmax(logits, mb)

        z_dose = (h2 @ p["W_dose"] + p["b_dose"])[:, 0]
        u_dose = _sigmoid(z_dose)
        dose_gy = unit_to_dose_gy(u_dose)

        val = (h2 @ p["W_val"] + p["b_val"])[:, 0]

        out = {
            "X": Xb, "mask": mb,
            "z1": z1, "h1": h1,
            "z2": z2, "h2": h2,
            "logits": logits, "log_probs": log_probs, "probs": probs,
            "z_dose": z_dose, "u_dose": u_dose, "dose_gy": dose_gy,
            "value": val,
        }
        return out

    def _backprop_trunk(self, cache: Dict[str, np.ndarray],
                        d_logits: np.ndarray,
                        d_z_dose: np.ndarray,
                        d_val: np.ndarray,
                        l2_reg: float) -> Dict[str, np.ndarray]:
        """Rétropropagation analytique des 3 têtes vers W1, b1, W2, b2."""
        p = self.params
        Xb = cache["X"]
        h1 = cache["h1"]
        h2 = cache["h2"]

        d_z_dose_col = d_z_dose[:, None]
        d_val_col = d_val[:, None]

        grads: Dict[str, np.ndarray] = {}
        grads["W_act"] = h2.T @ d_logits + l2_reg * p["W_act"]
        grads["b_act"] = np.sum(d_logits, axis=0)

        grads["W_dose"] = h2.T @ d_z_dose_col + l2_reg * p["W_dose"]
        grads["b_dose"] = np.sum(d_z_dose_col, axis=0)

        grads["W_val"] = h2.T @ d_val_col + l2_reg * p["W_val"]
        grads["b_val"] = np.sum(d_val_col, axis=0)

        d_h2 = (
            d_logits @ p["W_act"].T
            + d_z_dose_col @ p["W_dose"].T
            + d_val_col @ p["W_val"].T
        )
        d_z2 = d_h2 * (1.0 - h2 ** 2)

        grads["W2"] = h1.T @ d_z2 + l2_reg * p["W2"]
        grads["b2"] = np.sum(d_z2, axis=0)

        d_h1 = d_z2 @ p["W2"].T
        d_z1 = d_h1 * (1.0 - h1 ** 2)

        grads["W1"] = Xb.T @ d_z1 + l2_reg * p["W1"]
        grads["b1"] = np.sum(d_z1, axis=0)
        return grads

    def loss_and_grads_supervised(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        target_actions: np.ndarray,
        target_u_dose: np.ndarray,
        dose_mask: np.ndarray,
        target_values: np.ndarray,
        c_dose: float = 2.0,
        c_val: float = 0.5,
        l2_reg: float = 1e-4,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Calcule la loss supervisée (CE + MSE dose + MSE critique + L2) et ses gradients exacts."""
        cache = self.forward(X, mask)
        B = X.shape[0]
        log_probs = cache["log_probs"]
        probs = cache["probs"]
        u_dose = cache["u_dose"]
        val = cache["value"]

        # 1. Cross-Entropy sur les actions discrètes
        idx = np.arange(B)
        ce_loss = float(-np.mean(log_probs[idx, target_actions]))
        d_logits = probs.copy()
        d_logits[idx, target_actions] -= 1.0
        d_logits /= B

        # 2. MSE sur la dose continue (uniquement quand l'action cible est 'irradiate')
        denom_dose = float(np.sum(dose_mask)) + 1e-8
        diff_dose = (u_dose - target_u_dose) * dose_mask
        dose_loss = float(0.5 * c_dose * np.sum(diff_dose ** 2) / denom_dose)
        d_u_dose = c_dose * diff_dose / denom_dose
        d_z_dose = d_u_dose * u_dose * (1.0 - u_dose)

        # 3. MSE sur la valeur d'état (critique)
        diff_val = val - target_values
        val_loss = float(0.5 * c_val * np.mean(diff_val ** 2))
        d_val = (c_val / B) * diff_val

        # 4. Régularisation L2 sur les matrices de poids
        l2_sum = sum(float(np.sum(self.params[k] ** 2)) for k in ("W1", "W2", "W_act", "W_dose", "W_val"))
        l2_loss = 0.5 * l2_reg * l2_sum

        total_loss = ce_loss + dose_loss + val_loss + l2_loss
        grads = self._backprop_trunk(cache, d_logits, d_z_dose, d_val, l2_reg)

        acc = float(np.mean(np.argmax(probs, axis=1) == target_actions))
        metrics = {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "dose_loss": dose_loss,
            "val_loss": val_loss,
            "l2_loss": l2_loss,
            "accuracy": acc,
        }
        return metrics, grads

    def loss_and_grads_a2c(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        sampled_actions: np.ndarray,
        sampled_u_dose: np.ndarray,
        dose_mask: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        dose_std: float = 0.12,
        c_val: float = 0.5,
        c_ent: float = 0.015,
        l2_reg: float = 1e-4,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Loss Actor-Critic (Policy Gradient + Entropie + Critique MSE + L2) et gradients exacts."""
        cache = self.forward(X, mask)
        B = X.shape[0]
        idx = np.arange(B)
        log_probs = cache["log_probs"]
        probs = cache["probs"]
        u_dose = cache["u_dose"]
        val = cache["value"]

        # 1. Policy Gradient discret : -1/B sum A_i log pi(a_i | s_i)
        pg_act_loss = float(-np.mean(advantages * log_probs[idx, sampled_actions]))
        onehot = np.zeros_like(probs)
        onehot[idx, sampled_actions] = 1.0
        d_logits_pg = (advantages[:, None] * (probs - onehot)) / B

        # 2. Entropie de la politique : H(pi) = -sum_k pi_k log pi_k
        safe_log_p = np.where(mask > 0.5, log_probs, 0.0)
        ent_per_sample = -np.sum(probs * safe_log_p, axis=1)
        entropy = float(np.mean(ent_per_sample))
        ent_loss = -c_ent * entropy
        # d(-c_ent * mean(H)) / dz_k = +(c_ent / B) * pi_k * (log pi_k + H)
        d_logits_ent = (c_ent / B) * probs * (safe_log_p + ent_per_sample[:, None])
        d_logits = d_logits_pg + d_logits_ent

        # 3. Policy Gradient continu sur la dose gaussienne (quand action == irradiate)
        var = dose_std ** 2
        logp_dose = -0.5 * ((sampled_u_dose - u_dose) ** 2) / var
        pg_dose_loss = float(-np.mean(advantages * dose_mask * logp_dose))
        d_u_dose = -(advantages * dose_mask * (sampled_u_dose - u_dose)) / (B * var)
        d_z_dose = d_u_dose * u_dose * (1.0 - u_dose)

        # 4. Critique MSE : 0.5 * c_val * mean((V - G)^2)
        diff_val = val - returns
        val_loss = float(0.5 * c_val * np.mean(diff_val ** 2))
        d_val = (c_val / B) * diff_val

        # 5. L2
        l2_sum = sum(float(np.sum(self.params[k] ** 2)) for k in ("W1", "W2", "W_act", "W_dose", "W_val"))
        l2_loss = 0.5 * l2_reg * l2_sum

        total_loss = pg_act_loss + pg_dose_loss + ent_loss + val_loss + l2_loss
        grads = self._backprop_trunk(cache, d_logits, d_z_dose, d_val, l2_reg)

        return {
            "loss": total_loss,
            "pg_act_loss": pg_act_loss,
            "pg_dose_loss": pg_dose_loss,
            "entropy": entropy,
            "val_loss": val_loss,
        }, grads

    def input_saliency(self, x: np.ndarray, mask: np.ndarray, action_idx: int) -> np.ndarray:
        """Importance exacte |d(logit[action_idx])/dx_i * x_i| par rétropropagation jusqu'à l'entrée."""
        cache = self.forward(x, mask)
        p = self.params
        d_logit = np.zeros((1, self.n_actions), dtype=np.float64)
        d_logit[0, action_idx] = 1.0
        d_h2 = d_logit @ p["W_act"].T
        d_z2 = d_h2 * (1.0 - cache["h2"] ** 2)
        d_h1 = d_z2 @ p["W2"].T
        d_z1 = d_h1 * (1.0 - cache["h1"] ** 2)
        d_x = (d_z1 @ p["W1"].T)[0]
        # Attribution gradient * magnitude (avec plancher pour détecter aussi les absences d'examen)
        scale = np.maximum(np.abs(x), 0.25)
        return np.abs(d_x) * scale

    def save(self, path: str, metadata: Dict | None = None) -> None:
        payload = dict(self.params)
        if metadata is not None:
            import json
            payload["_metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> Tuple["PolicyValueNet", Dict]:
        data = np.load(path, allow_pickle=False)
        W1 = data["W1"]
        W2 = data["W2"]
        W_act = data["W_act"]
        net = cls(in_dim=W1.shape[0], hidden1=W1.shape[1],
                  hidden2=W2.shape[1], n_actions=W_act.shape[1], seed=0)
        for k in cls.PARAM_KEYS:
            net.params[k] = np.array(data[k], dtype=np.float64)
        meta = {}
        if "_metadata_json" in data:
            import json
            meta = json.loads(str(data["_metadata_json"].item()))
        return net, meta


class AdamOptimizer:
    """Optimiseur Adam standard avec correction de biais et clipping de norme globale."""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 max_grad_norm: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> float:
        total_norm_sq = sum(float(np.sum(g ** 2)) for g in grads.values())
        grad_norm = float(np.sqrt(total_norm_sq))
        clip_coef = min(1.0, self.max_grad_norm / (grad_norm + 1e-12))

        self.t += 1
        b1_corr = 1.0 - self.beta1 ** self.t
        b2_corr = 1.0 - self.beta2 ** self.t

        for k, p in params.items():
            g = grads[k] * clip_coef
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[k] / b1_corr
            v_hat = self.v[k] / b2_corr
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return grad_norm


def check_gradients(seed: int = 7, eps: float = 1e-6) -> Dict[str, float]:
    """Vérifie analytiquement tous les gradients (supervisé ET A2C) par différences finies."""
    rng = np.random.default_rng(seed)
    net = PolicyValueNet(in_dim=N_FEATURES, hidden1=12, hidden2=10, seed=seed)
    B = 6
    X = rng.normal(0.0, 0.7, (B, N_FEATURES))
    mask = np.ones((B, len(ACTIONS)), dtype=np.float64)
    mask[:2, 4] = 0.0  # teste aussi le masquage d'actions invalides
    target_actions = rng.integers(0, 4, size=B)
    target_u_dose = rng.uniform(0.1, 0.9, size=B)
    dose_mask = (target_actions == ACTION_TO_IDX["irradiate"]).astype(np.float64)
    dose_mask[0] = 1.0
    target_values = rng.normal(0.0, 1.0, size=B)
    advantages = rng.normal(0.0, 1.0, size=B)

    _, grads_sup = net.loss_and_grads_supervised(
        X, mask, target_actions, target_u_dose, dose_mask, target_values
    )
    max_rel_sup = 0.0
    for k in net.PARAM_KEYS:
        arr = net.params[k]
        g_ana = grads_sup[k]
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _ = net.loss_and_grads_supervised(
                X, mask, target_actions, target_u_dose, dose_mask, target_values
            )
            arr[idx] = orig - eps
            lm, _ = net.loss_and_grads_supervised(
                X, mask, target_actions, target_u_dose, dose_mask, target_values
            )
            arr[idx] = orig
            g_num = (lp["loss"] - lm["loss"]) / (2.0 * eps)
            rel = abs(g_ana[idx] - g_num) / max(1e-8, abs(g_ana[idx]) + abs(g_num))
            max_rel_sup = max(max_rel_sup, float(rel))
            it.iternext()

    _, grads_a2c = net.loss_and_grads_a2c(
        X, mask, target_actions, target_u_dose, dose_mask, target_values, advantages
    )
    max_rel_a2c = 0.0
    for k in net.PARAM_KEYS:
        arr = net.params[k]
        g_ana = grads_a2c[k]
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            lp, _ = net.loss_and_grads_a2c(
                X, mask, target_actions, target_u_dose, dose_mask, target_values, advantages
            )
            arr[idx] = orig - eps
            lm, _ = net.loss_and_grads_a2c(
                X, mask, target_actions, target_u_dose, dose_mask, target_values, advantages
            )
            arr[idx] = orig
            g_num = (lp["loss"] - lm["loss"]) / (2.0 * eps)
            rel = abs(g_ana[idx] - g_num) / max(1e-8, abs(g_ana[idx]) + abs(g_num))
            max_rel_a2c = max(max_rel_a2c, float(rel))
            it.iternext()

    return {
        "supervised_max_rel_error": max_rel_sup,
        "a2c_max_rel_error": max_rel_a2c,
        "passed": bool(max_rel_sup < 1e-5 and max_rel_a2c < 1e-5),
    }
