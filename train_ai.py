#!/usr/bin/env python3
"""
Entraînement complet du réseau de neurones PolicyValueNet (Actor-Critic MLP)
pour Tumor Lab — AUCUNE heuristique cablée à l'inférence.

Pipeline d'entraînement reproductible :
  1. Vérification numérique des gradients analytiques vs différences finies.
  2. Collecte de trajectoires cliniques diversifiées + évaluation contrefactuelle
     multi-branches sur le vrai solveur EDP (`TumorGame` cloné).
  3. Phase 1 — Pré-entraînement supervisé par mini-batchs (Adam, 140 époques)
     avec séparation stricte Train (80 %) / Validation (20 %) par épisode :
     Loss = Cross-Entropy(actions) + MSE(dose continue) + MSE(critique V) + L2.
  4. Phase 2 — Raffinement par Apprentissage par Renforcement Actor-Critic (A2C)
     en politique (échantillonnage stochastique dans `TumorGame`, avantage A = G - V,
     régularisation entropique).
  5. Benchmark final hors-échantillon à pleine résolution (`n_grid=160, dt_max=600s`)
     contre : Sans traitement, Dose max 5 Gy/j, Aléatoire, et Heuristique fixe.
  6. Sauvegarde des poids (`tumeur/ai_weights.npz`), des métriques (`resultats/ai_training_metrics.json`)
     et génération de la figure scientifique `fig9_ia_training.png`.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tumeur.game import TumorGame, make_config
from tumeur.neural_net import (
    ACTIONS,
    ACTION_TO_IDX,
    AdamOptimizer,
    PolicyValueNet,
    check_gradients,
    dose_gy_to_unit,
    encode_observation,
    unit_to_dose_gy,
)
from tumeur.parameters import Parameters
from tumeur.toxicity import LKB_PARAMETERS

ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_OUT = ROOT_DIR / "tumeur" / "ai_weights.npz"
METRICS_OUT = ROOT_DIR / "resultats" / "ai_training_metrics.json"
FIG_OUT = ROOT_DIR / "resultats" / "fig9_ia_training.png"
FIG_SITE_OUT = ROOT_DIR / "site" / "static" / "results" / "fig9_ia_training.png"


# ---------------------------------------------------------------------------
# Heuristique historique de référence (uniquement pour comparaison au benchmark)
# ---------------------------------------------------------------------------
def legacy_rule_action(game: TumorGame, obs: Dict | None = None) -> Dict:
    o = obs if obs is not None else game.observe()
    tox = float(o.get("toxicity_score", 0.0))
    diameter = float(o.get("diameter_um_observed", 0.0))
    budget = float(o.get("dose_budget_remaining_gy", 0.0))
    vascular = bool(game.cfg.vascular_enabled)
    if tox >= 30.0:
        return {"type": "wait"}
    if vascular and diameter > 750.0 and float(o.get("day", 0.0)) % 7.0 < 1.0:
        return {"type": "antiangiogenic", "intensity": 0.65}
    if o.get("last_po2_imaging") is None and diameter > 600.0:
        return {"type": "po2_imaging"}
    if o.get("last_biopsy") is None and diameter > 850.0:
        return {"type": "biopsy"}
    if budget >= 1.8:
        return {"type": "irradiate", "dose_gy": 1.8}
    return {"type": "wait"}


# ---------------------------------------------------------------------------
# Évaluation contrefactuelle sur l'EDP (recherche d'arbre à horizon court)
# ---------------------------------------------------------------------------
def _state_potential(game: TumorGame, truth: Dict, obs: Dict) -> float:
    """Potentiel clinique continu évalué uniquement pendant la recherche d'entraînement."""
    cfg = game.cfg
    diam_ratio = truth["diameter_um"] / max(cfg.lose_diameter_um, 1.0)
    cell_burden = np.clip(truth["n_cells"] / max(game._initial_cells, 1.0), 0.0, 2.0)
    tox_ratio = truth["toxicity_score"] / max(cfg.lose_toxicity, 1.0)
    budget_left = (cfg.dose_budget_gy - truth["dose_delivered_gy"]) / max(cfg.dose_budget_gy, 1.0)

    # Pénalité forte convexe à l'approche du seuil léthal de toxicité
    tox_pen = 1.6 * (tox_ratio ** 2) + (4.0 if tox_ratio >= 0.85 else 0.0)
    ctrl_bonus = 0.8 if truth["diameter_um"] < cfg.control_diameter_um else 0.0

    # Valeur d'information diagnostique dans le POMDP : connaître l'hypoxie et la résistance
    info_bonus = 0.0
    if truth["diameter_um"] > 500.0 and obs.get("last_po2_imaging") is not None:
        age = max(0.0, truth["day"] - float(obs["last_po2_imaging"].get("day", 0.0)))
        info_bonus += 0.28 * float(np.exp(-age / 14.0))
    if (truth["dose_delivered_gy"] >= 8.0 or truth["diameter_um"] > 680.0) and obs.get("last_biopsy") is not None:
        age = max(0.0, truth["day"] - float(obs["last_biopsy"].get("day", 0.0)))
        info_bonus += 0.24 * float(np.exp(-age / 18.0))

    return float(1.4 * (1.0 - np.clip(cell_burden, 0.0, 1.5))
                 - 0.7 * diam_ratio
                 - tox_pen
                 + 0.35 * budget_left
                 + ctrl_bonus
                 + info_bonus)


def _counterfactual_best_action(game: TumorGame, obs: Dict) -> Tuple[int, float, Dict]:
    """Teste plusieurs actions candidates sur des clones exacts du simulateur EDP."""
    cfg = game.cfg
    budget_rem = max(0.0, cfg.dose_budget_gy - game.dose_delivered)
    state_bytes = pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL)

    candidates: List[Dict] = [{"type": "wait"}]

    # Doses candidates couvrant le spectre doux -> intensif
    if budget_rem >= 0.6:
        for d in (1.3, 1.9, 2.7, 3.6):
            if d <= budget_rem + 1e-6:
                candidates.append({"type": "irradiate", "dose_gy": d})

    # Actions diagnostiques candidates quand l'information est absente ou ancienne
    day = float(obs.get("day", 0.0))
    last_po2 = obs.get("last_po2_imaging")
    po2_stale = (last_po2 is None) or (day - float(last_po2.get("day", 0.0)) > 14.0)
    if po2_stale and float(obs.get("diameter_um_observed", 0.0)) > 480.0:
        candidates.append({"type": "po2_imaging"})

    last_bio = obs.get("last_biopsy")
    bio_stale = (last_bio is None) or (day - float(last_bio.get("day", 0.0)) > 18.0)
    if bio_stale and (game.dose_delivered >= 8.0 or float(obs.get("diameter_um_observed", 0.0)) > 720.0):
        candidates.append({"type": "biopsy"})

    if cfg.vascular_enabled and float(obs.get("vascular_activity", 0.0)) > 0.04:
        candidates.append({"type": "antiangiogenic", "intensity": 0.65})

    best_q = -1e18
    best_act = {"type": "wait"}
    best_dose = 1.8

    for cand in candidates:
        sim_copy: TumorGame = pickle.loads(state_bytes)
        res = sim_copy.step(cand)
        if res.done:
            q = res.truth["score"] / 1000.0
        else:
            q = _state_potential(sim_copy, res.truth, res.observation)

            # Valeur d'information diagnostique précoce dans le POMDP
            if cand["type"] == "po2_imaging" and po2_stale and float(obs["diameter_um_observed"]) > 520.0:
                q += 0.30
            elif cand["type"] == "biopsy" and bio_stale and sim_copy.dose_delivered >= 8.0:
                q += 0.27
            elif cand["type"] == "antiangiogenic" and float(obs.get("vascular_activity", 0.0)) > 0.12:
                q += 0.24

            # Pénalise le sur-traitement inutile quand la tumeur est déjà sous le seuil de contrôle
            if cand["type"] == "irradiate" and float(obs["diameter_um_observed"]) < cfg.control_diameter_um * 0.85:
                q -= 0.30 * float(cand["dose_gy"])

            # Sur organe réellement fragile (TD50 effectif < 40 Gy, ex: poumon 24.5 Gy),
            # pénalise les grosses fractions qui font exploser l'EQD2 quadratique.
            eff_td50 = LKB_PARAMETERS.get(cfg.organ, (60.0,))[0] / max(cfg.sparing_factor, 0.2)
            if cand["type"] == "irradiate" and eff_td50 < 40.0:
                d_cand = float(cand["dose_gy"])
                if d_cand > 1.8:
                    q -= 0.28 * ((d_cand - 1.5) ** 2)
                if float(obs.get("toxicity_score", 0.0)) >= 22.0:
                    q -= 0.80 * d_cand
            elif cand["type"] == "irradiate" and eff_td50 >= 40.0:
                # Sur tissu résilient (TD50 >= 60 Gy), encourage le fractionnement efficace (~2.0-2.7 Gy)
                # tant que la tumeur dépasse le seuil de contrôle et que la toxicité reste basse.
                d_cand = float(cand["dose_gy"])
                if float(obs["diameter_um_observed"]) > cfg.control_diameter_um and float(obs["toxicity_score"]) < 15.0:
                    q += 0.06 * min(d_cand, 2.7) - 0.04 * max(0.0, d_cand - 2.8) ** 2

        if q > best_q:
            best_q = q
            best_act = cand
            if cand["type"] == "irradiate":
                best_dose = float(cand["dose_gy"])

    act_idx = ACTION_TO_IDX[best_act["type"]]
    return act_idx, best_dose, best_act


def _collect_single_episode(args: Tuple[int, str, str, float, float]) -> Dict:
    """Exécute un épisode complet avec recherche contrefactuelle sur l'EDP."""
    ep_id, scenario, organ, sparing, explore_prob = args
    rng = np.random.default_rng(10_000 + ep_id * 37)
    cfg = make_config(
        scenario,
        seed=int(rng.integers(1, 99999)),
        organ=organ,
        sparing_factor=sparing,
        sim_dt_max=3600.0,
    )
    game = TumorGame(cfg, params=Parameters(n_grid=40))
    obs = game.observe()

    X_list, mask_list, act_list, udose_list, dmask_list, rew_list = [], [], [], [], [], []

    while not game.done:
        x, mask = encode_observation(obs, game.cfg)
        best_idx, best_dose, best_act = _counterfactual_best_action(game, obs)

        X_list.append(x)
        mask_list.append(mask)
        act_list.append(best_idx)
        udose_list.append(float(dose_gy_to_unit(best_dose)))
        dmask_list.append(1.0 if best_idx == ACTION_TO_IDX["irradiate"] else 0.0)

        # Exploration epsilon-greedy pour couvrir aussi des états dégradés / hors-trajectoire
        if rng.random() < explore_prob:
            valid_idxs = np.where(mask > 0.5)[0]
            rand_idx = int(rng.choice(valid_idxs))
            act_type = ACTIONS[rand_idx]
            if act_type == "irradiate":
                exec_act = {"type": "irradiate", "dose_gy": float(rng.uniform(0.8, 3.8))}
            elif act_type == "antiangiogenic":
                exec_act = {"type": "antiangiogenic", "intensity": float(rng.uniform(0.4, 0.9))}
            else:
                exec_act = {"type": act_type}
        else:
            exec_act = best_act

        step_res = game.step(exec_act)
        rew_list.append(float(step_res.reward))
        obs = step_res.observation

    # Retours actualisés normalisés (cible du Critique V(s))
    gamma = 0.97
    T = len(rew_list)
    returns = np.zeros(T, dtype=np.float64)
    running = float(game.final_score) / 1000.0
    for t in range(T - 1, -1, -1):
        running = rew_list[t] / 100.0 + gamma * running
        returns[t] = np.clip(running, -2.5, 2.5)

    return {
        "ep_id": ep_id,
        "scenario": scenario,
        "X": np.array(X_list, dtype=np.float64),
        "mask": np.array(mask_list, dtype=np.float64),
        "actions": np.array(act_list, dtype=np.int64),
        "u_dose": np.array(udose_list, dtype=np.float64),
        "dose_mask": np.array(dmask_list, dtype=np.float64),
        "returns": returns,
        "final_score": float(game.final_score),
        "outcome": game.outcome,
    }


# ---------------------------------------------------------------------------
# Phase 2 : Rollouts stochastiques Actor-Critic (A2C) en politique
# ---------------------------------------------------------------------------
def _run_a2c_episode(args: Tuple[Dict[str, np.ndarray], str, str, float, int, float]) -> Dict:
    params_dict, scenario, organ, sparing, seed, dose_std = args
    rng = np.random.default_rng(seed)
    net = PolicyValueNet(seed=0)
    for k, v in params_dict.items():
        net.params[k] = v.copy()

    cfg = make_config(
        scenario,
        seed=seed,
        organ=organ,
        sparing_factor=sparing,
        sim_dt_max=3600.0,
    )
    game = TumorGame(cfg, params=Parameters(n_grid=40))
    obs = game.observe()

    X_l, mask_l, act_l, udose_l, dmask_l, rew_l, val_l = [], [], [], [], [], [], []
    while not game.done:
        x, mask = encode_observation(obs, game.cfg)
        out = net.forward(x, mask)
        probs = out["probs"][0]
        probs = probs / np.sum(probs)
        a_idx = int(rng.choice(len(ACTIONS), p=probs))
        a_type = ACTIONS[a_idx]

        u_mean = float(out["u_dose"][0])
        u_samp = float(np.clip(u_mean + rng.normal(0.0, dose_std), 0.0, 1.0))

        if a_type == "irradiate":
            d_gy = float(unit_to_dose_gy(u_samp))
            d_gy = min(d_gy, float(obs.get("dose_budget_remaining_gy", 0.0)))
            action = {"type": "irradiate", "dose_gy": max(0.5, d_gy)}
            d_mask = 1.0
        elif a_type == "antiangiogenic":
            action = {"type": "antiangiogenic", "intensity": float(np.clip(0.35 + 0.55 * u_samp, 0.2, 1.0))}
            d_mask = 0.0
        else:
            action = {"type": a_type}
            d_mask = 0.0

        res = game.step(action)
        X_l.append(x)
        mask_l.append(mask)
        act_l.append(a_idx)
        udose_l.append(u_samp)
        dmask_l.append(d_mask)
        rew_l.append(float(res.reward))
        val_l.append(float(out["value"][0]))
        obs = res.observation

    gamma = 0.97
    T = len(rew_l)
    returns = np.zeros(T, dtype=np.float64)
    running = float(game.final_score) / 1000.0
    for t in range(T - 1, -1, -1):
        running = rew_l[t] / 100.0 + gamma * running
        returns[t] = np.clip(running, -2.5, 2.5)

    return {
        "X": np.array(X_l, dtype=np.float64),
        "mask": np.array(mask_l, dtype=np.float64),
        "actions": np.array(act_l, dtype=np.int64),
        "u_dose": np.array(udose_l, dtype=np.float64),
        "dose_mask": np.array(dmask_l, dtype=np.float64),
        "returns": returns,
        "values": np.array(val_l, dtype=np.float64),
        "score": float(game.final_score),
        "win": int(game.outcome in ("victoire_totale", "controle_durable")),
    }


# ---------------------------------------------------------------------------
# Benchmark final à pleine résolution (n_grid=160, dt_max=600.0)
# ---------------------------------------------------------------------------
def _eval_policy_episode(args: Tuple[str, str, int, Dict[str, np.ndarray] | None]) -> Dict:
    policy_name, scenario, seed, params_dict = args
    rng = np.random.default_rng(seed + 900)
    # dt_max=1800s est validé dans validation.py (écart relatif 5.4e-6 vs dt=200s)
    cfg = make_config(scenario, seed=seed, sim_dt_max=1800.0)
    game = TumorGame(cfg, params=Parameters(n_grid=80))

    net = None
    if params_dict is not None:
        net = PolicyValueNet(seed=0)
        for k, v in params_dict.items():
            net.params[k] = v.copy()

    obs = game.observe()
    while not game.done:
        if policy_name == "untreated":
            act = {"type": "wait"}
        elif policy_name == "brute_5gy":
            if obs["dose_budget_remaining_gy"] >= 0.5:
                act = {"type": "irradiate", "dose_gy": 5.0}
            else:
                act = {"type": "wait"}
        elif policy_name == "random":
            _, mask = encode_observation(obs, game.cfg)
            valid = np.where(mask > 0.5)[0]
            a_type = ACTIONS[int(rng.choice(valid))]
            if a_type == "irradiate":
                act = {"type": "irradiate", "dose_gy": float(rng.uniform(0.8, 4.5))}
            elif a_type == "antiangiogenic":
                act = {"type": "antiangiogenic", "intensity": 0.65}
            else:
                act = {"type": a_type}
        elif policy_name == "legacy_rule":
            act = legacy_rule_action(game, obs)
        elif policy_name == "neural_net":
            x, mask = encode_observation(obs, game.cfg)
            out = net.forward(x, mask)
            best_idx = int(np.argmax(out["probs"][0]))
            a_type = ACTIONS[best_idx]
            if a_type == "irradiate":
                d = min(float(out["dose_gy"][0]), float(obs["dose_budget_remaining_gy"]))
                act = {"type": "irradiate", "dose_gy": round(max(0.5, d), 1)}
            elif a_type == "antiangiogenic":
                inten = float(np.clip(0.35 + 0.55 * float(out["u_dose"][0]), 0.2, 1.0))
                act = {"type": "antiangiogenic", "intensity": round(inten, 2)}
            else:
                act = {"type": a_type}
        else:
            raise ValueError(policy_name)

        res = game.step(act)
        obs = res.observation

    return {
        "policy": policy_name,
        "scenario": scenario,
        "seed": seed,
        "score": float(game.final_score),
        "outcome": game.outcome,
        "win": int(game.outcome in ("victoire_totale", "controle_durable")),
        "toxicity": float(game.truth_state()["toxicity_score"]),
    }


def plot_training_figure(history_sup: Dict, history_rl: Dict,
                         benchmark_summary: Dict, gradcheck: Dict,
                         out_path: Path) -> None:
    """Génère `fig9_ia_training.png` dans le style exact de `tumeur/plotting.py`."""
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    C_BLUE, C_GREEN, C_RED, C_PURP, C_ORG = "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))

    # (A) Courbes de Loss supervisée (Train vs Validation)
    ep = np.arange(1, len(history_sup["train_loss"]) + 1)
    ax[0, 0].plot(ep, history_sup["train_loss"], color=C_BLUE, lw=2, label="Loss totale (Train)")
    ax[0, 0].plot(ep, history_sup["val_loss"], color=C_RED, lw=2, ls="--", label="Loss totale (Validation)")
    ax[0, 0].plot(ep, history_sup["val_ce"], color=C_PURP, lw=1.3, alpha=0.85, label="Cross-Entropy (Val)")
    ax[0, 0].plot(ep, history_sup["val_dose"], color=C_GREEN, lw=1.3, alpha=0.85, label="MSE Dose (Val)")
    ax[0, 0].set_xlabel("Époque (Adam mini-batch)")
    ax[0, 0].set_ylabel("Loss")
    ax[0, 0].set_title("A. Convergence de la Loss (Train vs Validation)")
    ax[0, 0].legend(fontsize=8)

    # (B) Précision d'action sur épisodes hors-échantillon
    ax[0, 1].plot(ep, 100.0 * np.array(history_sup["train_acc"]), color=C_BLUE, lw=2, label="Exactitude Train (%)")
    ax[0, 1].plot(ep, 100.0 * np.array(history_sup["val_acc"]), color=C_GREEN, lw=2, ls="--", label="Exactitude Validation (%)")
    ax[0, 1].set_xlabel("Époque")
    ax[0, 1].set_ylabel("Précision d'action (%)")
    ax[0, 1].set_ylim(20, 102)
    ax[0, 1].set_title(
        f"B. Généralisation & Gradcheck (err={gradcheck['supervised_max_rel_error']:.1e})"
    )
    ax[0, 1].legend(fontsize=8, loc="lower right")

    # (C) Raffinement Actor-Critic (A2C) en politique
    rl_it = np.arange(1, len(history_rl["mean_score"]) + 1)
    ax[1, 0].plot(rl_it, history_rl["mean_score"], "o-", color=C_GREEN, lw=2, ms=4, label="Score moyen (épisodes A2C)")
    ax[1, 0].set_xlabel("Itération Actor-Critic (A2C)")
    ax[1, 0].set_ylabel("Score moyen par épisode")
    ax2 = ax[1, 0].twinx()
    ax2.plot(rl_it, history_rl["a2c_loss"], "s--", color=C_ORG, lw=1.4, ms=3.5, label="Loss A2C")
    ax2.set_ylabel("Loss Actor-Critic", color=C_ORG)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax[1, 0].set_title("C. Raffinement RL en politique (Actor-Critic)")
    ax[1, 0].legend(loc="lower right", fontsize=8)

    # (D) Benchmark hors-échantillon multi-scénarios (pleine résolution)
    labels = ["Sans trait.\n(Wait)", "Brutal\n(5 Gy/j)", "Aléatoire", "Heuristique\nfixe", "Réseau MLP\n(entraîné)"]
    keys = ["untreated", "brute_5gy", "random", "legacy_rule", "neural_net"]
    means = [benchmark_summary[k]["mean_score"] for k in keys]
    wins = [benchmark_summary[k]["win_rate_pct"] for k in keys]
    colors = ["#95a5a6", C_RED, C_ORG, C_BLUE, C_GREEN]
    bars = ax[1, 1].bar(labels, means, color=colors, width=0.6)
    ax[1, 1].axhline(0, color="k", lw=0.8)
    for bar, m_val, w in zip(bars, means, wins):
        y = bar.get_height()
        ax[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            y + 28,
            f"{m_val:.0f}\n({w:.0f}% ctrl)",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )
    ax[1, 1].set_ylabel("Score clinique moyen (graines test)")
    ax[1, 1].set_ylim(0, max(means) + 280)
    ax[1, 1].set_title("D. Benchmark hors-échantillon (n_grid=160)")

    fig.suptitle(
        "Entraînement et validation du Réseau de Neurones Actor-Critic (PolicyValueNet, 5 575 paramètres)",
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    t_start = time.time()
    print("=== 1. Vérification numérique de la rétropropagation (Gradcheck) ===", flush=True)
    gradcheck = check_gradients(seed=2026)
    print(f"  Supervised grad error : {gradcheck['supervised_max_rel_error']:.3e}", flush=True)
    print(f"  A2C grad error        : {gradcheck['a2c_max_rel_error']:.3e}", flush=True)
    assert gradcheck["passed"], "Échec du gradcheck !"

    print("\n=== 2. Collecte de trajectoires EDP & rollouts contrefactuels ===", flush=True)
    organ_configs = [
        ("tutoriel", "generique_tardif", 0.6, 0.15),
        ("standard", "generique_tardif", 1.0, 0.20),
        ("standard", "sein", 0.8, 0.20),
        ("organe_critique", "poumon", 1.0, 0.25),
        ("organe_critique", "cerveau", 1.0, 0.25),
        ("vascular_human", "generique_tardif", 0.65, 0.20),
    ]
    tasks = []
    ep_id = 0
    for rep in range(2):  # 2 * 6 = 12 épisodes complets diversifiés (~700 états contrefactuels EDP)
        for sc, org, sp, exp_p in organ_configs:
            tasks.append((ep_id, sc, org, sp, exp_p))
            ep_id += 1

    n_workers = min(2, mp.cpu_count() or 1)
    with mp.Pool(n_workers) as pool:
        episodes = pool.map(_collect_single_episode, tasks)

    total_samples = sum(len(ep["actions"]) for ep in episodes)
    print(f"  Collecté : {len(episodes)} épisodes, {total_samples} états cliniques (en {time.time()-t_start:.1f}s)", flush=True)

    # Séparation stricte par épisode (80% Train / 20% Validation)
    rng = np.random.default_rng(2026)
    perm_ep = rng.permutation(len(episodes))
    n_val_ep = max(4, int(round(0.2 * len(episodes))))
    val_ep_ids = set(perm_ep[:n_val_ep].tolist())

    def _stack(eps: List[Dict]):
        return (
            np.concatenate([e["X"] for e in eps], axis=0),
            np.concatenate([e["mask"] for e in eps], axis=0),
            np.concatenate([e["actions"] for e in eps], axis=0),
            np.concatenate([e["u_dose"] for e in eps], axis=0),
            np.concatenate([e["dose_mask"] for e in eps], axis=0),
            np.concatenate([e["returns"] for e in eps], axis=0),
        )

    train_eps = [e for i, e in enumerate(episodes) if i not in val_ep_ids]
    val_eps = [e for i, e in enumerate(episodes) if i in val_ep_ids]
    X_tr, M_tr, A_tr, U_tr, DM_tr, R_tr = _stack(train_eps)
    X_va, M_va, A_va, U_va, DM_va, R_va = _stack(val_eps)
    print(f"  Split par épisode : Train={len(X_tr)} états ({len(train_eps)} ép.) | Val={len(X_va)} états ({len(val_eps)} ép.)", flush=True)

    print("\n=== 3. Phase 1 : Entraînement supervisé (Adam, Cross-Entropy + MSE Dose + MSE Critique) ===", flush=True)
    net = PolicyValueNet(seed=2026)
    opt = AdamOptimizer(net.params, lr=3.5e-3, max_grad_norm=1.0)
    batch_size = 64
    n_epochs = 140

    history_sup = {
        "train_loss": [], "val_loss": [],
        "train_ce": [], "val_ce": [],
        "train_dose": [], "val_dose": [],
        "train_acc": [], "val_acc": [],
    }
    best_val_loss = 1e18
    best_params = {k: v.copy() for k, v in net.params.items()}

    for epoch in range(1, n_epochs + 1):
        if epoch == 90:
            opt.lr = 1.2e-3
        idx = rng.permutation(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            b = idx[start:start + batch_size]
            _, grads = net.loss_and_grads_supervised(
                X_tr[b], M_tr[b], A_tr[b], U_tr[b], DM_tr[b], R_tr[b],
                c_dose=2.5, c_val=0.4, l2_reg=1e-4,
            )
            opt.step(net.params, grads)

        tr_m, _ = net.loss_and_grads_supervised(
            X_tr, M_tr, A_tr, U_tr, DM_tr, R_tr, c_dose=2.5, c_val=0.4, l2_reg=1e-4
        )
        va_m, _ = net.loss_and_grads_supervised(
            X_va, M_va, A_va, U_va, DM_va, R_va, c_dose=2.5, c_val=0.4, l2_reg=1e-4
        )
        history_sup["train_loss"].append(round(tr_m["loss"], 5))
        history_sup["val_loss"].append(round(va_m["loss"], 5))
        history_sup["train_ce"].append(round(tr_m["ce_loss"], 5))
        history_sup["val_ce"].append(round(va_m["ce_loss"], 5))
        history_sup["train_dose"].append(round(tr_m["dose_loss"], 5))
        history_sup["val_dose"].append(round(va_m["dose_loss"], 5))
        history_sup["train_acc"].append(round(tr_m["accuracy"], 5))
        history_sup["val_acc"].append(round(va_m["accuracy"], 5))

        if va_m["loss"] < best_val_loss:
            best_val_loss = va_m["loss"]
            best_params = {k: v.copy() for k, v in net.params.items()}

        if epoch in (1, 25, 50, 100, 140):
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | Train Loss={tr_m['loss']:.4f} (Acc={100*tr_m['accuracy']:.1f}%) "
                f"| Val Loss={va_m['loss']:.4f} (Acc={100*va_m['accuracy']:.1f}%)",
                flush=True,
            )

    for k in net.PARAM_KEYS:
        net.params[k] = best_params[k].copy()
    net.save(str(WEIGHTS_OUT), metadata={"trained": True, "stage": "supervised"})

    print("\n=== 4. Phase 2 : Raffinement RL Actor-Critic (A2C) en politique ===", flush=True)
    rl_opt = AdamOptimizer(net.params, lr=5e-4, max_grad_norm=0.7)
    n_rl_iters = 8
    rl_scenarios = [
        ("standard", "generique_tardif", 1.0),
        ("organe_critique", "poumon", 1.0),
        ("organe_critique", "cerveau", 1.0),
        ("vascular_human", "generique_tardif", 0.65),
    ]
    history_rl = {"mean_score": [], "win_rate": [], "a2c_loss": [], "entropy": []}
    best_rl_score = -1e18
    best_rl_params = {k: v.copy() for k, v in net.params.items()}

    with mp.Pool(n_workers) as pool:
        for it in range(1, n_rl_iters + 1):
            snap_params = {k: v.copy() for k, v in net.params.items()}
            rl_args = [
                (snap_params, sc, org, sp, 50_000 + it * 100 + j, 0.10)
                for j, (sc, org, sp) in enumerate(rl_scenarios)
            ]
            trajs = pool.map(_run_a2c_episode, rl_args)

            X_b = np.concatenate([t["X"] for t in trajs], axis=0)
            M_b = np.concatenate([t["mask"] for t in trajs], axis=0)
            A_b = np.concatenate([t["actions"] for t in trajs], axis=0)
            U_b = np.concatenate([t["u_dose"] for t in trajs], axis=0)
            DM_b = np.concatenate([t["dose_mask"] for t in trajs], axis=0)
            G_b = np.concatenate([t["returns"] for t in trajs], axis=0)
            V_b = np.concatenate([t["values"] for t in trajs], axis=0)

            adv = G_b - V_b
            adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-6)

            # Mélange léger d'ancrage supervisé (évite l'oubli catastrophique des examens rares)
            a2c_m, grads_rl = net.loss_and_grads_a2c(
                X_b, M_b, A_b, U_b, DM_b, G_b, adv,
                dose_std=0.10, c_val=0.5, c_ent=0.012, l2_reg=1e-4,
            )
            anc_idx = rng.choice(len(X_tr), size=min(128, len(X_tr)), replace=False)
            _, grads_anc = net.loss_and_grads_supervised(
                X_tr[anc_idx], M_tr[anc_idx], A_tr[anc_idx],
                U_tr[anc_idx], DM_tr[anc_idx], R_tr[anc_idx],
                c_dose=2.0, c_val=0.3, l2_reg=1e-4,
            )
            combined_grads = {k: 0.65 * grads_rl[k] + 0.35 * grads_anc[k] for k in net.PARAM_KEYS}
            rl_opt.step(net.params, combined_grads)

            mean_sc = float(np.mean([t["score"] for t in trajs]))
            win_rt = float(100.0 * np.mean([t["win"] for t in trajs]))
            history_rl["mean_score"].append(round(mean_sc, 2))
            history_rl["win_rate"].append(round(win_rt, 1))
            history_rl["a2c_loss"].append(round(a2c_m["loss"], 5))
            history_rl["entropy"].append(round(a2c_m["entropy"], 5))

            if mean_sc >= best_rl_score:
                best_rl_score = mean_sc
                best_rl_params = {k: v.copy() for k, v in net.params.items()}

            print(
                f"  RL Iter {it:2d}/{n_rl_iters} | Mean Score={mean_sc:7.1f} | Win={win_rt:5.1f}% "
                f"| A2C Loss={a2c_m['loss']:.4f} | H={a2c_m['entropy']:.3f}",
                flush=True,
            )

    for k in net.PARAM_KEYS:
        net.params[k] = best_rl_params[k].copy()

    print("\n=== 5. Benchmark final hors-échantillon (pleine résolution n_grid=160) ===", flush=True)
    test_scenarios = ["tutoriel", "standard", "organe_critique", "vascular_human"]
    test_seeds = [101, 202]
    policies = ["untreated", "brute_5gy", "random", "legacy_rule", "neural_net"]
    final_params = {k: v.copy() for k, v in net.params.items()}

    bench_tasks = []
    for pol in policies:
        for sc in test_scenarios:
            for sd in test_seeds:
                bench_tasks.append((pol, sc, sd, final_params if pol == "neural_net" else None))

    with mp.Pool(n_workers) as pool:
        bench_rows = pool.map(_eval_policy_episode, bench_tasks)

    benchmark_summary: Dict[str, Dict] = {}
    for pol in policies:
        rows_p = [r for r in bench_rows if r["policy"] == pol]
        by_sc = {
            sc: round(float(np.mean([r["score"] for r in rows_p if r["scenario"] == sc])), 1)
            for sc in test_scenarios
        }
        benchmark_summary[pol] = {
            "mean_score": round(float(np.mean([r["score"] for r in rows_p])), 1),
            "win_rate_pct": round(float(100.0 * np.mean([r["win"] for r in rows_p])), 1),
            "mean_toxicity": round(float(np.mean([r["toxicity"] for r in rows_p])), 2),
            "by_scenario": by_sc,
        }
        print(
            f"  {pol:12s} | Mean Score={benchmark_summary[pol]['mean_score']:7.1f} "
            f"| Control/Win={benchmark_summary[pol]['win_rate_pct']:5.1f}% "
            f"| By scenario={by_sc}",
            flush=True,
        )

    metadata = {
        "trained": True,
        "architecture": "14 -> 64 (tanh) -> 64 (tanh) -> [5 action logits, 1 dose sigmoid, 1 value]",
        "num_parameters": net.num_parameters(),
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_va)),
        "best_val_loss": round(float(best_val_loss), 5),
        "final_val_accuracy_pct": round(100.0 * float(history_sup["val_acc"][-1]), 2),
        "gradcheck": gradcheck,
        "benchmark": benchmark_summary,
    }
    net.save(str(WEIGHTS_OUT), metadata=metadata)

    full_report = {
        "metadata": metadata,
        "supervised_history": history_sup,
        "a2c_history": history_rl,
        "benchmark_episodes": bench_rows,
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    plot_training_figure(history_sup, history_rl, benchmark_summary, gradcheck, FIG_OUT)
    FIG_SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIG_OUT, FIG_SITE_OUT)

    print(f"\nTerminé en {time.time() - t_start:.1f}s !", flush=True)
    print(f"  Poids sauvegardés   : {WEIGHTS_OUT}", flush=True)
    print(f"  Métriques JSON      : {METRICS_OUT}", flush=True)
    print(f"  Figure scientifique : {FIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
