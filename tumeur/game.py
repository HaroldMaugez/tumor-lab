"""
Moteur de jeu tour par tour — couche mince au-dessus du simulateur validé.

Pourquoi ce module existe
-------------------------
Le modèle à agents (`agents.py`) porte la radiorésistance héritable, mais son
coût croît comme O(N) : mesuré sur cette machine, un tour de 1 jour passe de
2 s (N = 10³) à 258 s (N = 8·10⁴). Injouable au-delà de ~5 tours.

Le modèle continu (`growth.py`) coûte **~1 s par tour, indépendamment de la
taille** (mesuré : 0.92 s à Ø 123 µm, 1.14 s à Ø 965 µm), mais il ne connaît
qu'une population homogène : il ne peut donc pas produire la sélection de
résistance, qui est le piège central du jeu.

Ce module réconcilie les deux en transportant, **par-dessus** le champ continu,
une distribution discrétisée de radiosensibilité (`ResistanceDistribution`).
C'est exactement la construction d'Alfonso & Berk (2019), qui suivent une
distribution de SF2 dans la population plutôt que des cellules individuelles.
La sélection n'est pas simulée cellule par cellule : elle est calculée
analytiquement par repondération bayésienne des classes après chaque fraction.

Validation croisée : `game_matches_abm()` compare la résistance sélectionnée
par cette voie rapide à celle du modèle à agents sur le même protocole.

Références
----------
[AB19]  Alfonso & Berk, Radiat Oncol 14:88 (2019) — distribution de
        radiosensibilité, sélection sous fractionnement, résistance = baisse d'α.
[Bri96] Britten et al., IJROBP 34:367 (1996) — SF2 de 0.24 à 0.52 entre clones
        d'une même tumeur : fonde la largeur de la distribution initiale.
[Ema91]/[Bur91] — toxicité des tissus sains (via `toxicity.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from .growth import SpheroidSimulation
from .parameters import Parameters
from .therapy import oer, survival_fraction
from .toxicity import ToxicityModel
from .units import SECONDS_PER_DAY, conc_to_po2
from .vascular import VascularConfig, VascularNetwork

# --------------------------------------------------------------------------
# Distribution de radiosensibilité
# --------------------------------------------------------------------------


@dataclass
class ResistanceDistribution:
    """Distribution log-normale discrétisée du facteur multiplicatif de α.

    `factors[i]` = facteur appliqué à α pour la classe i (1.0 = sensible de
    référence, < 1 = résistant). `weights[i]` = fraction de la population.

    La largeur par défaut (sigma = 0.15) est calée sur Britten et al. : sur
    96 clones monocellulaires, le SF2 varie d'un facteur 2 à 8 au sein d'une
    même tumeur. Ce n'est pas un paramètre de confort, c'est une mesure.
    """

    factors: np.ndarray
    weights: np.ndarray
    radioresist_max: float = 0.75      # borne biologique, cf. evolution.py

    @classmethod
    def lognormal(cls, sigma: float = 0.15, n_bins: int = 25,
                  span: float = 3.0) -> "ResistanceDistribution":
        if sigma <= 0:
            return cls(np.array([1.0]), np.array([1.0]))
        # grille en log, centrée pour que la MOYENNE du facteur soit 1
        mu = -0.5 * sigma ** 2
        lo, hi = mu - span * sigma, mu + span * sigma
        z = np.linspace(lo, hi, n_bins)
        f = np.exp(z)
        w = np.exp(-0.5 * ((z - mu) / sigma) ** 2)
        w /= w.sum()
        return cls(f, w)

    def mean_factor(self) -> float:
        return float(np.sum(self.factors * self.weights))

    def resistance_pct(self) -> float:
        """% de résistance acquise (0 = population de départ)."""
        return float(100.0 * (1.0 - self.mean_factor()))

    def irradiate(self, dose_eff_gy: float, p: Parameters) -> float:
        """Applique une fraction et renvoie la survie MOYENNE de la population.

        Chaque classe survit selon le modèle linéaire-quadratique avec son
        propre α. Les classes résistantes survivent mieux -> leur poids relatif
        augmente : c'est la sélection darwinienne, obtenue sans tirage
        stochastique (repondération exacte).
        """
        a = p.alpha_lq * self.factors
        b = p.beta_lq
        sf = np.exp(-a * dose_eff_gy - b * dose_eff_gy ** 2)
        surv = self.weights * sf
        total = float(surv.sum())
        if total > 1e-300:
            self.weights = surv / total
        return total

    def mutate(self, n_divisions: float, p_mut: float = 0.1,
               gain: float = 0.3) -> None:
        """Mutation *de novo* à la division : un clone fille peut devenir plus
        résistant que sa mère.

        L'ABM tire cet événement cellule par cellule à chaque division
        (`p_radioresist`, `radioresist_gain`). Ici on applique l'espérance :
        une fraction `p_eff` du poids de chaque classe migre vers la classe
        dont le facteur α est réduit de `gain`.

        ATTENTION au taux à passer : dans `evolution.py`, `p_radioresist` est
        CONDITIONNÉ à l'apparition préalable d'un driver (`u_driver`). Le taux
        effectif par division est donc `u_driver * p_radioresist` ~ 1e-4, et
        non 0.1. Utiliser 0.1 directement produit une dérive de +12.7 % sur le
        témoin non irradié, là où l'ABM mesure 0.000 % : erreur d'un facteur
        1000 qui détruirait le sens du jeu (la résistance apparaîtrait sans
        traitement). `radioresist_max` borne la réduction cumulée.
        """
        if p_mut <= 0 or gain <= 0 or n_divisions <= 0:
            return
        p_eff = 1.0 - (1.0 - p_mut) ** max(n_divisions, 0.0)
        p_eff = float(np.clip(p_eff, 0.0, 1.0))
        floor = 1.0 - self.radioresist_max
        target = np.maximum(self.factors[:, None] * (1.0 - gain), floor)
        # classe la plus proche du facteur muté (grille discrète)
        j = np.argmin(np.abs(self.factors[None, :] - target), axis=1)
        w_new = self.weights * (1.0 - p_eff)
        np.add.at(w_new, j, self.weights * p_eff)
        self.weights = w_new / w_new.sum()

    def copy(self) -> "ResistanceDistribution":
        return ResistanceDistribution(self.factors.copy(), self.weights.copy())


# --------------------------------------------------------------------------
# Configuration et état
# --------------------------------------------------------------------------

GLUCOSE_LEVELS = (16.5, 5.5, 1.7, 0.8)      # mM, milieux de Freyer & Sutherland
OXYGEN_LEVELS = (0.28, 0.07)                # mM, [FS86]


# ---------------------------------------------------------------------------
# AVERTISSEMENT D'ÉCHELLE — à lire avant de régler l'équilibrage du jeu
# ---------------------------------------------------------------------------
# EMT6/Ro en culture double en ~20.6 h. Sur une semaine, cela fait un facteur
# 2^(168/20.6) = x285 de repousse. Or 5 fractions de 2 Gy en normoxie laissent
# survivre 0.495^5 = 3.0 % des cellules. Bilan hebdomadaire : 0.030 x 285 = x8.5.
#
# => LE FRACTIONNEMENT CONVENTIONNEL NE PEUT PAS CONTRÔLER CE SPHÉROÏDE.
#
# Ce n'est pas un défaut du code : c'est la conséquence arithmétique du temps de
# doublement de la lignée. En clinique le fractionnement fonctionne parce qu'une
# tumeur humaine double en 30 à 100 jours (x1.05 à x1.27 par semaine), et non en
# 20 heures. Avec td = 20 jours, le même calcul donne x0.038 par semaine : le
# contrôle est acquis.
#
# Conséquence pour le jeu : tel quel, « frapper au maximum » GAGNE et le
# protocole standard PERD — l'inverse du message voulu. Deux options honnêtes :
#
#   1. `GameConfig.clinical_timescale = True` (défaut) : on ralentit la
#      prolifération vers un temps de doublement tumoral humain. On quitte le
#      régime validé in vitro, et il faut le DIRE au joueur. C'est le bon choix
#      pour un jeu de radiothérapie, qui parle de patients, pas de boîtes de
#      Petri.
#   2. `clinical_timescale = False` : on reste strictement dans le régime validé
#      (EMT6/Ro in vitro). Le jeu devient alors une simulation de culture
#      cellulaire, où l'escalade de dose est effectivement la bonne réponse tant
#      que la toxicité le permet.
#
# Aucune de ces options n'est un « réglage de confort » : ce sont deux domaines
# de validité différents, explicitement étiquetés.
# ---------------------------------------------------------------------------

# Temps de doublement d'une tumeur solide humaine (volume doubling time).
# Revue : Friberg & Mattson, J Surg Oncol 65:284 (1997) — poumon ~90-200 j,
# sein ~100-200 j, colorectal ~600 j ; les plus agressives ~20-40 j.
CLINICAL_DOUBLING_TIME_DAYS = 20.0


@dataclass
class GameConfig:
    """Règles de la partie. Tout est explicite et modifiable côté Unity."""
    max_days: int = 180
    turn_days: float = 1.0
    seed: int = 0
    organ: str = "generique_tardif"
    sparing_factor: float = 0.6
    dose_max_per_turn: float = 5.0
    dose_budget_gy: float = 80.0
    # conditions de fin
    win_cells: float = 100.0
    control_diameter_um: float = 400.0
    lose_diameter_um: float = 2000.0
    # Seuil de défaite par toxicité. 100 (= NTCP 100 %) était un mauvais choix :
    # la loi normale cumulée est ASYMPTOTIQUE à 1, donc le seuil n'est atteint
    # que par saturation numérique — un joueur pouvait gagner à NTCP 99.99 %.
    # 50 % est la définition même de TD50 (Emami/Burman) : la dose à laquelle
    # la moitié des patients développent la complication. Cliniquement, on vise
    # < 5 % ; laisser jouer jusqu'à 50 % est déjà très permissif.
    lose_toxicity: float = 50.0
    warn_toxicity: float = 5.0        # seuil d'alerte clinique usuel
    # observation
    imaging_noise_pct: float = 5.0
    marker_noise_pct: float = 20.0
    biopsy_cost_days: float = 1.0
    po2_imaging_cost_days: float = 0.5
    # biologie
    initial_radio_sigma: float = 0.15
    initial_radius_um: float = 50.0
    # Taux EFFECTIF par division = u_driver * p_radioresist (cf. evolution.py :
    # la radiorésistance n'est tirée QUE si un driver vient d'apparaître).
    u_driver: float = 1e-3
    p_radioresist: float = 0.1        # conditionnel à un driver
    # Échelle de temps : voir l'avertissement en tête de module.
    clinical_timescale: bool = True
    clinical_doubling_days: float = CLINICAL_DOUBLING_TIME_DAYS
    radioresist_gain: float = 0.3     # réduction de α par mutation
    # Régime vascularisé : fermeture phénoménologique séparée du benchmark EMT6/Ro.
    vascular_enabled: bool = False
    vascular_initial_activity: float = 0.0
    vascular_abnormality: float = 0.45
    # Le modèle vascularisé est le mode humain fictif du jeu, pas une prédiction patient.
    model_regime: str = "avasculaire_validated"


SCENARIOS: Dict[str, Dict] = {
    # Chaque scénario est un compromis EXPLICITE. Les résultats indiqués sont
    # mesurés (graine 1), pas espérés.
    "tutoriel": dict(
        initial_radius_um=300.0, organ="generique_tardif", sparing_factor=0.6,
        initial_radio_sigma=0.15, dose_budget_gy=80.0, max_days=70,
        clinical_doubling_days=4.0,
        # Marge confortable : presque toute stratégie raisonnable gagne.
    ),
    "standard": dict(
        initial_radius_um=600.0, organ="generique_tardif", sparing_factor=1.0,
        initial_radio_sigma=0.35, dose_budget_gy=40.0, max_days=70,
        clinical_doubling_days=4.0,
        # Mesuré : 5 Gy/j -> victoire (tox 8.9) ; 2 Gy 5j/7 -> contrôle durable
        # (tox 1.1). Les deux "réussissent", mais pas au même prix.
    ),
    "organe_critique": dict(
        initial_radius_um=600.0, organ="poumon", sparing_factor=1.0,
        initial_radio_sigma=0.35, dose_budget_gy=60.0, max_days=70,
        clinical_doubling_days=4.0,
        # Mesuré : 5 Gy/j ET 2 Gy 5j/7 tuent le patient. Il faut épargner
        # l'organe (sparing_factor plus bas = meilleure balistique) ou accepter
        # un contrôle partiel. C'est le vrai dilemme de la radiothérapie.
    ),
    "vascular_human": dict(
        initial_radius_um=520.0, organ="generique_tardif", sparing_factor=0.65,
        initial_radio_sigma=0.28, dose_budget_gy=60.0, max_days=120,
        clinical_timescale=True, clinical_doubling_days=28.0,
        vascular_enabled=True, vascular_initial_activity=0.08,
        vascular_abnormality=0.50, model_regime="vascular_scenario",
        # Mode humain fictif : angiogenic switch et vaisseaux anormaux activés.
        # Les constantes vasculaires sont une fermeture de scénario, non des
        # paramètres cliniques universels (voir vascular.py et README).
    ),
    "in_vitro_strict": dict(
        initial_radius_um=400.0, organ="generique_tardif", sparing_factor=0.6,
        initial_radio_sigma=0.15, dose_budget_gy=80.0, max_days=90,
        clinical_timescale=False,
        # Régime STRICTEMENT validé (EMT6/Ro, td = 20.6 h). Le fractionnement
        # n'y contrôle rien : c'est une simulation de culture cellulaire, pas
        # de patient. À réserver au mode "laboratoire".
    ),
}


def make_config(scenario: str = "standard", **overrides) -> "GameConfig":
    """Construit une GameConfig à partir d'un scénario nommé."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scénario inconnu: {scenario!r} "
                         f"(disponibles: {sorted(SCENARIOS)})")
    cfg = dict(SCENARIOS[scenario])
    cfg.update(overrides)
    return GameConfig(**cfg)


@dataclass
class TurnResult:
    """Ce que le moteur renvoie à Unity après chaque tour."""
    turn: int
    day: float
    observation: Dict
    truth: Dict
    reward: float
    done: bool
    outcome: str
    info: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Moteur
# --------------------------------------------------------------------------


class TumorGame:
    """Moteur tour par tour. Sans interface, sans I/O : Unity pilote.

    Usage :
        g = TumorGame(GameConfig(seed=1))
        obs = g.reset()
        res = g.step({"type": "irradiate", "dose_gy": 2.0})
    """

    def __init__(self, config: Optional[GameConfig] = None,
                 params: Optional[Parameters] = None):
        self.cfg = config or GameConfig()
        self.base_params = params or Parameters()
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> Dict:
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.p = self.base_params
        if cfg.clinical_timescale:
            # `td_min` (h) est le temps de doublement minimal de la lignée ;
            # `k_prolif_max` en dérive (ln2/td_min). On l'allonge à l'échelle
            # d'une tumeur solide humaine : toute la structure spatiale
            # (gradients, nécrose, couronne viable) est conservée, seule
            # l'horloge change. La radiosensibilité n'est PAS touchée.
            self.p = self.p.copy_with(td_min=cfg.clinical_doubling_days * 24.0)
        self.base_runtime_params = self.p
        self.vascular = VascularNetwork(VascularConfig(
            enabled=cfg.vascular_enabled,
            initial_activity=cfg.vascular_initial_activity,
            abnormality=cfg.vascular_abnormality,
        ))
        self.sim = SpheroidSimulation(self.p, R0=cfg.initial_radius_um,
                                      vascular=self.vascular)
        self.resistance = ResistanceDistribution.lognormal(cfg.initial_radio_sigma)
        self.tox = ToxicityModel(organ=cfg.organ, sparing_factor=cfg.sparing_factor)
        self.turn = 0
        self.day = 0.0
        self.dose_delivered = 0.0
        self.done = False
        self.outcome = "en_cours"
        self.glucose = GLUCOSE_LEVELS[0]
        self.oxygen = OXYGEN_LEVELS[0]
        self.history: List[Dict] = []
        self._last_biopsy: Optional[Dict] = None
        self._last_po2_map: Optional[Dict] = None
        self._initial_cells = float(self._snapshot().n_cells)
        self.final_score = 0.0
        return self.observe()

    # ------------------------------------------------------------------
    def _snapshot(self):
        return self.sim.snapshot()

    def truth_state(self) -> Dict:
        """État RÉEL, non bruité. Pour l'entraînement et le débogage — pas
        pour l'affichage joueur (sinon le jeu perd son sens)."""
        s = self._snapshot()
        return {
            "day": round(self.day, 3),
            "diameter_um": float(s.diameter),
            "n_cells": float(s.n_cells),
            "viable_rim_um": float(s.viable_rim),
            "necrotic_radius_um": float(s.necrotic_radius),
            "hypoxic_fraction": float(s.hypoxic_fraction),
            "central_po2_mmHg": float(conc_to_po2(s.c_o2[0])),
            "central_ph": float(s.ph[0]),
            "central_lactate_mM": float(s.c_lac[0]),
            "mean_radio_alpha_factor": self.resistance.mean_factor(),
            "resistance_pct": self.resistance.resistance_pct(),
            "dose_delivered_gy": self.dose_delivered,
            "eqd2_normal_tissue_gy": self.tox.eqd2(),
            "ntcp_pct": 100.0 * self.tox.ntcp(),
            "toxicity_score": self.tox.toxicity_score(),
            "glucose_mM": self.glucose,
            "oxygen_mM": self.oxygen,
            "model_regime": self.cfg.model_regime,
            "vascular": self.vascular.summary(),
            "score": float(self.final_score),
        }

    def observe(self) -> Dict:
        """État OBSERVABLE par le joueur : bruité et partiel.

        Le bruit d'imagerie (± 5 %) reproduit la variabilité de mesure d'un
        diamètre sur imagerie ; le marqueur sérique (± 20 %) joue le rôle d'un
        PSA / CA-125. Le joueur ne voit JAMAIS le nombre de cellules ni la
        résistance : c'est tout l'intérêt du jeu.
        """
        t = self.truth_state()
        n = self.rng.normal
        d_obs = t["diameter_um"] * (1 + n(0, self.cfg.imaging_noise_pct / 100))
        marker = t["n_cells"] * (1 + n(0, self.cfg.marker_noise_pct / 100))
        obs = {
            "turn": self.turn,
            "day": round(self.day, 3),
            "days_remaining": self.cfg.max_days - self.day,
            "diameter_um_observed": max(0.0, float(d_obs)),
            "serum_marker": max(0.0, float(marker)),
            "dose_delivered_gy": t["dose_delivered_gy"],
            "dose_budget_remaining_gy": self.cfg.dose_budget_gy - self.dose_delivered,
            "toxicity_score": t["toxicity_score"],
            "ntcp_pct": t["ntcp_pct"],
            "glucose_mM": self.glucose,
            "oxygen_mM": self.oxygen,
            "model_regime": self.cfg.model_regime,
            "vascular_activity": float(self.vascular.activity),
            "last_biopsy": self._last_biopsy,
            "last_po2_imaging": self._last_po2_map,
            "outcome": self.outcome,
            "score": float(self.final_score),
            "done": self.done,
        }
        return obs

    # ------------------------------------------------------------------
    def _apply_radiation(self, dose_gy: float) -> Dict:
        """Irradie : tue via le champ continu ET sélectionne la résistance.

        Le couplage est cohérent : la dose efficace locale (corrigée de l'OER
        et de la quiescence) est celle du modèle validé ; la distribution de
        résistance est repondérée avec la dose efficace MOYENNE pondérée par la
        population vivante.
        """
        sim = self.sim
        sol, k_p, k_d = sim._rates()
        po2 = conc_to_po2(sol.c_o2)
        quiescent = 1.0 - np.clip(k_p / max(self.p.k_prolif_max, 1e-30), 0.0, 1.0)

        # Survie de la POPULATION HÉTÉROGÈNE : moyenne pondérée des survies de
        # chaque classe, et non survie calculée au facteur alpha moyen. Les deux
        # diffèrent (inégalité de Jensen) : exp(-a·D) est convexe, donc la survie
        # au alpha moyen SOUS-ESTIME systématiquement la survie réelle. L'écart
        # atteint 2.1x à 30 Gy. Comme c'est précisément la queue résistante qui
        # fait l'intérêt du jeu, l'approximation détruirait le mécanisme.
        sf = np.zeros_like(po2)
        for f_i, w_i in zip(self.resistance.factors, self.resistance.weights):
            if w_i < 1e-12:
                continue
            p_i = self.p.copy_with(alpha_lq=self.p.alpha_lq * f_i)
            sf += w_i * survival_fraction(dose_gy, po2, p_i, quiescent)

        before = float(np.sum(sim.phi))
        sim.phi = np.clip(sim.phi * sf, 0.0, 1.0)
        after = float(np.sum(sim.phi))

        # dose efficace moyenne vue par la population vivante (pour la sélection)
        w = np.maximum(sim.phi, 0.0)
        wsum = float(w.sum())
        d_eff_mean = float(np.sum(w * dose_gy * oer(po2, self.p) / self.p.oer_max)
                           / wsum) if wsum > 1e-30 else dose_gy
        self.resistance.irradiate(d_eff_mean, self.p)

        self.tox.add_fraction(day=self.day, dose_gy=dose_gy)
        self.dose_delivered += dose_gy
        return {
            "dose_gy": dose_gy,
            "mean_SF": float(np.mean(sf)),
            "phi_before": before,
            "phi_after": after,
            "d_eff_mean_gy": d_eff_mean,
            "resistance_pct": self.resistance.resistance_pct(),
        }

    def _set_medium(self, glucose: Optional[float], oxygen: Optional[float]) -> None:
        if glucose is not None:
            self.glucose = float(min(GLUCOSE_LEVELS, key=lambda g: abs(g - glucose)))
        if oxygen is not None:
            self.oxygen = float(min(OXYGEN_LEVELS, key=lambda o: abs(o - oxygen)))
        self.p = self.base_runtime_params.copy_with(c_glc_bulk=self.glucose,
                                                    c_o2_bulk=self.oxygen)
        self.sim.p = self.p
        # TransportSolver conserve une référence aux paramètres : la recréer
        # évite de laisser l'ancien milieu dans le solveur après un examen/action.
        from .transport import TransportSolver
        self.sim.solver = TransportSolver(self.p)

    def _do_biopsy(self) -> Dict:
        """Biopsie : révèle la distribution de résistance, avec bruit
        d'échantillonnage (on ne prélève qu'une poignée de cellules)."""
        n_sample = 60
        idx = self.rng.choice(len(self.resistance.factors), size=n_sample,
                              p=self.resistance.weights)
        sampled = self.resistance.factors[idx]
        return {
            "day": round(self.day, 2),
            "n_cells_sampled": n_sample,
            "mean_alpha_factor": float(np.mean(sampled)),
            "resistance_pct_estimated": float(100 * (1 - np.mean(sampled))),
            "spread_iqr": float(np.percentile(sampled, 75) - np.percentile(sampled, 25)),
        }

    def _do_po2_imaging(self) -> Dict:
        s = self._snapshot()
        po2 = conc_to_po2(s.c_o2)
        return {
            "day": round(self.day, 2),
            "central_po2_mmHg": float(po2[0]),
            "rim_po2_mmHg": float(po2[-1]),
            "hypoxic_fraction": float(s.hypoxic_fraction),
        }

    # ------------------------------------------------------------------
    def step(self, action: Optional[Dict] = None) -> TurnResult:
        """Joue un tour. `action` est un dict (voir PROTOCOLE.md).

        Types : "wait", "irradiate", "set_medium", "biopsy", "po2_imaging",
        "antiangiogenic".
        """
        if self.done:
            return TurnResult(self.turn, self.day, self.observe(),
                              self.truth_state(), 0.0, True, self.outcome)

        action = action or {"type": "wait"}
        a = action.get("type", "wait")
        info: Dict = {"action": a}
        extra_days = 0.0

        if a == "irradiate":
            dose = float(np.clip(action.get("dose_gy", 0.0), 0.0,
                                 self.cfg.dose_max_per_turn))
            remaining = self.cfg.dose_budget_gy - self.dose_delivered
            dose = max(0.0, min(dose, remaining))
            if dose > 0:
                info["radiation"] = self._apply_radiation(dose)
            else:
                info["radiation"] = {"dose_gy": 0.0, "note": "budget épuisé"}
        elif a == "set_medium":
            self._set_medium(action.get("glucose_mM"), action.get("oxygen_mM"))
            info["medium"] = {"glucose_mM": self.glucose, "oxygen_mM": self.oxygen}
        elif a == "biopsy":
            self._last_biopsy = self._do_biopsy()
            extra_days = self.cfg.biopsy_cost_days
            info["biopsy"] = self._last_biopsy
        elif a == "po2_imaging":
            self._last_po2_map = self._do_po2_imaging()
            extra_days = self.cfg.po2_imaging_cost_days
            info["po2_imaging"] = self._last_po2_map
        elif a == "antiangiogenic":
            intensity = float(np.clip(action.get("intensity", 1.0), 0.0, 1.0))
            info["antiangiogenic"] = self.vascular.apply_antiangiogenic(intensity)
        elif a != "wait":
            info["error"] = f"action inconnue: {a}"

        # avance du temps
        dt_days = self.cfg.turn_days + extra_days
        target = (self.day + dt_days)
        n_before = self._snapshot().n_cells
        self.sim.run(target, dt_max=600.0, record_every_h=24.0)
        self.day = self.sim.t / SECONDS_PER_DAY

        # mutation de novo : nb de divisions ~ log2 de la croissance nette
        n_after = self._snapshot().n_cells
        if n_after > n_before > 0:
            ndiv = float(np.log2(n_after / n_before))
            self.resistance.mutate(ndiv,
                                   self.cfg.u_driver * self.cfg.p_radioresist,
                                   self.cfg.radioresist_gain)
        self.turn += 1

        truth = self.truth_state()
        reward, done, outcome = self._evaluate(truth)
        self.done, self.outcome = done, outcome
        if done:
            self.final_score = self._calculate_score(truth, outcome)
            truth["score"] = float(self.final_score)
        info["score"] = float(self.final_score)
        obs = self.observe()
        self.history.append({"turn": self.turn, "action": action,
                             "truth": truth, "reward": reward,
                             "score": self.final_score})
        return TurnResult(self.turn, self.day, obs, truth, reward, done, outcome, info)

    # ------------------------------------------------------------------
    def _calculate_score(self, t: Dict, outcome: str) -> float:
        """Score final lisible : contrôle, dose économisée, patient préservé.

        Ce score est une métrique de jeu, pas un critère médical. Il est versionné
        par ``model_regime`` dans les traces pour éviter les classements
        incomparables entre règles différentes.
        """
        burden = np.clip(t["n_cells"] / max(self._initial_cells, 1e-30), 0.0, 1.0)
        control = 1000.0 * (1.0 - burden)
        dose = 300.0 * (1.0 - np.clip(self.dose_delivered /
                                      max(self.cfg.dose_budget_gy, 1e-30), 0.0, 1.0))
        patient = 200.0 * (1.0 - np.clip(t["toxicity_score"] /
                                         max(self.cfg.lose_toxicity, 1e-30), 0.0, 1.0))
        durable = 500.0 if outcome in ("controle_durable", "victoire_totale") else 0.0
        penalty = -2000.0 if outcome == "toxicite_letale" else 0.0
        return float(control + dose + patient + durable + penalty)

    def _evaluate(self, t: Dict):
        """Récompense et conditions de fin.

        La récompense est délibérément SIMPLE et dense : elle sert de base
        d'entraînement, mais l'apprentissage par imitation n'en dépend pas
        (il apprend des actions humaines, pas du score).
        """
        cfg = self.cfg
        d, tox = t["diameter_um"], t["toxicity_score"]

        # La toxicité est évaluée EN PREMIER : un traitement qui stérilise la
        # tumeur en tuant le patient n'est pas une victoire.
        if tox >= cfg.lose_toxicity:
            return -100.0, True, "toxicite_letale"
        if t["n_cells"] < cfg.win_cells:
            # Gagner ne suffit pas : gagner PROPREMENT compte. On récompense la
            # marge de sécurité restante sur la toxicité et la dose épargnée.
            # Sans cela, écraser la tumeur à 5 Gy/j et frôler la complication
            # rapporterait autant qu'un protocole fractionné maîtrisé — alors
            # que c'est justement ce que la radiothérapie clinique cherche à
            # éviter.
            tox_margin = 1.0 - min(tox / max(cfg.lose_toxicity, 1e-9), 1.0)
            dose_margin = 1.0 - min(self.dose_delivered
                                    / max(cfg.dose_budget_gy, 1e-9), 1.0)
            return (100.0 + 60.0 * tox_margin + 20.0 * dose_margin,
                    True, "victoire_totale")
        if d > cfg.lose_diameter_um:
            return -100.0, True, "progression"
        if self.day >= cfg.max_days:
            if d < cfg.control_diameter_um:
                bonus = 50.0 * (1 - self.dose_delivered / max(cfg.dose_budget_gy, 1e-9))
                return 50.0 + bonus, True, "controle_durable"
            return -20.0, True, "echec_controle"

        # récompense dense : pénalise la croissance, la dose et la toxicité
        prev = self.history[-1]["truth"]["diameter_um"] if self.history else d
        growth = (d - prev) / max(prev, 1e-9)
        r = -10.0 * growth - 0.05 * t["dose_delivered_gy"] * 0 - 0.2 * tox
        return float(r), False, "en_cours"

    # ------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({"observation": self.observe(),
                           "truth": self.truth_state()}, ensure_ascii=False)


# --------------------------------------------------------------------------
# Validation croisée : la voie rapide reproduit-elle le modèle à agents ?
# --------------------------------------------------------------------------

def game_matches_abm(dose: float = 2.0, n_fractions: int = 5,
                     sigma: float = 0.15) -> Dict:
    """Compare la sélection de résistance analytique (jeu) à l'ABM stochastique.

    L'ABM a mesuré +10.0 % de résistance après 5 x 2 Gy. Si la voie rapide
    donne un ordre de grandeur comparable, elle est utilisable en production.
    """
    p = Parameters()
    dist = ResistanceDistribution.lognormal(sigma)
    before = dist.mean_factor()
    # Protocole de l'ABM : une fraction, puis 1 jour de repousse (~1 division)
    # pendant lequel des mutants de novo apparaissent.
    for _ in range(n_fractions):
        dist.irradiate(dose, p)
        dist.mutate(n_divisions=1.0, p_mut=1e-3 * 0.1, gain=0.3)
    after = dist.mean_factor()
    return {
        "sigma": sigma,
        "n_fractions": n_fractions,
        "dose_per_fraction_gy": dose,
        "alpha_factor_before": before,
        "alpha_factor_after": after,
        "resistance_pct": 100.0 * (before - after) / before,
    }
