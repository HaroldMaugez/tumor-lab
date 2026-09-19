"""
Modèle à agents "center-based" en 3D (cellules individuelles).

Physique
--------
Dynamique sur-amortie (le nombre de Reynolds cellulaire est ~1e-10 : l'inertie
est négligeable devant la friction) — cf. Drasdo & Höhme, Phys Biol 2 (2005) :

    eta dx_i/dt = Somme_j F_ij

Force de contact entre deux cellules de rayons R_i, R_j séparées de d :
    * répulsion élastique de type Hertz pour le recouvrement,
    * adhésion linéaire jusqu'à une distance de rupture.

    F_rep = k_rep * (s - d)^{3/2} * sqrt(R_eff)      pour d < s = R_i + R_j
    F_adh = -k_adh * (d - s)                          pour s <= d < s + d_max

Cycle cellulaire
----------------
Chaque cellule accumule un "âge de cycle" à la vitesse locale k_prolif(x)
fournie par la bioénergétique (couplée au champ de nutriments radial, cohérent
avec le modèle continu). À maturité, la cellule se divise en deux cellules
filles placées symétriquement selon une direction aléatoire.
Les cellules dont la production d'ATP passe sous le seuil de mort deviennent
nécrotiques (elles cessent de proliférer, perdent leur adhésion et se résorbent).

Voisinage : grille de hachage spatial -> coût O(N) par pas.
Le module sert de validation croisée du modèle continu : les deux approches,
indépendantes numériquement, doivent produire la même cinétique de croissance
et la même stratification (proliférant / quiescent / nécrotique).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import metabolism as met
from .evolution import CloneTree, EvolutionParams, clonal_summary, neutrality_test
from .parameters import Parameters
from . import therapy as th
from .transport import RadialGrid, TransportSolver
from .units import SECONDS_PER_DAY, SECONDS_PER_HOUR, conc_to_po2

ALIVE, QUIESCENT, NECROTIC = 0, 1, 2


@dataclass
class AgentParams:
    """Paramètres mécaniques du modèle à agents.

    Les forces sont exprimées en unités de friction (F/eta a la dimension d'une
    vitesse, µm/s). Plutôt que de fixer arbitrairement eta, on impose
    directement la MOBILITÉ `mobility` = 1/eta, calibrée pour que la relaxation
    mécanique soit rapide devant le cycle cellulaire (séparation d'échelles :
    le tissu relaxe en minutes, il prolifère en heures) tout en restant
    numériquement stable.
    """
    mobility: float = 0.02           # µm/s par unité de force
    k_rep: float = 12.0              # module de répulsion (Hertz)
    k_adh: float = 1.0               # raideur d'adhésion
    d_max: float = 1.5               # µm : distance de rupture de l'adhésion
    relax_substeps: int = 12         # sous-pas de relaxation par pas biologique
    max_disp_frac: float = 0.05      # déplacement max par sous-pas (fraction du rayon)
    cycle_noise: float = 0.15        # dispersion relative de la durée du cycle


class AgentSimulation:
    """Sphéroïde discret : chaque cellule est un agent 3D."""

    def __init__(self, params: Parameters, agent_params: AgentParams | None = None,
                 n_seed: int = 1, seed: int | None = 0,
                 seed_radius: float | None = None,
                 evolution: EvolutionParams | None = None):
        self.p = params
        self.ap = agent_params or AgentParams()
        # seed=None -> entropie systeme : chaque execution est une realisation
        # differente du processus stochastique. seed=int -> reproductible au bit pres.
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        # --- Moteur evolutif (optionnel) ---
        self.evo: EvolutionParams | None = evolution
        self.tree: CloneTree | None = None
        if evolution is not None:
            self.tree = CloneTree(evolution, self.rng, founder_drivers=1)

        r = params.cell_radius
        self.pos = self._seed_positions(n_seed, seed_radius, r)

        n = self.pos.shape[0]
        self.radius = np.full(n, r)
        self.state = np.zeros(n, dtype=int)
        self.cycle = self.rng.random(n)           # progression dans le cycle [0,1]
        self.cycle_len = self._draw_cycle_len(n)
        # Clone porte par chaque cellule (0 = clone fondateur)
        self.clone_id = np.zeros(n, dtype=np.int64)
        # Hétérogénéité préexistante de radiosensibilité [Britten ; Allam].
        # Chaque cellule fondatrice reçoit son propre clone porteur d'un
        # facteur alpha tiré d'une log-normale. Sans cette variabilité initiale,
        # la radiothérapie ne peut RIEN sélectionner sur quelques semaines : les
        # mutations de novo sont trop rares (cf. Alfonso & Berk 2019).
        if self.tree is not None and self.evo.initial_radio_sigma > 0.0:
            sig = float(self.evo.initial_radio_sigma)
            for i in range(n):
                cid = self.tree.spawn_variant(
                    0, radio_alpha_factor=float(
                        self.rng.lognormal(-sig ** 2 / 2, sig)))
                self.clone_id[i] = cid

        self.solver = TransportSolver(params)
        self.history: List[dict] = []

    # ------------------------------------------------------------------
    def _seed_positions(self, n_seed: int, seed_radius: float | None,
                        r: float) -> np.ndarray:
        """Amorçage : cellule unique, amas compact, ou sphère remplie.

        Pour la validation croisée avec le modèle continu, on remplit une
        sphère de rayon `seed_radius` sur un réseau cubique faces centrées
        (empilement compact) légèrement bruité, de façon à reproduire la
        compacité tissulaire imposée par `cell_density`.
        """
        if seed_radius is not None:
            # Pas du réseau fixé par la densité cellulaire MESURÉE :
            #   spacing = (1 / cell_density)^(1/3)
            # ce qui garantit que le sphéroïde initial a exactement la
            # compacité tissulaire du modèle continu (cohérence des deux
            # descriptions, indispensable pour la validation croisée).
            a = float(self.p.cell_density) ** (-1.0 / 3.0)
            m = int(np.ceil(seed_radius / a)) + 1
            g = np.arange(-m, m + 1) * a
            X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            d = np.linalg.norm(pts, axis=1)
            pts = pts[d <= seed_radius]
            return pts + self.rng.normal(scale=0.06 * r, size=pts.shape)
        if n_seed == 1:
            return np.zeros((1, 3))
        u = self.rng.normal(size=(n_seed, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        rad = r * (self.rng.random(n_seed) ** (1 / 3)) * n_seed ** (1 / 3)
        return u * rad[:, None]

    def _draw_cycle_len(self, n: int) -> np.ndarray:
        """Durée nominale du cycle (log-normale, dispersion mesurée ~15 %)."""
        mu = self.p.td_min * SECONDS_PER_HOUR
        s = self.ap.cycle_noise
        return self.rng.lognormal(mean=np.log(mu) - 0.5 * s ** 2, sigma=s, size=n)

    @property
    def n_cells(self) -> int:
        return self.pos.shape[0]

    def spheroid_radius(self) -> float:
        d = np.linalg.norm(self.pos, axis=1)
        return float(np.percentile(d, 99.0) + self.p.cell_radius) if d.size else self.p.cell_radius

    # ------------------------------------------------------------------
    def _neighbor_pairs(self, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
        """Paires en interaction via hachage spatial (cellules de taille cutoff)."""
        pos = self.pos
        if pos.shape[0] < 2:
            return np.empty(0, int), np.empty(0, int)
        keys = np.floor(pos / cutoff).astype(np.int64)
        table: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, k in enumerate(map(tuple, keys)):
            table.setdefault(k, []).append(idx)

        ii, jj = [], []
        offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for k, members in table.items():
            for off in offsets:
                nk = (k[0] + off[0], k[1] + off[1], k[2] + off[2])
                if nk < k or nk not in table:
                    continue
                other = table[nk]
                if nk == k:
                    for a in range(len(members)):
                        for b in range(a + 1, len(members)):
                            ii.append(members[a]); jj.append(members[b])
                else:
                    for a in members:
                        for b in other:
                            ii.append(a); jj.append(b)
        return np.asarray(ii, int), np.asarray(jj, int)

    def _relax(self, dt: float) -> None:
        """Relaxation mécanique sur-amortie, inconditionnellement bornée.

        Intégration explicite d'Euler sur-amortie avec (i) sous-pas et
        (ii) limitation stricte du déplacement par sous-pas. Cette limitation
        garantit la stabilité même en cas de fort recouvrement transitoire
        juste après une division, sans imposer un pas de temps minuscule.
        """
        ap = self.ap
        if self.n_cells < 2:
            return
        cutoff = 2.0 * float(self.radius.max()) + ap.d_max
        ii, jj = self._neighbor_pairs(cutoff)
        if ii.size == 0:
            return

        sub_dt = dt / ap.relax_substeps
        max_disp = ap.max_disp_frac * self.p.cell_radius

        for _ in range(ap.relax_substeps):
            d_vec = self.pos[jj] - self.pos[ii]
            d = np.maximum(np.linalg.norm(d_vec, axis=1), 1e-9)
            u = d_vec / d[:, None]
            s_sum = self.radius[ii] + self.radius[jj]
            R_eff = (self.radius[ii] * self.radius[jj]) / s_sum

            overlap = np.maximum(s_sum - d, 0.0)
            f = -ap.k_rep * np.sqrt(R_eff) * overlap ** 1.5      # répulsion

            gap = d - s_sum
            adh = (gap > 0) & (gap < ap.d_max) & \
                  (self.state[ii] != NECROTIC) & (self.state[jj] != NECROTIC)
            f = f + np.where(adh, ap.k_adh * gap, 0.0)           # adhésion

            force = f[:, None] * u
            net = np.zeros_like(self.pos)
            np.add.at(net, ii, force)
            np.add.at(net, jj, -force)

            disp = ap.mobility * sub_dt * net
            # Limitation stricte du déplacement (stabilité inconditionnelle)
            norm = np.linalg.norm(disp, axis=1)
            too_big = norm > max_disp
            if np.any(too_big):
                disp[too_big] *= (max_disp / norm[too_big])[:, None]
            self.pos += disp

    # ------------------------------------------------------------------
    def _local_rates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Champs de nutriments (résolus en radial) interpolés sur les agents."""
        R = max(self.spheroid_radius(), 2.0 * self.p.cell_radius)
        n_grid = max(40, min(self.p.n_grid, 160))
        grid = RadialGrid(n_grid, R)

        # Fraction vivante radiale estimée par comptage des agents
        d = np.linalg.norm(self.pos, axis=1)
        edges = grid.r_face
        alive = self.state != NECROTIC
        cnt_all, _ = np.histogram(d, bins=edges)
        cnt_alive, _ = np.histogram(d[alive], bins=edges)
        occupancy = np.zeros(n_grid)
        cell_vol = (4.0 / 3.0) * np.pi * self.p.cell_radius ** 3
        with np.errstate(divide="ignore", invalid="ignore"):
            occupancy = np.nan_to_num(cnt_alive * cell_vol / grid.vol)
        phi = np.clip(occupancy, 0.0, 1.0)

        sol = self.solver.solve(grid, phi)
        idx = np.clip(np.searchsorted(grid.r_face, d) - 1, 0, n_grid - 1)
        return sol.state.k_prolif[idx], sol.state.k_death[idx], conc_to_po2(sol.c_o2)[idx]

    def _genotype_modifiers(self) -> tuple[np.ndarray, np.ndarray]:
        """Facteurs (fitness, tolérance) portés par le génotype de chaque cellule.

        - `fit`  : multiplicateur du taux de prolifération, (1+s)^k pour k drivers.
        - `tol`  : multiplicateur du taux de mort, réduit chez les clones ayant
                   acquis le phénotype glycolytique ou l'acido-résistance
                   (Gatenby & Gillies, Nat Rev Cancer 2004).

        La SÉLECTION est donc entièrement médiée par le microenvironnement :
        un clone avantagé ne prolifère plus vite que là où les ressources le
        permettent, et un clone glycolytique n'est favorisé que dans les zones
        hypoxiques. C'est ce couplage qui distingue ce modèle d'un modèle de
        génétique des populations bien mélangé.
        """
        n = self.n_cells
        if self.tree is None:
            return np.ones(n), np.ones(n)
        uniq, inv = np.unique(self.clone_id, return_inverse=True)
        fit_u = np.empty(uniq.size)
        tol_u = np.empty(uniq.size)
        ev = self.evo
        for j, cid in enumerate(uniq):
            cl = self.tree.clones[int(cid)]
            fit_u[j] = cl.fitness
            t = 1.0
            if cl.glycolytic:
                t *= (1.0 - ev.glycolytic_gain)
            if cl.acid_resistant:
                t *= (1.0 - ev.acid_resist_gain)
            tol_u[j] = t
        return fit_u[inv], tol_u[inv]

    def step(self, dt: float) -> None:
        p = self.p
        k_p, k_d, po2 = self._local_rates()

        # --- Sélection : le génotype module prolifération et survie ---
        fit, tol = self._genotype_modifiers()
        k_p = k_p * fit
        k_d = k_d * tol

        # État : nécrose stochastique sous déficit énergétique
        alive = self.state != NECROTIC
        p_die = 1.0 - np.exp(-k_d * dt)
        dying = alive & (self.rng.random(self.n_cells) < p_die)
        self.state[dying] = NECROTIC
        self.radius[dying] = p.cell_radius * 0.9        # léger retrait

        # Quiescence (marqueur, réversible)
        alive = self.state != NECROTIC
        quiescent = alive & (k_p < 0.05 * p.k_prolif_max)
        self.state[alive & ~quiescent] = ALIVE
        self.state[quiescent] = QUIESCENT

        # Progression du cycle proportionnelle au taux local
        adv = np.where(alive, k_p / max(p.k_prolif_max, 1e-30), 0.0) * dt / self.cycle_len
        self.cycle = self.cycle + adv

        # Division
        divide = np.where(alive & (self.cycle >= 1.0))[0]
        if divide.size:
            self._divide(divide)

        # Desquamation de surface : les cellules les plus externes sont perdues
        # dans le milieu. Le nombre attendu par pas de temps se déduit du flux
        # surfacique  v_shed * 4 pi R^2 * densité cellulaire  (même terme que
        # dans le modèle continu, ce qui rend les deux descriptions comparables).
        R_now = self.spheroid_radius()
        n_shed_exp = (p.shedding_velocity * dt * 4.0 * np.pi * R_now ** 2
                      * p.cell_density)
        n_shed = self.rng.poisson(max(n_shed_exp, 0.0))
        if n_shed > 0 and self.n_cells > n_shed:
            d_all = np.linalg.norm(self.pos, axis=1)
            # tirage parmi les cellules de la couche externe
            outer = np.where(d_all > R_now - 2.0 * p.cell_radius)[0]
            if outer.size:
                n_shed = int(min(n_shed, outer.size))
                gone = self.rng.choice(outer, size=n_shed, replace=False)
                keep = np.ones(self.n_cells, bool)
                keep[gone] = False
                self._subset(keep)

        # Résorption du matériel nécrotique
        nec = np.where(self.state == NECROTIC)[0]
        if nec.size:
            gone = nec[self.rng.random(nec.size) < (1.0 - np.exp(-p.k_lysis * dt))]
            if gone.size:
                keep = np.ones(self.n_cells, bool)
                keep[gone] = False
                self._subset(keep)

        self._relax(dt)
        self.t += dt

    def _divide(self, idx: np.ndarray) -> None:
        n_new = idx.size
        # --- Mutation : chaque fille peut acquerir de nouvelles mutations ---
        if self.tree is not None:
            p_any = self.tree.p_any_mutation
            parent_clones = self.clone_id[idx]
            child_clones = parent_clones.copy()
            # fille restant a la position du parent
            mut_a = self.rng.random(n_new) < p_any
            for j in np.where(mut_a)[0]:
                self.clone_id[idx[j]] = self.tree.mutate_conditional(
                    int(parent_clones[j]), self.t)
            # fille nouvellement creee
            mut_b = self.rng.random(n_new) < p_any
            for j in np.where(mut_b)[0]:
                child_clones[j] = self.tree.mutate_conditional(
                    int(parent_clones[j]), self.t)
            self.clone_id = np.concatenate([self.clone_id, child_clones])
        u = self.rng.normal(size=(n_new, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        off = 0.5 * self.p.cell_radius * u

        new_pos = self.pos[idx] + off
        self.pos[idx] = self.pos[idx] - off

        self.pos = np.vstack([self.pos, new_pos])
        self.radius = np.concatenate([self.radius, np.full(n_new, self.p.cell_radius)])
        self.state = np.concatenate([self.state, np.zeros(n_new, int)])
        self.cycle[idx] = 0.0
        self.cycle = np.concatenate([self.cycle, np.zeros(n_new)])
        self.cycle_len = np.concatenate([self.cycle_len, self._draw_cycle_len(n_new)])

    def _subset(self, keep: np.ndarray) -> None:
        self.pos = self.pos[keep]
        self.radius = self.radius[keep]
        self.state = self.state[keep]
        self.cycle = self.cycle[keep]
        self.cycle_len = self.cycle_len[keep]
        if self.tree is not None:
            self.clone_id = self.clone_id[keep]

    # ------------------------------------------------------------------
    def record(self) -> dict:
        d = np.linalg.norm(self.pos, axis=1)
        nec = self.state == NECROTIC
        rec = {
            "t_days": self.t / SECONDS_PER_DAY,
            "n_cells": int(self.n_cells),
            "n_alive": int(np.sum(self.state == ALIVE)),
            "n_quiescent": int(np.sum(self.state == QUIESCENT)),
            "n_necrotic": int(np.sum(nec)),
            "radius_um": self.spheroid_radius(),
            "necrotic_radius_um": float(np.percentile(d[nec], 95.0)) if nec.sum() > 10 else 0.0,
        }
        rec["viable_rim_um"] = rec["radius_um"] - rec["necrotic_radius_um"]
        if self.tree is not None:
            counts: Dict[int, int] = {}
            for cid in self.clone_id.tolist():
                counts[cid] = counts.get(cid, 0) + 1
            rec.update(clonal_summary(self.tree, counts))
            rec["n_cells"] = int(self.n_cells)
        self.history.append(rec)
        return rec

    def irradiate(self, dose_gy: float) -> dict:
        """Applique une fraction de radiothérapie, cellule par cellule.

        Trois modulations, toutes déjà validées séparément :

        1. **Oxygène (OER)** — les cellules hypoxiques du coeur sont jusqu'à
           ~3x plus résistantes [Alper & Howard-Flanders 1956]. Le facteur de
           dose effective est OER(pO2)/OER_max.
        2. **Quiescence** — les cellules non proliférantes sont plus résistantes.
        3. **Radiorésistance HÉRITABLE** — `clone.radio_alpha_factor` réduit
           alpha et se transmet à la descendance. C'est ce qui permet la
           sélection clonale sous fractionnement [Alfonso & Berk 2019].

        La survie est tirée cellule par cellule (Bernoulli) : le traitement est
        donc lui aussi stochastique, et reproductible à graine fixée.
        """
        if self.n_cells == 0 or dose_gy <= 0:
            return {"dose_gy": dose_gy, "n_before": self.n_cells,
                    "n_after": self.n_cells, "mean_sf": 1.0}

        p = self.p
        k_p, _k_d, po2 = self._local_rates()
        quiescent = 1.0 - np.clip(k_p / max(p.k_prolif_max, 1e-30), 0.0, 1.0)

        # dose effective corrigée de l'oxygène
        d_eff = dose_gy * th.oer(po2, p) / p.oer_max
        a = np.full(self.n_cells, p.alpha_lq)
        b = np.full(self.n_cells, p.beta_lq)

        # quiescence
        scale = 1.0 - quiescent * (1.0 - p.quiescent_radioresistance)
        a = a * scale
        b = b * scale

        # radiorésistance héritable portée par le génotype
        if self.tree is not None:
            uniq, inv = np.unique(self.clone_id, return_inverse=True)
            fac_u = np.array([self.tree.clones[int(c)].radio_alpha_factor
                              for c in uniq])
            a = a * fac_u[inv]

        sf = np.exp(-a * d_eff - b * d_eff ** 2)
        survives = self.rng.random(self.n_cells) < sf
        n_before = self.n_cells
        mean_sf = float(np.mean(sf))
        self._subset(survives)
        return {"dose_gy": float(dose_gy), "n_before": int(n_before),
                "n_after": int(self.n_cells), "mean_sf": mean_sf,
                "mean_radio_factor": float(np.mean(
                    [self.tree.clones[int(c)].radio_alpha_factor
                     for c in self.clone_id])) if self.tree is not None
                    and self.n_cells else 1.0}

    def mean_radioresistance(self) -> float:
        """Facteur alpha moyen de la population (1 = sensible, < 1 = résistant)."""
        if self.tree is None or self.n_cells == 0:
            return 1.0
        return float(np.mean([self.tree.clones[int(c)].radio_alpha_factor
                              for c in self.clone_id]))

    def clone_counts(self) -> Dict[int, int]:
        """Effectif de chaque clone présent."""
        counts: Dict[int, int] = {}
        for cid in self.clone_id.tolist():
            counts[cid] = counts.get(cid, 0) + 1
        return counts

    def allele_frequencies(self) -> np.ndarray:
        """Spectre de fréquences alléliques de la tumeur simulée."""
        if self.tree is None:
            return np.zeros(0)
        return self.tree.allele_frequencies(self.clone_counts())

    def advance(self, days: float, dt: float = 900.0,
                record_every_h: float = 24.0, max_cells: int = 200000,
                verbose: bool = False) -> List[dict]:
        """Avance la simulation de `days` jours À PARTIR DE L'INSTANT COURANT.

        `run()` prend un temps de fin ABSOLU : l'appeler deux fois de suite avec
        la même valeur ne fait rien la seconde fois. Pour une boucle de jeu au
        tour par tour, c'est un piège. `advance()` est la méthode à utiliser
        côté moteur : elle est relative et donc composable.
        """
        return self.run(self.t / SECONDS_PER_DAY + days, dt=dt,
                        record_every_h=record_every_h, max_cells=max_cells,
                        verbose=verbose)

    def run(self, t_end_days: float, dt: float = 900.0,
            record_every_h: float = 12.0, max_cells: int = 60000,
            verbose: bool = False) -> List[dict]:
        t_end = t_end_days * SECONDS_PER_DAY
        rec_dt = record_every_h * SECONDS_PER_HOUR
        next_rec = 0.0
        while self.t < t_end and self.n_cells < max_cells:
            self.step(min(dt, t_end - self.t))
            if self.t >= next_rec:
                r = self.record()
                next_rec += rec_dt
                if verbose:
                    print(f"  [ABM] t={r['t_days']:5.2f} j  N={r['n_cells']:6d}  "
                          f"R={r['radius_um']:6.1f} µm  nec={r['n_necrotic']:5d}")
        return self.history
