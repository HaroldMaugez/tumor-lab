"""
Modèle continu de croissance avasculaire à frontière libre.

Formulation (Greenspan 1972 ; Ward & King 1997 ; Byrne & Chaplain 1996)
-----------------------------------------------------------------------
Le tissu est un milieu saturé, incompressible, composé d'une fraction
volumique de cellules vivantes phi et de matériel nécrotique (1 - phi).
La prolifération crée du volume, la lyse du matériel nécrotique en détruit :
la vitesse d'advection v du tissu découle de l'incompressibilité

    div(v) = k_p(x) phi - k_l (1 - phi)                                    (1)

soit, en symétrie sphérique,

    v(r) = (1/r^2) * Int_0^r [k_p phi - k_l (1-phi)] s^2 ds .              (2)

La fraction vivante est advectée et échange avec le compartiment nécrotique :

    d(phi)/dt + (1/r^2) d(r^2 v phi)/dr = k_p phi - k_d phi                (3)

Le rayon extérieur suit la vitesse du tissu, moins la perte par desquamation
(cell shedding, documentée chez EMT6/Ro) :

    dR/dt = v(R) - v_shed .                                                (4)

Les taux k_p (prolifération) et k_d (mort) sont fournis par la bioénergétique
locale (module `metabolism`) évaluée sur les champs de nutriments résolus à
l'état quasi stationnaire (module `transport`).

Implémentation
--------------
Coordonnée fixe xi = r/R(t) (méthode des grilles mobiles). Dans ce repère,
l'équation (3) devient, sous forme conservative,

    d(phi)/dt|_xi = -(1/(R xi^2)) d(xi^2 (v - xi Rdot) phi)/dxi
                    - 3 (Rdot/R) phi + (k_p - k_d) phi ,

le terme -3(Rdot/R) phi traduisant la dilatation du domaine. L'advection est
discrétisée en volumes finis avec limiteur de flux van Leer (TVD, d'ordre 2
dans les zones régulières, sans oscillation aux fronts). L'intégration en temps
est un Runge-Kutta d'ordre 2 SSP (Heun) avec pas adaptatif sous contrainte CFL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from . import metabolism as met
from .parameters import Parameters
from .transport import RadialGrid, TransportSolver, TransportSolution
from .units import SECONDS_PER_DAY, SECONDS_PER_HOUR, conc_to_po2


# ---------------------------------------------------------------------------
# Limiteur de flux
# ---------------------------------------------------------------------------
def _van_leer(r: np.ndarray) -> np.ndarray:
    """Limiteur van Leer, symétrique et TVD."""
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (r + np.abs(r)) / (1.0 + np.abs(r))
    return np.nan_to_num(phi, nan=0.0, posinf=2.0, neginf=0.0)


def _advect_flux(q: np.ndarray, w_face: np.ndarray) -> np.ndarray:
    """Flux upwind limité aux faces internes (q défini aux centres).

    w_face : vitesse aux faces internes (taille n-1).
    Retourne le flux advecté q* aux faces internes.
    """
    n = q.size
    qL = q[:-1]
    qR = q[1:]

    # Pentes amont pour la reconstruction
    dq = np.zeros(n + 1)
    dq[1:n] = q[1:] - q[:-1]
    dq[0] = 0.0
    dq[n] = 0.0

    # Ratio de pentes pour chaque face interne i+1/2 (i = 0..n-2)
    eps = 1e-30
    denom = dq[1:n] + eps                       # saut à travers la face
    r_plus = dq[0:n - 1] / denom                # amont gauche
    r_minus = dq[2:n + 1] / denom               # amont droite

    q_left = qL + 0.5 * _van_leer(r_plus) * denom
    q_right = qR - 0.5 * _van_leer(r_minus) * denom

    return np.where(w_face >= 0.0, q_left, q_right)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """État du sphéroïde à un instant donné."""
    t: float                       # s
    radius: float                  # µm
    xi: np.ndarray                 # coordonnée normalisée
    phi_live: np.ndarray
    c_o2: np.ndarray               # mM
    c_glc: np.ndarray              # mM
    c_lac: np.ndarray              # mM
    ph: np.ndarray
    k_prolif: np.ndarray           # 1/s
    k_death: np.ndarray            # 1/s
    velocity: np.ndarray           # µm/s
    n_cells: float
    n_prolif: float
    n_quiescent: float
    volume: float                  # µm^3
    necrotic_radius: float         # µm
    viable_rim: float              # µm
    hypoxic_fraction: float        # fraction de cellules vivantes < 10 mmHg

    @property
    def t_days(self) -> float:
        return self.t / SECONDS_PER_DAY

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def po2(self) -> np.ndarray:
        return conc_to_po2(self.c_o2)

    @property
    def r(self) -> np.ndarray:
        return self.xi * self.radius


# ---------------------------------------------------------------------------
# Simulateur
# ---------------------------------------------------------------------------
class SpheroidSimulation:
    """Croissance d'un sphéroïde tumoral avasculaire, symétrie sphérique."""

    def __init__(self, params: Parameters, R0: float = 50.0,
                 n_grid: Optional[int] = None, vascular=None):
        self.p = params
        self.vascular = vascular
        self.n = int(n_grid or params.n_grid)
        self.R = float(R0)
        self.t = 0.0
        self.phi = np.ones(self.n)             # tissu initialement 100 % vivant
        self.solver = TransportSolver(params)
        self._last: TransportSolution | None = None

        # Grille normalisée (volumes finis en xi)
        self.dxi = 1.0 / self.n
        self.xi = (np.arange(self.n) + 0.5) * self.dxi
        self.xi_face = np.arange(self.n + 1) * self.dxi
        self.dvol_unit = (4.0 * np.pi / 3.0) * (self.xi_face[1:] ** 3
                                                - self.xi_face[:-1] ** 3)

        self.history: List[Snapshot] = []
        # Évènement thérapeutique éventuel : callable(sim) -> None
        self.therapy_hook: Optional[Callable[["SpheroidSimulation"], None]] = None

    # ------------------------------------------------------------------
    @property
    def grid(self) -> RadialGrid:
        return RadialGrid(self.n, self.R)

    def _rates(self) -> tuple[TransportSolution, np.ndarray, np.ndarray]:
        """Résout le transport et renvoie (solution, k_prolif, k_death)."""
        grid = self.grid
        vascular_profile = (self.vascular.profiles(self.xi)
                            if self.vascular is not None and self.vascular.enabled
                            else None)
        sol = self.solver.solve(grid, self.phi, guess=self._last,
                                vascular=vascular_profile)
        self._last = sol
        return sol, sol.state.k_prolif, sol.state.k_death

    def _velocity(self, k_p: np.ndarray, k_d: np.ndarray) -> tuple[np.ndarray, float]:
        """Vitesse d'advection aux faces + Rdot, par intégration de (2)."""
        p = self.p
        source = k_p * self.phi - p.k_lysis * (1.0 - self.phi)
        # Intégrale cumulée Int_0^r S s^2 ds en volumes finis
        r = self.xi * self.R
        r_face = self.xi_face * self.R
        dvol = (r_face[1:] ** 3 - r_face[:-1] ** 3) / 3.0     # = Int s^2 ds
        cum = np.concatenate([[0.0], np.cumsum(source * dvol)])
        with np.errstate(divide="ignore", invalid="ignore"):
            v_face = np.where(r_face > 0, cum / np.maximum(r_face ** 2, 1e-30), 0.0)
        v_face[0] = 0.0
        Rdot = v_face[-1] - p.shedding_velocity
        return v_face, float(Rdot)

    def _rhs(self, phi: np.ndarray, k_p: np.ndarray, k_d: np.ndarray,
             v_face: np.ndarray, Rdot: float) -> np.ndarray:
        """Membre de droite de l'équation d'advection-réaction pour phi."""
        R = self.R
        # Vitesse dans le repère mobile
        w_face = v_face - self.xi_face * Rdot
        w_int = w_face[1:-1]
        q_face = _advect_flux(phi, w_int)

        flux = np.zeros(self.n + 1)
        flux[1:-1] = self.xi_face[1:-1] ** 2 * w_int * q_face
        # Bords : xi=0 -> aire nulle ; xi=1 -> w = v(R) - Rdot = v_shed (sortant)
        flux[0] = 0.0
        flux[-1] = 1.0 ** 2 * (v_face[-1] - Rdot) * phi[-1]

        div = (flux[1:] - flux[:-1]) / self.dxi
        adv = -div / (R * np.maximum(self.xi ** 2, 1e-30))

        dilat = -3.0 * (Rdot / R) * phi
        react = (k_p - k_d) * phi
        return adv + dilat + react

    # ------------------------------------------------------------------
    def step(self, dt: float) -> float:
        """Un pas de temps Heun (SSP-RK2) avec contrainte CFL ; renvoie dt utilisé."""
        p = self.p
        sol, k_p, k_d = self._rates()
        v_face, Rdot = self._velocity(k_p, k_d)

        # Contrainte CFL sur l'advection dans le repère mobile
        w = np.abs(v_face - self.xi_face * Rdot).max()
        dt_cfl = 0.4 * self.R * self.dxi / max(w, 1e-30)
        dt_reac = 0.2 / max(float(np.max(k_p + k_d)), 1e-30)
        dt_rad = 0.02 * self.R / max(abs(Rdot), 1e-30)
        dt = float(min(dt, dt_cfl, dt_reac, dt_rad))

        phi0, R0, t0 = self.phi.copy(), self.R, self.t

        # Étape prédictive
        k1 = self._rhs(phi0, k_p, k_d, v_face, Rdot)
        self.phi = np.clip(phi0 + dt * k1, 0.0, 1.0)
        self.R = max(R0 + dt * Rdot, 2.0 * p.cell_radius)
        self.t = t0 + dt

        # Étape correctrice
        sol2, k_p2, k_d2 = self._rates()
        v2, Rdot2 = self._velocity(k_p2, k_d2)
        k2 = self._rhs(self.phi, k_p2, k_d2, v2, Rdot2)

        self.phi = np.clip(0.5 * (phi0 + self.phi + dt * k2), 0.0, 1.0)
        self.R = max(0.5 * (R0 + self.R + dt * Rdot2), 2.0 * p.cell_radius)
        self.t = t0 + dt
        return dt

    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        p = self.p
        sol, k_p, k_d = self._rates()
        v_face, Rdot = self._velocity(k_p, k_d)
        vol_cells = self.dvol_unit * self.R ** 3

        n_cells = float(np.sum(p.cell_density * self.phi * vol_cells))
        prolif_mask = sol.state.proliferating & (self.phi > 0.05)
        n_prolif = float(np.sum(p.cell_density * self.phi * vol_cells * prolif_mask))
        n_quiescent = n_cells - n_prolif

        # Rayon nécrotique : dernier point (en partant du centre) où phi < 0.5
        nec_idx = np.where(self.phi < 0.5)[0]
        r_nec = float(self.xi[nec_idx[-1]] * self.R) if nec_idx.size else 0.0
        if nec_idx.size and nec_idx[0] != 0:
            r_nec = 0.0 if self.phi[0] >= 0.5 else r_nec

        po2 = conc_to_po2(sol.c_o2)
        live_w = p.cell_density * self.phi * vol_cells
        hyp = float(np.sum(live_w * (po2 < 10.0)) / max(n_cells, 1e-30))

        return Snapshot(
            t=self.t, radius=self.R, xi=self.xi.copy(), phi_live=self.phi.copy(),
            c_o2=sol.c_o2.copy(), c_glc=sol.c_glc.copy(), c_lac=sol.c_lac.copy(),
            ph=sol.ph.copy(), k_prolif=k_p.copy(), k_death=k_d.copy(),
            velocity=v_face.copy(), n_cells=n_cells, n_prolif=n_prolif,
            n_quiescent=n_quiescent,
            volume=float(4.0 / 3.0 * np.pi * self.R ** 3),
            necrotic_radius=r_nec, viable_rim=self.R - r_nec,
            hypoxic_fraction=hyp,
        )

    def run(self, t_end_days: float, dt_max: float = 600.0,
            record_every_h: float = 6.0, verbose: bool = False) -> List[Snapshot]:
        """Intègre jusqu'à t_end_days, en enregistrant régulièrement."""
        t_end = t_end_days * SECONDS_PER_DAY
        rec_dt = record_every_h * SECONDS_PER_HOUR
        next_rec = 0.0
        if not self.history:
            self.history.append(self.snapshot())
            next_rec = rec_dt

        guard = 0
        vascular_elapsed = 0.0
        while self.t < t_end and guard < 2_000_000:
            guard += 1
            dt_used = self.step(min(dt_max, t_end - self.t))
            if self.vascular is not None and self.vascular.enabled:
                # Le réseau est une variable lente : le mettre à jour à chaque
                # sous-pas de l'EDP coûterait des résolutions inutiles et ne
                # serait pas biologiquement défendable. Un quart de jour garde
                # le couplage stable tout en laissant le solveur rapide.
                vascular_elapsed += dt_used / SECONDS_PER_DAY
                if vascular_elapsed >= 0.25 or self.t >= t_end:
                    snap_for_vascular = self.snapshot()
                    self.vascular.step(
                        vascular_elapsed,
                        snap_for_vascular.hypoxic_fraction,
                        float(np.mean(snap_for_vascular.po2)),
                    )
                    vascular_elapsed = 0.0
            if self.therapy_hook is not None:
                self.therapy_hook(self)
            if self.t >= next_rec:
                snap = self.snapshot()
                self.history.append(snap)
                next_rec += rec_dt
                if verbose:
                    print(f"  t={snap.t_days:6.2f} j  d={snap.diameter:7.1f} µm  "
                          f"N={snap.n_cells:.3e}  rim={snap.viable_rim:6.1f} µm  "
                          f"pO2(0)={conc_to_po2(snap.c_o2[0]):6.2f} mmHg")
        return self.history

    # ------------------------------------------------------------------
    def as_arrays(self) -> dict:
        """Séries temporelles agrégées (pour tracé/export)."""
        h = self.history
        return {
            "t_days": np.array([s.t_days for s in h]),
            "diameter_um": np.array([s.diameter for s in h]),
            "radius_um": np.array([s.radius for s in h]),
            "volume_um3": np.array([s.volume for s in h]),
            "n_cells": np.array([s.n_cells for s in h]),
            "n_prolif": np.array([s.n_prolif for s in h]),
            "n_quiescent": np.array([s.n_quiescent for s in h]),
            "necrotic_radius_um": np.array([s.necrotic_radius for s in h]),
            "viable_rim_um": np.array([s.viable_rim for s in h]),
            "hypoxic_fraction": np.array([s.hypoxic_fraction for s in h]),
            "po2_center_mmHg": np.array([conc_to_po2(s.c_o2[0]) for s in h]),
            "glc_center_mM": np.array([s.c_glc[0] for s in h]),
            "ph_center": np.array([s.ph[0] for s in h]),
        }
