"""
Transport réaction-diffusion en géométrie sphérique (volumes finis conservatifs).

Hypothèse de quasi-stationnarité
--------------------------------
Le temps caractéristique de diffusion de l'O2 sur un sphéroïde de 500 µm est
    tau_diff = R^2 / (6 D) = 500^2 / (6 * 1.82e4) ~= 2.3 s,
à comparer au temps caractéristique de croissance (~20 h = 7.2e4 s).
Le rapport est de ~3e-5 : les champs de nutriments sont résolus à l'état
stationnaire à chaque pas de temps (approximation de quasi-équilibre standard,
cf. Ward & King 1997, Jiang et al. 2005). Cette hypothèse est vérifiée
numériquement dans `validation.py`.

Discrétisation
--------------
Volumes finis sur [0, R] avec N mailles, nœuds au centre des mailles.
    Face i+1/2 de rayon r_f = (i+1) h, aire A = 4 pi r_f^2
    Volume V_i = (4 pi / 3) (r_{i+1/2}^3 - r_{i-1/2}^3)
Le schéma est exactement conservatif : le flux entrant à la surface égale
l'intégrale volumique de la consommation (test dans `validation.py`).

Conditions aux limites
----------------------
    r = 0 : symétrie, flux nul.
    r = R : Dirichlet C = C_bulk, ou Robin  -D dC/dr = k_m (C - C_bulk)
            si un coefficient de transfert de masse externe est fourni.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import metabolism as met
from .parameters import Parameters


@dataclass
class RadialGrid:
    """Maillage radial en volumes finis."""
    n: int
    R: float

    def __post_init__(self) -> None:
        self.h = self.R / self.n
        self.r = (np.arange(self.n) + 0.5) * self.h        # centres
        self.r_face = np.arange(self.n + 1) * self.h       # faces (0 .. R)
        self.area = 4.0 * np.pi * self.r_face ** 2
        self.vol = (4.0 * np.pi / 3.0) * (self.r_face[1:] ** 3 - self.r_face[:-1] ** 3)

    def integrate(self, f: np.ndarray) -> float:
        """Intégrale volumique de f sur la sphère."""
        return float(np.sum(f * self.vol))


def _diffusion_matrix(grid: RadialGrid, D: float,
                      bc: str = "dirichlet",
                      k_mass: float | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    """Matrice de diffusion L et vecteur second membre b tels que L @ c + b
    approche  int div(D grad c) dV  sur chaque maille.

    Retourne (L, b_unit) où b_unit doit être multiplié par C_bulk.
    """
    n = grid.n
    main = np.zeros(n)
    lower = np.zeros(n - 1)
    upper = np.zeros(n - 1)
    b_unit = np.zeros(n)

    # Faces internes 1..n-1 (entre mailles i-1 et i)
    for i in range(1, n):
        t = D * grid.area[i] / grid.h
        main[i] -= t
        main[i - 1] -= t
        lower[i - 1] = t     # coefficient de c_{i-1} dans l'équation i
        upper[i - 1] = t     # coefficient de c_i dans l'équation i-1

    # Face externe r = R
    A_R = grid.area[n]
    if bc == "dirichlet":
        # Gradient évalué entre le dernier centre (R - h/2) et la surface.
        t = D * A_R / (0.5 * grid.h)
        main[n - 1] -= t
        b_unit[n - 1] += t
    elif bc == "robin":
        if k_mass is None:
            raise ValueError("k_mass requis pour la condition de Robin")
        # Résistances en série : film externe (1/k_m) + demi-maille (h/2D)
        t = A_R / (1.0 / k_mass + 0.5 * grid.h / D)
        main[n - 1] -= t
        b_unit[n - 1] += t
    else:
        raise ValueError(f"condition limite inconnue : {bc}")

    L = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")
    return L, b_unit


@dataclass
class TransportSolution:
    """Champs convergés."""
    c_o2: np.ndarray        # mM
    c_glc: np.ndarray       # mM
    c_lac: np.ndarray       # mM
    ph: np.ndarray
    state: met.MetabolicState
    iterations: int
    residual: float

    @property
    def po2(self) -> np.ndarray:
        from .units import conc_to_po2
        return conc_to_po2(self.c_o2)


class TransportSolver:
    """Solveur couplé O2 / glucose (Newton amorti) + lactate (linéaire)."""

    def __init__(self, params: Parameters):
        self.p = params

    # ------------------------------------------------------------------
    def solve(self, grid: RadialGrid, phi_live: np.ndarray,
              guess: TransportSolution | None = None,
              vascular: dict | None = None) -> TransportSolution:
        """Résout le système stationnaire couplé.

        Parameters
        ----------
        grid : maillage radial
        phi_live : fraction volumique de cellules vivantes par maille (0..1)
        guess : solution précédente pour initialiser Newton
        vascular : profils d'échange renvoyés par ``VascularNetwork.profiles``.
            None conserve exactement le solveur avasculaire historique.
        """
        p = self.p
        n = grid.n
        bc = "robin" if p.mass_transfer_coeff else "dirichlet"

        L_o2, b_o2 = _diffusion_matrix(grid, p.D_o2, bc, p.mass_transfer_coeff)
        L_glc, b_glc = _diffusion_matrix(grid, p.D_glc, bc, p.mass_transfer_coeff)
        L_lac, b_lac = _diffusion_matrix(grid, p.D_lac, bc, p.mass_transfer_coeff)

        rhs_o2 = b_o2 * p.c_o2_bulk
        rhs_glc = b_glc * p.c_glc_bulk
        rhs_lac = b_lac * p.c_lac_bulk

        # Réseau vasculaire : terme d'échange local k_v(C_v - C). Les valeurs
        # de k_v sont des paramètres de scénario, jamais des mesures patient.
        # On l'écrit dans la matrice pour conserver positivité et conservation.
        if vascular is not None:
            def _exchange(key: str) -> np.ndarray:
                return np.asarray(vascular.get(key, np.zeros(n)), dtype=float)

            ev_o2 = _exchange("exchange_o2_per_s") * grid.vol
            ev_glc = _exchange("exchange_glucose_per_s") * grid.vol
            ev_lac = _exchange("exchange_lactate_per_s") * grid.vol
            L_o2 = L_o2 - sp.diags(ev_o2)
            L_glc = L_glc - sp.diags(ev_glc)
            L_lac = L_lac - sp.diags(ev_lac)
            rhs_o2 = rhs_o2 + ev_o2 * float(vascular.get("o2_vessel_mM", p.c_o2_bulk))
            rhs_glc = rhs_glc + ev_glc * float(vascular.get("glucose_vessel_mM", p.c_glc_bulk))
            rhs_lac = rhs_lac + ev_lac * float(vascular.get("lactate_vessel_mM", p.c_lac_bulk))

        # Densité cellulaire vivante volumique [cellules/µm^3]
        rho = p.cell_density * np.clip(phi_live, 0.0, 1.0)
        # Conversion mol/cellule/s -> mM/s : * rho * 1e18
        conv = rho * 1.0e18
        w = conv * grid.vol      # facteur maille par maille (mM/s -> quantité)

        if guess is not None and len(guess.c_o2) == n:
            c_o2 = guess.c_o2.copy()
            c_glc = guess.c_glc.copy()
            c_lac = guess.c_lac.copy()
        else:
            c_o2 = np.full(n, p.c_o2_bulk)
            c_glc = np.full(n, p.c_glc_bulk)
            c_lac = np.full(n, p.c_lac_bulk)

        residual = np.inf
        it = 0
        for it in range(1, p.newton_maxiter + 1):
            q_o2 = met.oxygen_uptake(c_o2, c_glc, p)
            q_glc = met.glucose_uptake(c_o2, c_glc, p)

            F_o2 = L_o2 @ c_o2 + rhs_o2 - w * q_o2
            F_glc = L_glc @ c_glc + rhs_glc - w * q_glc
            F = np.concatenate([F_o2, F_glc])

            scale = max(abs(rhs_o2).max(), abs(rhs_glc).max(), 1e-30)
            residual = float(np.max(np.abs(F)) / scale)
            if residual < p.newton_tol:
                break

            # Jacobien : blocs diagonaux (dérivées analytiques du métabolisme)
            J11 = L_o2 - sp.diags(w * met.d_oxygen_uptake_do2(c_o2, c_glc, p))
            J12 = -sp.diags(w * met.d_oxygen_uptake_dglc(c_o2, c_glc, p))
            J21 = -sp.diags(w * met.d_glucose_uptake_do2(c_o2, c_glc, p))
            J22 = L_glc - sp.diags(w * met.d_glucose_uptake_dglc(c_o2, c_glc, p))
            J = sp.bmat([[J11, J12], [J21, J22]], format="csc")

            try:
                delta = spla.spsolve(J, -F)
            except Exception:                                  # pragma: no cover
                delta = spla.lsqr(J, -F)[0]
            if not np.all(np.isfinite(delta)):                 # pragma: no cover
                break

            # Recherche linéaire amortie + projection sur les concentrations >= 0
            lam = 1.0
            base = float(np.linalg.norm(F))
            for _ in range(30):
                trial_o2 = np.maximum(c_o2 + lam * delta[:n], 0.0)
                trial_glc = np.maximum(c_glc + lam * delta[n:], 0.0)
                tq_o2 = met.oxygen_uptake(trial_o2, trial_glc, p)
                tq_glc = met.glucose_uptake(trial_o2, trial_glc, p)
                nF = np.concatenate([
                    L_o2 @ trial_o2 + rhs_o2 - w * tq_o2,
                    L_glc @ trial_glc + rhs_glc - w * tq_glc,
                ])
                if np.linalg.norm(nF) < (1.0 - 1e-4 * lam) * base:
                    break
                lam *= 0.5
            c_o2, c_glc = trial_o2, trial_glc

            # Lactate : linéaire à q_glc, q_o2 figés (couplage par le pH via
            # une itération de point fixe englobante ci-dessous)
            q_o2 = met.oxygen_uptake(c_o2, c_glc, p)
            q_glc = met.glucose_uptake(c_o2, c_glc, p)
            src_lac = w * met.lactate_production(q_o2, q_glc)
            c_lac = np.maximum(spla.spsolve(L_lac.tocsc(), -(rhs_lac + src_lac)), 0.0)

        state = met.evaluate(c_o2, c_glc, c_lac, p)
        return TransportSolution(c_o2=c_o2, c_glc=c_glc, c_lac=c_lac,
                                 ph=state.ph, state=state,
                                 iterations=it, residual=residual)

    # ------------------------------------------------------------------
    def oxygen_flux_balance(self, grid: RadialGrid, sol: TransportSolution,
                            phi_live: np.ndarray) -> tuple[float, float]:
        """Contrôle de conservation : (flux entrant en R, consommation totale)."""
        p = self.p
        D, h = p.D_o2, grid.h
        A_R = grid.area[grid.n]
        influx = D * A_R * (p.c_o2_bulk - sol.c_o2[-1]) / (0.5 * h)
        rho = p.cell_density * np.clip(phi_live, 0.0, 1.0) * 1.0e18
        consumed = float(np.sum(rho * sol.state.q_o2 * grid.vol))
        return influx, consumed
