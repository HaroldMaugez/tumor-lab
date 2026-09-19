"""
Système d'unités et constantes physico-chimiques.

Convention interne (cohérente dans TOUT le code) :
    longueur      : micromètre            [µm]
    temps         : seconde               [s]
    quantité      : mole                  [mol]
    concentration : mol/m^3 == millimolaire [mM]
    pression      : millimètre de mercure  [mmHg]

Attention aux facteurs de conversion volume :
    1 m^3 = 1e18 µm^3   =>   (mol/cell/s) * (cells/µm^3) * 1e18 = mol/(m^3.s) = mM/s
"""

from __future__ import annotations

# ----------------------------------------------------------------------------
# Conversions géométriques
# ----------------------------------------------------------------------------
UM3_PER_M3 = 1.0e18          # µm^3 dans 1 m^3
CM2_PER_S_TO_UM2_PER_S = 1.0e8   # 1 cm^2/s = 1e8 µm^2/s
M2_PER_S_TO_UM2_PER_S = 1.0e12   # 1 m^2/s  = 1e12 µm^2/s

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0

# ----------------------------------------------------------------------------
# Solubilité de l'oxygène (loi de Henry) à 37 °C
# ----------------------------------------------------------------------------
# Solubilité de O2 dans le plasma/milieu aqueux à 37 °C :
#   0.0031 mL O2 / (dL . mmHg)  ->  1.38e-3 mM/mmHg
# (Valeur classique de physiologie respiratoire ; cf. West, Respiratory Physiology.)
ALPHA_O2_MM_PER_MMHG = 1.38e-3   # [mM/mmHg]


def po2_to_conc(po2_mmHg: float) -> float:
    """Pression partielle d'O2 [mmHg] -> concentration dissoute [mM]."""
    return ALPHA_O2_MM_PER_MMHG * po2_mmHg


def conc_to_po2(c_mM: float) -> float:
    """Concentration d'O2 dissous [mM] -> pression partielle [mmHg]."""
    return c_mM / ALPHA_O2_MM_PER_MMHG


def per_cell_to_volumetric(q_mol_cell_s: float, cell_density_per_um3: float) -> float:
    """Taux par cellule [mol/cellule/s] -> taux volumique [mM/s] (= mol/m^3/s)."""
    return q_mol_cell_s * cell_density_per_um3 * UM3_PER_M3


def doubling_time_to_rate(td_hours: float) -> float:
    """Temps de doublement [h] -> taux de croissance exponentielle [1/s]."""
    from math import log
    return log(2.0) / (td_hours * SECONDS_PER_HOUR)
