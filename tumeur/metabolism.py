"""
Métabolisme énergétique cellulaire (bioénergétique locale).

Le modèle suit la logique de Bertuzzi, Fasano, Gandolfi & Sinisgalli
(J Theor Biol 2010) : la production locale d'ATP est calculée à partir de la
glycolyse et de la phosphorylation oxydative, puis comparée à deux seuils
(maintenance, mort) pour déterminer prolifération, quiescence et nécrose.

Loi de Pirt (maintenance + croissance)
--------------------------------------
Freyer & Sutherland (J Cell Physiol 1985) mesurent in situ une baisse d'un
facteur ~2.5 des taux CELLULAIRES de consommation d'O2 et de glucose entre les
petits spheroides et un diametre de 1300 um, attribuee a l'accumulation de
cellules quiescentes a metabolisme intrinsequement plus lent. La demande suit
donc une loi de Pirt : une part incompressible de maintenance plus une part
associee a la croissance,

    q = q_max * MM(substrats) * [ f_m + (1 - f_m) * g ]

ou g dans [0,1] est l'activite de croissance locale et f_m = 1/2.5 = 0.4 est la
fraction de maintenance MESUREE par [FS85]. Pour garder le systeme de transport
explicite en les concentrations (et donc differentiable analytiquement pour
Newton), g est pris egal au produit des termes de Michaelis-Menten : g vaut 1
la ou les substrats abondent et tend vers 0 en carence.
Sans ce terme, la couronne viable est surestimee et l'O2 central s'annule bien
trop tot au regard des mesures de Mueller-Klieser & Sutherland (Br J Cancer
45:256, 1982 : 66 mmHg au centre a 400 um de diametre, 13 mmHg a 1000 um).

Cinétiques
----------
Consommation d'O2 (Michaelis-Menten, cofacteur substrat, x Pirt) :
    q_O2  = q_O2_max * [O2]/(Km_O2+[O2]) * [Glc]/(Km_Glc+[Glc]) * pirt

Consommation de glucose avec effet Pasteur (amplification glycolytique
quand l'O2 chute ; documenté chez EMT6/Ro par Casciari et al. 1992) :
    q_Glc = q_Glc_max * [Glc]/(Km_Glc+[Glc]) * (1 + (P-1)*Km_P/(Km_P+[O2]))

Bilan énergétique agrégé :
    r_ATP = 2 * q_Glc + (P/O -> 5) * q_O2

Production de lactate : tout glucose non oxydé est fermenté (2 lactate/glucose).
La stœchiométrie de l'oxydation complète est de 6 O2 par glucose :
    q_Lac = max(0, 2 * (q_Glc - q_O2/6))

Toutes les fonctions sont vectorisées (numpy) et renvoient des dérivées
analytiques pour le solveur de Newton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import Parameters

O2_PER_GLUCOSE = 6.0


def _mm(c: np.ndarray, km: float) -> np.ndarray:
    """Terme de Michaelis-Menten, borné à des concentrations positives."""
    c = np.maximum(c, 0.0)
    return c / (km + c)


def _dmm(c: np.ndarray, km: float) -> np.ndarray:
    """Dérivée du terme de Michaelis-Menten."""
    c = np.maximum(c, 0.0)
    return km / (km + c) ** 2


@dataclass
class MetabolicState:
    """Sortie du calcul métabolique local (par cellule vivante)."""
    q_o2: np.ndarray        # mol O2/cellule/s
    q_glc: np.ndarray       # mol glucose/cellule/s
    q_lac: np.ndarray       # mol lactate/cellule/s (production, >= 0)
    r_atp: np.ndarray       # mol ATP/cellule/s
    k_prolif: np.ndarray    # 1/s
    k_death: np.ndarray     # 1/s
    proliferating: np.ndarray   # booléen : au-dessus du seuil de prolifération
    ph: np.ndarray          # pH extracellulaire local


def _pirt(c_o2: np.ndarray, c_glc: np.ndarray, p: Parameters) -> np.ndarray:
    """Facteur de Pirt : maintenance + part liee a la croissance."""
    g = _mm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)
    return p.f_maint_uptake + (1.0 - p.f_maint_uptake) * g


def _dpirt_do2(c_o2: np.ndarray, c_glc: np.ndarray, p: Parameters) -> np.ndarray:
    return (1.0 - p.f_maint_uptake) * _dmm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)


def _dpirt_dglc(c_o2: np.ndarray, c_glc: np.ndarray, p: Parameters) -> np.ndarray:
    return (1.0 - p.f_maint_uptake) * _mm(c_o2, p.km_o2) * _dmm(c_glc, p.km_glc)


def oxygen_uptake(c_o2: np.ndarray, c_glc: np.ndarray, p: Parameters) -> np.ndarray:
    return (p.q_o2_max * _mm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)
            * _pirt(c_o2, c_glc, p))


def d_oxygen_uptake_do2(c_o2: np.ndarray, c_glc: np.ndarray,
                        p: Parameters) -> np.ndarray:
    base = _mm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)
    dbase = _dmm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)
    return p.q_o2_max * (dbase * _pirt(c_o2, c_glc, p)
                         + base * _dpirt_do2(c_o2, c_glc, p))


def d_oxygen_uptake_dglc(c_o2: np.ndarray, c_glc: np.ndarray,
                         p: Parameters) -> np.ndarray:
    base = _mm(c_o2, p.km_o2) * _mm(c_glc, p.km_glc)
    dbase = _mm(c_o2, p.km_o2) * _dmm(c_glc, p.km_glc)
    return p.q_o2_max * (dbase * _pirt(c_o2, c_glc, p)
                         + base * _dpirt_dglc(c_o2, c_glc, p))


def _pasteur(c_o2: np.ndarray, p: Parameters) -> np.ndarray:
    return 1.0 + (p.pasteur_factor - 1.0) * p.km_pasteur / (p.km_pasteur
                                                            + np.maximum(c_o2, 0.0))


def _dpasteur_do2(c_o2: np.ndarray, p: Parameters) -> np.ndarray:
    return -(p.pasteur_factor - 1.0) * p.km_pasteur / (p.km_pasteur
                                                       + np.maximum(c_o2, 0.0)) ** 2


def glucose_uptake(c_o2: np.ndarray, c_glc: np.ndarray, p: Parameters) -> np.ndarray:
    return (p.q_glc_max * _mm(c_glc, p.km_glc) * _pasteur(c_o2, p)
            * _pirt(c_o2, c_glc, p))


def d_glucose_uptake_dglc(c_o2: np.ndarray, c_glc: np.ndarray,
                          p: Parameters) -> np.ndarray:
    pa = _pasteur(c_o2, p)
    return p.q_glc_max * pa * (_dmm(c_glc, p.km_glc) * _pirt(c_o2, c_glc, p)
                               + _mm(c_glc, p.km_glc) * _dpirt_dglc(c_o2, c_glc, p))


def d_glucose_uptake_do2(c_o2: np.ndarray, c_glc: np.ndarray,
                         p: Parameters) -> np.ndarray:
    mg = _mm(c_glc, p.km_glc)
    return p.q_glc_max * mg * (_dpasteur_do2(c_o2, p) * _pirt(c_o2, c_glc, p)
                               + _pasteur(c_o2, p) * _dpirt_do2(c_o2, c_glc, p))


def lactate_production(q_o2: np.ndarray, q_glc: np.ndarray) -> np.ndarray:
    """2 lactate par glucose fermenté ; l'oxydation consomme 6 O2 par glucose."""
    return np.maximum(0.0, 2.0 * (q_glc - q_o2 / O2_PER_GLUCOSE))


def atp_rate(q_o2: np.ndarray, q_glc: np.ndarray, p: Parameters) -> np.ndarray:
    return p.atp_glyc_per_glc * q_glc + p.atp_ox_per_o2 * q_o2


def ph_from_lactate(c_lac: np.ndarray, p: Parameters) -> np.ndarray:
    """pH extracellulaire à partir du lactate accumulé (tampon linéaire)."""
    return p.ph_bulk - np.maximum(c_lac, 0.0) / p.buffer_capacity


def ph_growth_factor(ph: np.ndarray, p: Parameters) -> np.ndarray:
    """Facteur multiplicatif [0,1] de l'acidité sur la prolifération.

    Nul à pH_e = 6.60 (arrêt de croissance mesuré chez EMT6/Ro, Casciari 1992),
    unitaire au pH physiologique.
    """
    span = p.ph_bulk - p.ph_min_growth
    return np.clip((ph - p.ph_min_growth) / span, 0.0, 1.0)


def evaluate(c_o2: np.ndarray, c_glc: np.ndarray, c_lac: np.ndarray,
             p: Parameters) -> MetabolicState:
    """Évalue l'état métabolique complet en chaque point."""
    c_o2 = np.asarray(c_o2, dtype=float)
    c_glc = np.asarray(c_glc, dtype=float)
    c_lac = np.asarray(c_lac, dtype=float)

    q_o2 = oxygen_uptake(c_o2, c_glc, p)
    q_glc = glucose_uptake(c_o2, c_glc, p)
    q_lac = lactate_production(q_o2, q_glc)
    r_atp = atp_rate(q_o2, q_glc, p)

    ph = ph_from_lactate(c_lac, p)
    f_ph = ph_growth_factor(ph, p)

    # Prolifération : réponse SATURANTE à l'ATP disponible au-delà de la
    # maintenance, modulée par le pH.
    #
    # Justification expérimentale de la saturation : Freyer & Sutherland (1986)
    # mesurent des temps de doublement initiaux IDENTIQUES (20-24 h) sur une
    # gamme de 20x en glucose et de 4x en oxygène. Une réponse linéaire au
    # niveau d'ATP prédirait au contraire un ralentissement d'un facteur ~3 en
    # milieu pauvre : seule une réponse saturante (Hill) reproduit l'invariance
    # observée. On prend un exposant de Hill n = 2 (réponse coopérative
    # standard) et une demi-saturation s50 calibrée.
    surplus = np.clip((r_atp - p.r_atp_maint)
                      / max(p.r_atp_max - p.r_atp_maint, 1e-30), 0.0, 1.0)
    x = surplus / max(p.s50_growth, 1e-9)
    hill = x ** 2 / (1.0 + x ** 2)
    k_prolif = p.k_prolif_max * hill * f_ph

    # Nécrose : déficit énergétique sous le seuil de mort.
    deficit = np.clip((p.r_atp_death - r_atp) / max(p.r_atp_death, 1e-30), 0.0, 1.0)
    k_death = p.k_necrosis * deficit

    # Acidose létale : contribution additive au taux de mort.
    lethal = np.clip((p.ph_min_growth - ph) / max(p.ph_min_growth - p.ph_lethal, 1e-30),
                     0.0, 1.0)
    k_death = k_death + p.k_necrosis * lethal

    # Toxicité directe du lactate [Jia05, BF91], indépendante de l'ATP et du pH.
    # Justification : au centre d'un sphéroïde de 674 µm en milieu riche, le
    # modèle sans ce terme donne r_atp = 2.2 x le seuil de mort et pH = 6.60 —
    # donc aucune mort — alors que la nécrose est expérimentalement systématique
    # au-delà de 600 µm [Wal90]. Le mécanisme manquant est l'inhibition de
    # viabilité par le lactate lui-même, documentée sur cette lignée.
    #
    # IMPLÉMENTATION EN CONJONCTION, et non en addition. Jiang et al. précisent
    # explicitement que « cells can survive these nutrient/waste concentrations
    # INDIVIDUALLY » : un lactate > 8 mM ne tue donc pas à lui seul. L'effet
    # correct est un DURCISSEMENT du seuil énergétique — en milieu acide et
    # chargé en lactate, la cellule doit produire plus d'ATP pour survivre
    # (coût du maintien du pH intracellulaire, extrusion des protons).
    lac_tox = np.clip((c_lac - p.lac_death_onset)
                      / max(p.lac_death_lethal - p.lac_death_onset, 1e-30), 0.0, 1.0)
    r_death_eff = p.r_atp_death * (1.0 + p.lac_death_weight * lac_tox)
    deficit2 = np.clip((r_death_eff - r_atp) / np.maximum(r_death_eff, 1e-30), 0.0, 1.0)
    k_death = np.maximum(k_death, p.k_necrosis * deficit2)

    return MetabolicState(
        q_o2=q_o2, q_glc=q_glc, q_lac=q_lac, r_atp=r_atp,
        k_prolif=k_prolif, k_death=k_death,
        proliferating=k_prolif > 0.05 * p.k_prolif_max,
        ph=ph,
    )
