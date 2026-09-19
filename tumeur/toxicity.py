"""Toxicité cumulée des tissus sains — modèle sourcé.

Ce module est la brique « patient » qui manquait au simulateur. Sans elle, une
IA entraînée sur ce modèle découvrirait qu'irradier à dose illimitée est optimal,
puisque rien ne représenterait le coût pour l'organisme.

RIEN N'EST INVENTÉ ICI. Le modèle est le standard de la radio-oncologie :

1. **Dose biologiquement efficace (BED)** — modèle linéaire-quadratique.
       BED = D · (1 + d/(α/β))
   avec **α/β = 3 Gy pour les tissus sains à réponse tardive** (contre 10 Gy pour
   les tumeurs). C'est cette différence qui fonde tout le fractionnement en
   radiothérapie : fractionner épargne le tissu sain plus que la tumeur.
   [Fowler 1989 ; Joiner & van der Kogel, *Basic Clinical Radiobiology*]

2. **EQD2** — dose équivalente en fractions de 2 Gy, unité dans laquelle sont
   exprimées toutes les contraintes cliniques :
       EQD2 = BED / (1 + 2/(α/β))

3. **NTCP de Lyman-Kutcher-Burman** — probabilité de complication du tissu sain :
       NTCP = Φ[(EQD2 − TD50) / (m · TD50)]
   Φ = fonction de répartition normale centrée réduite.
   [Lyman 1985 ; Kutcher & Burman 1989 ; Burman et al., IJROBP 21:123 (1991),
    ajustés sur les données de tolérance d'Emami et al., IJROBP 21:109 (1991)]

4. **Réparation sublétale entre fractions** — la réparation des lésions
   sublétales suit une décroissance exponentielle de demi-vie **T½ ≈ 1.5 h**
   [Bentzen ; Fowler]. Deux fractions trop rapprochées ne sont pas pleinement
   réparées : le modèle applique une pénalité de réparation incomplète.

5. **Repopulation** — les tissus à réponse tardive se repeuplent peu :
   **k ≈ 0-0.1/jour** (contre 0.2-0.3/jour pour les tissus à réponse aiguë).
   [AAPM, cours de radiobiologie]

Les valeurs de TD50 et m proviennent d'Emami/Burman pour plusieurs organes ; le
défaut utilisé dans le jeu correspond à un tissu à réponse tardive générique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, sqrt
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paramètres de tolérance publiés (Emami 1991 / Burman 1991, et QUANTEC 2010)
# organe -> (TD50 en Gy pour irradiation uniforme de l'organe entier,
#            m = pente, n = effet de volume, référence)
# ---------------------------------------------------------------------------
LKB_PARAMETERS: Dict[str, Tuple[float, float, float, str]] = {
    "moelle_epiniere": (66.5, 0.175, 0.05,
                        "Emami/Burman 1991 ; myélite. Effet de volume faible "
                        "(organe série) ; TD5/5 = 50 Gy"),
    "poumon": (24.5, 0.18, 0.87,
               "Emami/Burman 1991 ; pneumopathie radique. TD5/5 = 17.5 Gy. "
               "Effet de volume fort (organe parallèle)"),
    "rectum": (80.0, 0.15, 0.12,
               "Emami/Burman 1991 ; proctite sévère/sténose/nécrose. "
               "TD5/5 = 60 Gy"),
    "foie": (45.8, 0.12, 0.97,
             "Dawson et al. 2002, modèle LKB sur 203 patients ; RILD grade ≥3"),
    "oesophage": (51.0, 0.32, 0.44,
                  "ajustement par maximum de vraisemblance ; œsophagite aiguë "
                  "grade 2-3"),
    # Nécrose cérébrale après irradiation : paramétrage LKB usuel de QUANTEC.
    # Le jeu utilise « cerveau » dans son interface, d'où cette clé explicite.
    "cerveau": (72.0, 0.15, 0.25,
                "QUANTEC ; nécrose cérébrale radio-induite, TD50 ≈ 72 Gy"),
    "sein": (72.0, 0.16, 0.35,
             "Emami/Burman 1991 ; fibrose mammaire radio-induite"),
    # L'interface parle de colorectal; le paramètre disponible dans les tables
    # cliniques est celui du rectum, utilisé comme approximation du tissu sain.
    "colorectal": (80.0, 0.15, 0.12,
                    "Emami/Burman 1991 ; approximation rectale pour colorectal"),
    # Défaut du jeu : tissu à réponse tardive générique, valeurs médianes.
    "generique_tardif": (60.0, 0.15, 0.30,
                         "tissu sain à réponse tardive générique ; valeurs "
                         "médianes de la table d'Emami"),
}

ALPHA_BETA_LATE = 3.0     # Gy, tissus sains à réponse tardive
ALPHA_BETA_TUMOR = 10.0   # Gy, tumeurs et tissus à réponse aiguë
REPAIR_HALFTIME_H = 1.5   # h, demi-vie de réparation sublétale
REPOP_LATE_PER_DAY = 0.05  # /jour, repopulation des tissus tardifs (0-0.1)


def _phi(x: float) -> float:
    """Fonction de répartition de la loi normale centrée réduite."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass
class Fraction:
    """Une séance d'irradiation."""
    day: float
    dose_gy: float


@dataclass
class ToxicityModel:
    """Suivi de la toxicité cumulée des tissus sains.

    Deux sorties complémentaires :

    - `eqd2()` : dose cumulée en équivalent 2 Gy — la grandeur que les
      radiothérapeutes contraignent réellement.
    - `ntcp()` : probabilité de complication (0-1) via le modèle LKB.

    `sparing_factor` représente la fraction de la dose tumorale effectivement
    reçue par le tissu sain. En radiothérapie conforme moderne, le tissu sain
    reçoit une dose inférieure à celle de la cible ; 0.6 est une valeur
    conservatrice pour un traitement conforme.
    """
    organ: str = "generique_tardif"
    sparing_factor: float = 0.6
    alpha_beta: float = ALPHA_BETA_LATE
    repair_halftime_h: float = REPAIR_HALFTIME_H
    repop_per_day: float = REPOP_LATE_PER_DAY
    fractions: List[Fraction] = field(default_factory=list)

    @property
    def td50(self) -> float:
        return LKB_PARAMETERS[self.organ][0]

    @property
    def m(self) -> float:
        return LKB_PARAMETERS[self.organ][1]

    @property
    def reference(self) -> str:
        return LKB_PARAMETERS[self.organ][3]

    def add_fraction(self, day: float, dose_gy: float) -> None:
        """Enregistre une séance. `dose_gy` est la dose prescrite à la tumeur."""
        if dose_gy > 0:
            self.fractions.append(Fraction(float(day), float(dose_gy)))

    # -- grandeurs radiobiologiques ----------------------------------------
    def bed(self) -> float:
        """BED cumulée du tissu sain, avec réparation incomplète.

        Le terme de réparation incomplète (facteur h de Thames) majore l'effet
        quadratique quand deux fractions sont trop rapprochées pour que la
        réparation sublétale soit complète.
        """
        if not self.fractions:
            return 0.0
        total = 0.0
        mu = log(2.0) / self.repair_halftime_h      # constante de réparation, /h
        for i, f in enumerate(self.fractions):
            d = f.dose_gy * self.sparing_factor
            # effet propre de la fraction
            total += d * (1.0 + d / self.alpha_beta)
            # réparation incomplète vis-à-vis des fractions précédentes
            for g in self.fractions[:i]:
                dg = g.dose_gy * self.sparing_factor
                dt_h = max((f.day - g.day) * 24.0, 0.0)
                total += 2.0 * d * dg / self.alpha_beta * exp(-mu * dt_h)
        # repopulation : le tissu récupère avec le temps écoulé
        span = self.fractions[-1].day - self.fractions[0].day
        total -= self.repop_per_day * max(span, 0.0)
        return max(total, 0.0)

    def eqd2(self) -> float:
        """Dose équivalente en fractions de 2 Gy (unité des contraintes cliniques)."""
        return self.bed() / (1.0 + 2.0 / self.alpha_beta)

    def ntcp(self) -> float:
        """Probabilité de complication du tissu sain, modèle LKB (0-1)."""
        e = self.eqd2()
        if e <= 0:
            return 0.0
        return _phi((e - self.td50) / (self.m * self.td50))

    def toxicity_score(self) -> float:
        """Score de toxicité 0-100 pour le jeu (= NTCP en pourcentage).

        Interprétation directe : 100 signifie complication quasi certaine.
        Le seuil de 100 utilisé comme condition de défaite dans `JEU.md`
        correspond donc à une probabilité de complication de 100 %.
        """
        return 100.0 * self.ntcp()

    def total_physical_dose(self) -> float:
        return sum(f.dose_gy for f in self.fractions)

    def summary(self) -> Dict[str, float]:
        return {
            "dose_physique_gy": self.total_physical_dose(),
            "bed_gy": self.bed(),
            "eqd2_gy": self.eqd2(),
            "ntcp": self.ntcp(),
            "toxicite_0_100": self.toxicity_score(),
            "n_fractions": float(len(self.fractions)),
        }


def conventional_schedule_check() -> Dict[str, float]:
    """Contrôle de cohérence : un schéma standard doit rester sous tolérance.

    30 × 2 Gy = 60 Gy à la tumeur, schéma conventionnel courant. Avec un
    facteur d'épargne de 0.6, le tissu sain reçoit 36 Gy physiques, ce qui doit
    laisser la NTCP à un niveau faible — sinon le modèle serait inutilisable en
    clinique.
    """
    tox = ToxicityModel()
    for i in range(30):
        tox.add_fraction(day=float(i * 7 // 5), dose_gy=2.0)   # 5 séances/semaine
    return tox.summary()
