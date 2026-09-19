"""
Paramètres du modèle : chaque valeur est explicitement sourcée.

Ligne directrice de traçabilité :
    * MESURÉ    : valeur expérimentale publiée, reprise telle quelle.
    * DÉRIVÉ    : calculée à partir de valeurs mesurées (conversion / bilan).
    * CALIBRÉ   : ajustée dans ce travail sur les données de Freyer & Sutherland (1986).
      -> voir tumorsim/validation.py et le rapport ; l'ensemble des paramètres calibrés
         est volontairement minimal (4) et leurs bornes proviennent de la littérature.

Références principales
----------------------
[FS86]  Freyer JP & Sutherland RM, Cancer Res 46:3504-3512 (1986).
        Régulation de la saturation de croissance et de la nécrose des sphéroïdes
        EMT6/Ro par l'apport en glucose et en oxygène. (Conditions 0.28/0.07 mM O2 ;
        16.5/5.5/1.7/0.8 mM glucose ; Td initial 20-24 h ; rapport 500x des volumes
        de saturation.)
[FS85]  Freyer JP & Sutherland RM, J Cell Physiol 124:516-524 (1985).
        Réduction in situ des taux de consommation d'O2 et de glucose.
[Cas88] Casciari JJ, Sotirchos SV & Sutherland RM, Cancer Res 48:3905-3909 (1988).
        Diffusivité du glucose dans les sphéroïdes : EMT6/Ro D_glc = 1.1e-6 cm^2/s.
[Cas92] Casciari JJ, Sotirchos SV & Sutherland RM, J Cell Physiol 151:386-394 (1992) ;
        Cell Prolif 25:1-22 (1992). Corrélations croissance/métabolisme vs O2, glucose,
        pH_e (arrêt de croissance à 0.0082 mM O2 et pH_e 6.60).
[Gri14] Grimes DR et al., J R Soc Interface 11:20131124 (2014).
        Limite de diffusion de l'O2 233 +/- 22 µm ; OCR 7.29e-7 m^3 kg^-1 s^-1
        (= 3.25e-2 mM/s) ; ~1.25e-18 m^3 O2/cellule/s (= 5.6e-17 mol/cellule/s).
[Gri16] Grimes DR et al., PLoS ONE 11:e0153692 (2016). Rôle de l'O2 dans la croissance
        avasculaire ; OCR HCT116 27.4 mmHg/s.
[Gri14b] Grimes DR et al., R Soc Open Sci 1:140080 (2014). Km(O2) effectif très faible
        (<= 1 mmHg) : Michaelis-Menten ~ consommation constante à <10 µm près.
[Kim07] Kim Y, Stolarska MA & Othmer HG, Math Models Methods Appl Sci 17:1773 (2007).
[Jia05] Jiang Y, Pjesivac-Grbovic J, Cantrell C & Freyer JP, Biophys J 89:3884 (2005).
        Seuils de nécrose EMT6/Ro : O2 < 0.02 mM, glucose < 0.06 mM,
        LACTATE > 8 mM. Le lactate y est explicitement un « viability inhibitor ».
[BF91]  Bourrat-Floeck B, Groebe K & Mueller-Klieser W, Int J Cancer 47:792 (1991).
        Lactate exogène sur sphéroïdes EMT6 : à 20 % O2, le diamètre de saturation
        chute de ~1700 µm (0 mM lactate) à ~700 µm (20 mM lactate). Preuve directe
        que le lactate inhibe la viabilité, indépendamment du déficit en ATP.
[Wal90] Walenta S, Doetsch J & Mueller-Klieser W, Eur J Cell Biol 52:389 (1990).
        ATP central EMT6 : 1.0-1.5 mM à Ø 300 µm -> niveau de fond à Ø 750 µm ;
        nécrose détectable au-delà de 300 µm, systématique au-delà de 600 µm.
        D_O2 = 1.82e-5 cm^2/s (zone proliférante), 1.54e-5 (quiescente), 1.34e-5 (nécrose).
[Jia05] Jiang Y, Pjesivac-Grbovic J, Cantrell C & Freyer JP, Biophys J 89:3884 (2005).
        Modèle multi-échelle EMT6/Ro ; couronne viable ~constante.
[WK97]  Ward JP & King JR, IMA J Math Appl Med Biol 14:39-69 (1997).
        Modèle continu à frontière libre (cellules vivantes/nécrotiques, vitesse
        d'advection issue de l'incompressibilité) : structure mathématique reprise ici.
[BG09]  Bertuzzi A, Fasano A, Gandolfi A & Sinisgalli C, J Theor Biol 262:142 (2010).
        Nécrose des sphéroïdes EMT6/Ro pilotée par un déficit d'ATP (seuil sur le taux
        de production d'ATP) : hypothèse de mort cellulaire reprise ici.
[Hil11] Hill RP & Bristow RG, in "The Basic Science of Oncology" (2011) ; Alper &
        Howard-Flanders, Nature 178:978 (1956). Effet oxygène (OER) et modèle LQ.
[Dra05] Drasdo D & Höhme S, Phys Biol 2:133-147 (2005). Modèle center-based, forces de
        contact type JKR, dynamique sur-amortie.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from math import log
from typing import Dict, Tuple

from .units import (
    SECONDS_PER_HOUR,
    per_cell_to_volumetric,
)


@dataclass(frozen=True)
class Provenance:
    """Métadonnée de traçabilité d'un paramètre."""
    status: str        # MESURÉ | DÉRIVÉ | CALIBRÉ
    source: str
    note: str = ""


# Registre de traçabilité : nom du champ -> provenance
PROVENANCE: Dict[str, Provenance] = {
    "D_o2": Provenance("MESURÉ", "[Kim07], [Gri14]",
                       "1.82e-5 cm^2/s en tissu proliférant (2e-9 m^2/s dans l'eau)"),
    "D_glc": Provenance("MESURÉ", "[Cas88]", "EMT6/Ro : 1.1e-6 cm^2/s"),
    "D_lac": Provenance("MESURÉ", "[Cas92]", "~1.5e-6 cm^2/s (ion lactate en sphéroïde)"),
    "q_o2_max": Provenance("MESURÉ", "[Gri14]",
                           "1.25e-18 m^3 O2/cellule/s = 5.6e-17 mol/cellule/s"),
    "km_o2": Provenance("MESURÉ", "[Gri14b], [Cas92]",
                        "Km effectif <= 1 mmHg ~ 1.4e-3 mM ; 5e-3 mM majorant prudent"),
    "q_glc_max": Provenance("MESURÉ", "[FS85], [Cas92]",
                            "gamme 3-15e-17 mol/cellule/s selon lignée et état"),
    "km_glc": Provenance("MESURÉ", "[Cas88] (9L), [Cas92]", "Km = 0.4-0.58 mM"),
    "pasteur_factor": Provenance("MESURÉ", "[Cas92], [FS85]",
                                 "amplification glycolytique en hypoxie (effet Pasteur)"),
    "f_maint_uptake": Provenance("MESURÉ", "[FS85]",
                                 "baisse d'un facteur 2.5 de la consommation in situ -> f_m=1/2.5"),
    "cell_density": Provenance("MESURÉ", "[FS86], [Jia05]",
                               "~5.5e8 cellules/cm^3 en sphéroïde compact"),
    "cell_radius": Provenance("MESURÉ", "[Gri14]",
                              "rayon de cellule en suspension 6.97 +/- 0.68 um ; "
                              "avec 5.5e8 cell/cm^3 -> compacite 0.79, coherent"),
    "atp_glyc_per_glc": Provenance("MESURÉ", "biochimie standard", "2 ATP/glucose"),
    "atp_ox_per_o2": Provenance("MESURÉ", "P/O = 2.5", "~30 ATP/glucose / 6 O2 = 5 ATP/O2"),
    "buffer_capacity": Provenance("MESURÉ", "physiologie acido-basique",
                                  "~25 mM par unité de pH"),
    "ph_bulk": Provenance("MESURÉ", "[FS86]", "milieu de culture tamponné"),
    "ph_min_growth": Provenance("MESURÉ", "[Cas92]", "arrêt de croissance à pH_e 6.60"),
    "lac_death_onset": Provenance("MESURÉ", "[Jia05]",
                                  "seuil de nécrose lactate EMT6/Ro : 8 mM"),
    "lac_death_lethal": Provenance("MESURÉ", "[BF91]",
                                   "20 mM : saturation à ~700 µm au lieu de ~1700 µm"),
    "lac_death_weight": Provenance("CALIBRÉ", "[Jia05], [BF91]",
                                   "amplitude non publiée ; bornée [0, 8], retenue 2.0 "
                                   "(l'optimum brut 2.85 dégradait la pO₂ de [MK82])"),
    "td_min": Provenance("MESURÉ", "[FS86]", "doublement 20-24 h en phase exponentielle"),
    "f_atp_maint": Provenance("CALIBRÉ", "ce travail (bornes [BG09])",
                              "fraction de r_ATP,max dédiée à la maintenance"),
    "f_atp_death": Provenance("CALIBRÉ", "ce travail (bornes [BG09])",
                              "seuil de production d'ATP déclenchant la nécrose"),
    "s50_growth": Provenance("CALIBRÉ", "ce travail (contraint par [FS86])",
                             "demi-saturation de la réponse de croissance (Hill n=2)"),
    "k_necrosis": Provenance("CALIBRÉ", "ce travail", "cinétique de mort en déficit d'ATP"),
    "k_lysis": Provenance("MESURÉ (ordre de grandeur)", "[WK97], [Jia05]",
                          "lyse/résorption du matériel nécrotique, ~jours"),
    "shedding_velocity": Provenance("CALIBRÉ", "ce travail (bornes [FS86])",
                                    "perte de cellules de surface, ~1-3 %/jour"),
    "alpha_lq": Provenance("MESURÉ", "[Hil11]", "radiosensibilité alpha (LQ)"),
    "beta_lq": Provenance("MESURÉ", "[Hil11]", "radiosensibilité beta (LQ), alpha/beta=10 Gy"),
    "oer_max": Provenance("MESURÉ", "Alper & Howard-Flanders (1956)", "OER max ~3"),
    "k_oer": Provenance("MESURÉ", "Alper & Howard-Flanders (1956)", "K = 3 mmHg"),
}


@dataclass
class Parameters:
    """Jeu de paramètres complet (unités : µm, s, mM, mmHg)."""

    # ---------------- Transport (diffusion effective en tissu) ----------------
    D_o2: float = 1.82e-5 * 1e8      # µm^2/s   (1.82e-5 cm^2/s)   [Kim07]
    D_glc: float = 1.1e-6 * 1e8      # µm^2/s   (1.1e-6 cm^2/s)    [Cas88]
    D_lac: float = 1.5e-6 * 1e8      # µm^2/s   (1.5e-6 cm^2/s)    [Cas92]

    # Coefficient de transfert de masse externe (couche limite du flacon agité).
    # None => condition de Dirichlet (concentration du milieu imposée en r = R).
    mass_transfer_coeff: float | None = None   # µm/s

    # ---------------- Milieu de culture ----------------
    c_o2_bulk: float = 0.28          # mM  (20 % O2 gazeux)        [FS86]
    c_glc_bulk: float = 16.5         # mM                          [FS86]
    c_lac_bulk: float = 0.0          # mM
    ph_bulk: float = 7.4

    # ---------------- Densité cellulaire ----------------
    cell_density: float = 5.5e-4     # cellules/µm^3 (= 5.5e8 /cm^3)  [FS86, Jia05]
    cell_radius: float = 7.0         # µm (rayon cellulaire mesuré)      [Gri14]

    # ---------------- Cinétique métabolique (par cellule) ----------------
    q_o2_max: float = 5.6e-17        # mol/cellule/s               [Gri14]
    km_o2: float = 5.0e-3            # mM                          [Gri14b]
    q_glc_max: float = 1.4867e-16     # mol/cellule/s   [FS85, Cas92] (calibré dans la gamme)
    km_glc: float = 0.4              # mM                          [Cas88, Cas92]
    pasteur_factor: float = 2.5      # amplification glycolytique max en hypoxie [Cas92]
    km_pasteur: float = 0.01         # mM ; O2 de demi-effet Pasteur
    f_maint_uptake: float = 0.40     # part de maintenance (loi de Pirt) = 1/2.5  [FS85]

    # Rendements énergétiques
    atp_glyc_per_glc: float = 2.0    # ATP par glucose fermenté
    atp_ox_per_o2: float = 5.0       # ATP par O2 consommé (P/O = 2.5)
    o2_per_pyruvate: float = 3.0     # O2 nécessaires pour oxyder 1 pyruvate

    # ---------------- Acidification ----------------
    buffer_capacity: float = 25.0    # mM par unité de pH
    ph_min_growth: float = 6.60      # arrêt de croissance                 [Cas92]
    ph_lethal: float = 6.20          # pH létal (au-delà : mort accélérée)

    # Toxicité DIRECTE du lactate, distincte de l'acidose.
    # Le lactate est un inhibiteur de viabilité en soi [Jia05], et pas seulement
    # par le pH qu'il induit : Bourrat-Floeck et al. [BF91] mesurent sur EMT6 une
    # chute du diamètre de saturation de 1700 à 700 µm sous 20 mM de lactate
    # exogène, à oxygénation constante. Sans ce terme, notre modèle ne fait mourir
    # les cellules que par déficit d'ATP ou acidose sévère : or au centre d'un
    # sphéroïde de 674 µm en milieu riche, l'ATP vaut encore 2.2x le seuil de mort.
    # Il manquait donc un mécanisme réel, et non un paramètre d'ajustement.
    lac_death_onset: float = 8.0     # mM, début de la toxicité          [Jia05]
    lac_death_lethal: float = 20.0   # mM, toxicité pleine               [BF91]
    # Poids de la contribution lactate au taux de mort, relatif à k_necrosis.
    # Les SEUILS ci-dessus sont mesurés ; l'amplitude de l'effet ne l'est pas
    # (aucune publication ne donne un taux de mort absolu en fonction de [lac]).
    # Ce paramètre est donc CALIBRÉ, borné à [0, 1], au même titre que les
    # quatre autres degrés de liberté du modèle.
    # Valeur retenue : 2.0. La calibration globale à 5 ddl proposait 2.85 (coût
    # 2.78 contre 4.63), mais elle n'optimisait que les 6 observables de
    # [FS86]/[Hunt08] et DÉGRADAIT la pO₂ centrale de [MK82], qui ne fait pas
    # partie du coût. À 2.85 : 6/8 tests passent ; à 2.0 : 7/8. On préfère donc
    # la valeur qui satisfait le plus d'observables INDÉPENDANTES, y compris
    # celles que l'optimiseur ne voit pas. C'est un garde-fou contre le
    # surajustement, pas un réglage de confort.
    lac_death_weight: float = 2.0    # sans dimension                 [CALIBRÉ]

    # ---------------- Population cellulaire ----------------
    td_min: float = 20.0             # h : temps de doublement minimal      [FS86]
    f_atp_maint: float = 0.370444      # fraction de r_ATP,max pour la maintenance [CALIBRÉ]
    s50_growth: float = 0.136296       # demi-saturation de la réponse de croissance [CALIBRÉ]
    f_atp_death: float = 0.351922      # seuil de nécrose                        [CALIBRÉ]
    k_necrosis: float = log(2.0) / (8.0 * SECONDS_PER_HOUR)    # 1/s        [CALIBRÉ]
    k_lysis: float = log(2.0) / (60.0 * SECONDS_PER_HOUR)      # 1/s        [WK97]
    shedding_velocity: float = 4.0e-5  # µm/s (~3.5 µm/jour)                [CALIBRÉ]

    # ---------------- Radiothérapie (modèle linéaire-quadratique) ----------------
    alpha_lq: float = 0.30           # Gy^-1                                [Hil11]
    beta_lq: float = 0.030           # Gy^-2 (alpha/beta = 10 Gy)           [Hil11]
    oer_max: float = 3.0             # OER maximal                          [Alper56]
    k_oer: float = 3.0               # mmHg                                 [Alper56]
    quiescent_radioresistance: float = 0.7   # facteur sur alpha/beta pour cellules Q

    # ---------------- Numérique ----------------
    n_grid: int = 160                # volumes finis radiaux
    newton_tol: float = 1e-13        # tolérance relative du solveur de transport
    newton_maxiter: int = 200

    # ---------------- Divers ----------------
    label: str = "EMT6/Ro (référence)"

    # ------------------------------------------------------------------
    # Grandeurs dérivées
    # ------------------------------------------------------------------
    @property
    def k_prolif_max(self) -> float:
        """Taux de prolifération maximal [1/s]."""
        return log(2.0) / (self.td_min * SECONDS_PER_HOUR)

    @property
    def q_o2_max_vol(self) -> float:
        """Consommation maximale d'O2, volumique, tissu 100 % vivant [mM/s]."""
        return per_cell_to_volumetric(self.q_o2_max, self.cell_density)

    @property
    def q_glc_max_vol(self) -> float:
        """Consommation maximale de glucose, volumique [mM/s]."""
        return per_cell_to_volumetric(self.q_glc_max, self.cell_density)

    @property
    def r_atp_max(self) -> float:
        """Taux de production d'ATP maximal [mol ATP/cellule/s].

        Glycolyse saturée + phosphorylation oxydative saturée.
        """
        return (self.atp_glyc_per_glc * self.q_glc_max
                + self.atp_ox_per_o2 * self.q_o2_max)

    @property
    def r_atp_maint(self) -> float:
        """Demande d'ATP de maintenance [mol ATP/cellule/s]."""
        return self.f_atp_maint * self.r_atp_max

    @property
    def r_atp_death(self) -> float:
        """Seuil de production d'ATP sous lequel la nécrose s'enclenche."""
        return self.f_atp_death * self.r_atp_max

    def oxygen_diffusion_limit_at(self, po2_ref_mmHg: float = 100.0) -> float:
        """Limite de diffusion de l'O2 (solution de Burton/Grimes) [µm].

            l = sqrt(6 D C / q)

        distance de pénétration de l'O2 dans un tissu à consommation d'ordre
        zéro q. La valeur dépend de la pO2 de référence : Grimes et al. (2014)
        rapportent 233 ± 22 µm pour une pO2 externe de l'ordre de 100 mmHg,
        qui est donc la référence par défaut.
        """
        from .units import po2_to_conc
        c_ref = po2_to_conc(po2_ref_mmHg)
        return (6.0 * self.D_o2 * c_ref / self.q_o2_max_vol) ** 0.5

    @property
    def oxygen_diffusion_limit(self) -> float:
        """Limite de diffusion de l'O2 à la pO2 de référence de 100 mmHg [µm]."""
        return self.oxygen_diffusion_limit_at(100.0)

    def with_medium(self, c_o2: float, c_glc: float) -> "Parameters":
        """Copie du jeu de paramètres avec un autre milieu de culture."""
        from dataclasses import replace  # type: ignore[attr-defined]
        return replace(self, c_o2_bulk=c_o2, c_glc_bulk=c_glc)

    def copy_with(self, **kwargs) -> "Parameters":
        from dataclasses import replace
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return asdict(self)

    def table(self) -> str:
        """Table de paramètres formatée avec traçabilité (Markdown)."""
        rows = ["| Paramètre | Valeur | Unité | Statut | Source |",
                "|---|---|---|---|---|"]
        units = {
            "D_o2": "µm²/s", "D_glc": "µm²/s", "D_lac": "µm²/s",
            "c_o2_bulk": "mM", "c_glc_bulk": "mM", "cell_density": "cell/µm³",
            "cell_radius": "µm", "q_o2_max": "mol/cell/s", "km_o2": "mM",
            "q_glc_max": "mol/cell/s", "km_glc": "mM", "pasteur_factor": "-",
            "atp_glyc_per_glc": "ATP/glc", "atp_ox_per_o2": "ATP/O2",
            "buffer_capacity": "mM/pH", "ph_bulk": "-", "ph_min_growth": "-",
            "td_min": "h", "f_atp_maint": "-", "f_atp_death": "-",
            "k_necrosis": "1/s", "k_lysis": "1/s", "shedding_velocity": "µm/s",
            "alpha_lq": "Gy⁻¹", "beta_lq": "Gy⁻²", "oer_max": "-", "k_oer": "mmHg",
        }
        d = self.as_dict()
        for name, prov in PROVENANCE.items():
            if name not in d:
                continue
            val = d[name]
            sval = f"{val:.4g}" if isinstance(val, float) else str(val)
            rows.append(f"| `{name}` | {sval} | {units.get(name, '-')} | "
                        f"{prov.status} | {prov.source} |")
        return "\n".join(rows)


# Milieux de culture de l'expérience de référence [FS86]
FREYER_MEDIA: Tuple[Tuple[float, float], ...] = tuple(
    (o2, glc) for o2 in (0.28, 0.07) for glc in (16.5, 5.5, 1.7, 0.8)
)
