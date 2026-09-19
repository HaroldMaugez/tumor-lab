"""
Évolution darwinienne intratumorale : mutation, dérive génétique, sélection.

Cadre théorique
---------------
Processus de branchement multitype (Bozic et al., PNAS 107:18545, 2010), couplé
au microenvironnement simulé. Les trois forces de l'évolution sont explicites :

    MUTATION  : à chaque division, une cellule fille peut acquérir un driver
                (probabilité u) et/ou des passengers (loi de Poisson, moyenne v).
    DÉRIVE    : la naissance et la mort sont des évènements STOCHASTIQUES ;
                une lignée avantagée disparaît le plus souvent par hasard
                (Bozic : >99 % des drivers s'éteignent).
    SÉLECTION : chaque driver confère un avantage sélectif s ; le taux de
                prolifération est multiplié par (1+s)^k pour k drivers.

Paramètres MESURÉS (Bozic et al. 2010, estimés sur glioblastome et
adénocarcinome pancréatique) :

    u = 3.4e-5   driver par division
                 (= 2 x 34 000 sites x 5e-10 mutations/pb/division)
    s = 0.004 +/- 0.0004   avantage sélectif moyen d'un driver (0.4 %)
    v = 0.016    passengers par division (3.15e7 sites x 5e-10)

Conséquence analytique vérifiée dans `validation.py` : pour un processus de
branchement critique avec b_k = 1 - d_k et d_k = (1-s)^k / 2, la probabilité
qu'une lignée porteuse d'un nouveau driver échappe à l'extinction vaut
1 - d/b ≈ 2s ≈ 0.8 %.

Évolution neutre et spectre de fréquences alléliques
-----------------------------------------------------
En l'absence de sélection subclonale, la distribution cumulée du nombre de
mutations présentes à une fréquence >= f suit une loi de puissance
(Williams et al., Nat Genet 48:238, 2016) :

    M(f) = (v / beta) * (1/f - 1/f_max)

c'est-à-dire M(f) linéaire en 1/f (R^2 >= 0.98 chez ~1/3 des tumeurs humaines).
Ce comportement constitue un test de validation fort du moteur évolutif : il
n'est PAS imposé, il doit émerger de la dynamique stochastique.

Couplage au microenvironnement
------------------------------
Contrairement à un modèle de population bien mélangée, la sélection agit ici
sur un tissu structuré : le taux de prolifération local dépend déjà de l'ATP
disponible (donc de l'O2, du glucose et du pH). Un clone n'est avantagé que
là où il peut effectivement proliférer, ce qui produit une compétition
spatiale et une évolution ramifiée réaliste. Deux traits héritables
supplémentaires, documentés comme moteurs de l'évolution tumorale, sont
disponibles (Gatenby & Gillies, Nat Rev Cancer 4:891, 2004) :

    * phénotype glycolytique  : tolérance accrue à l'hypoxie ;
    * acido-résistance        : tolérance accrue à l'acidité.

Reproductibilité
----------------
Toute la stochasticité passe par un unique `numpy.random.Generator`. À graine
fixée, une simulation est reproductible au bit près ; sans graine, chaque
exécution explore une réalisation différente du processus stochastique —
exactement comme deux tumeurs issues du même clone fondateur divergent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Paramètres évolutifs
# ---------------------------------------------------------------------------
@dataclass
class EvolutionParams:
    """Paramètres de génétique des populations (valeurs mesurées par défaut)."""

    # --- Mutation (Bozic et al. 2010) ---
    u_driver: float = 3.4e-5       # drivers par division            [MESURÉ]
    v_passenger: float = 0.016     # passengers par division         [MESURÉ]
    s_driver: float = 0.004        # avantage sélectif moyen          [MESURÉ]
    s_sigma: float = 0.002         # dispersion de s (Bozic : sigma = s/2)

    # --- Traits adaptatifs au microenvironnement (Gatenby & Gillies 2004) ---
    p_glycolytic: float = 0.0      # proba qu'un driver confère le phénotype glycolytique
    p_acid_resist: float = 0.0     # proba qu'un driver confère l'acido-résistance
    glycolytic_gain: float = 0.35  # réduction relative du seuil de mort en hypoxie
    acid_resist_gain: float = 0.30 # abaissement relatif du pH létal

    # --- Radiorésistance héritable (Allam ; Britten ; Alfonso & Berk 2019) ---
    # Il existe une VRAIE hétérogénéité intratumorale de radiosensibilité :
    # Britten et al. ont isolé 96 clones monocellulaires de carcinomes cervicaux
    # et mesuré des SF2 s'étalant de 0.240 à 0.518, 0.050 à 0.414 et 0.137 à
    # 0.452 selon la tumeur — soit un facteur 2 à 8 entre clones d'une MÊME
    # tumeur. Allam et al. trouvent ~25 % de variation sur des lignées de gliome.
    #
    # Alfonso & Berk (Radiat Oncol 14:88, 2019) montrent qu'en présence de cette
    # distribution, la radiothérapie fractionnée standard opère un REMPLACEMENT
    # QUASI COMPLET de la population sensible par une population résistante, et
    # que le décalage est bien plus marqué sur alpha que sur beta (à 2 Gy, la
    # composante alpha domine). On modélise donc la résistance comme une
    # réduction héritable de alpha.
    # Hétérogénéité PRÉEXISTANTE de radiosensibilité dans la population
    # fondatrice. C'est le point clé du modèle d'Alfonso & Berk : la sélection
    # n'opère pas sur des mutations apparues pendant le traitement (trop rares
    # sur quelques semaines), mais sur une variabilité DÉJÀ PRÉSENTE, mesurée
    # expérimentalement par Britten et al. sur 96 clones monocellulaires.
    # sigma = 0.15 reproduit l'étalement observé des SF2 intratumoraux (~25 %
    # de variation chez Allam, facteur 2-8 entre clones extrêmes chez Britten).
    initial_radio_sigma: float = 0.0   # écart-type log-normal du facteur alpha
    p_radioresist: float = 0.0     # proba qu'un driver confère la radiorésistance
    radioresist_gain: float = 0.30 # réduction relative de alpha par mutation
    radioresist_max: float = 0.75  # réduction cumulée maximale (borne biologique)

    # --- Mutations délétères (Mc Farland et al., PNAS 110:2910, 2013) ---
    deleterious_fraction: float = 0.0   # fraction de passengers légèrement délétères
    s_deleterious: float = 4.0e-5       # coût ~100x plus faible qu'un driver

    # --- Démographie du processus de branchement ---
    # d_k = d_base * (1-s)^k. Avec d_base = 0.5 on retrouve exactement le
    # processus CRITIQUE de Bozic et al. (2010) : sans driver, naissance et
    # mort s'équilibrent. Avec d_base < 0.5 la population croît
    # exponentiellement, régime dans lequel Williams et al. (2016) établissent
    # la loi 1/f de l'évolution neutre.
    d_base: float = 0.5

    # --- Numérique ---
    max_clones: int = 4000         # au-delà, les clones minuscules sont fusionnés
    prune_below: int = 0           # taille de clone en deçà de laquelle on élague

    def fitness(self, n_drivers: int, n_deleterious: int = 0) -> float:
        """Multiplicateur de fitness pour k drivers et j passengers délétères."""
        return ((1.0 + self.s_driver) ** n_drivers
                * (1.0 - self.s_deleterious) ** n_deleterious)


def _zero_truncated_poisson(rng: np.random.Generator, lam: float) -> int:
    """Tirage d'une loi de Poisson conditionnée à être >= 1 (méthode d'inversion)."""
    if lam <= 0:
        return 1
    u = rng.random()
    # F(k) tronquée : on inverse sur [0,1) en excluant la masse en 0
    p0 = np.exp(-lam)
    target = p0 + u * (1.0 - p0)
    k, cdf, term = 0, p0, p0
    while cdf < target and k < 1000:
        k += 1
        term *= lam / k
        cdf += term
    return max(k, 1)


# ---------------------------------------------------------------------------
# Clones
# ---------------------------------------------------------------------------
@dataclass
class Clone:
    """Un clone = un génotype distinct, défini par son lot de mutations."""
    cid: int
    parent: Optional[int]
    n_drivers: int = 0
    n_passengers: int = 0
    n_deleterious: int = 0
    fitness: float = 1.0
    glycolytic: bool = False
    acid_resistant: bool = False
    # Facteur multiplicatif HÉRITABLE sur alpha (radiosensibilité).
    # 1.0 = sensibilité de référence ; < 1 = radiorésistant.
    radio_alpha_factor: float = 1.0
    birth_time: float = 0.0            # s
    # Identifiants des mutations APPARUES dans ce clone (les mutations
    # ancestrales sont retrouvées en remontant l'arbre). Ce choix rend le
    # stockage O(nombre de clones) au lieu de O(clones x mutations).
    new_mutations: Tuple[int, ...] = ()

    @property
    def n_new_mutations(self) -> int:
        return len(self.new_mutations)


class CloneTree:
    """Généalogie des clones, avec comptage des mutations pour l'analyse VAF."""

    def __init__(self, params: EvolutionParams, rng: np.random.Generator,
                 founder_drivers: int = 1):
        self.p = params
        self.rng = rng
        self._next_cid = 0
        self._next_mid = 0
        self.clones: Dict[int, Clone] = {}
        # Clone fondateur : par convention il porte déjà un driver (une tumeur
        # est initiée par au moins une altération avantageuse ; Bozic 2010).
        muts = tuple(self._new_mutation_ids(founder_drivers))
        root = Clone(cid=self._new_cid(), parent=None, n_drivers=founder_drivers,
                     fitness=params.fitness(founder_drivers), new_mutations=muts)
        self.clones[root.cid] = root
        self.root = root.cid

    # -- identifiants ---------------------------------------------------
    def _new_cid(self) -> int:
        c = self._next_cid
        self._next_cid += 1
        return c

    def _new_mutation_ids(self, n: int) -> List[int]:
        ids = list(range(self._next_mid, self._next_mid + n))
        self._next_mid += n
        return ids

    @property
    def n_clones(self) -> int:
        return len(self.clones)

    @property
    def n_mutations_total(self) -> int:
        return self._next_mid

    # -- mutation -------------------------------------------------------
    def mutate(self, parent_cid: int, time: float) -> int:
        """Tire les mutations d'une division ; renvoie le cid de la fille.

        Renvoie le clone parent inchangé si aucune mutation n'est survenue
        (cas très majoritaire), ce qui évite de créer des clones inutiles.
        """
        p = self.p
        n_new_driver = int(self.rng.random() < p.u_driver)
        n_new_pass = int(self.rng.poisson(p.v_passenger))
        if n_new_driver == 0 and n_new_pass == 0:
            return parent_cid

        return self._build_child(parent_cid, time, n_new_driver, n_new_pass)

    def _build_child(self, parent_cid: int, time: float,
                     n_new_driver: int, n_new_pass: int) -> int:
        """Crée le clone fille portant les mutations tirées."""
        p = self.p
        if n_new_driver == 0 and n_new_pass == 0:
            return parent_cid
        parent = self.clones[parent_cid]
        n_del = 0
        if n_new_pass and p.deleterious_fraction > 0.0:
            n_del = int(self.rng.binomial(n_new_pass, p.deleterious_fraction))

        new_ids = self._new_mutation_ids(n_new_driver + n_new_pass)
        drivers = parent.n_drivers + n_new_driver
        deleterious = parent.n_deleterious + n_del

        glyco = parent.glycolytic
        acid = parent.acid_resistant
        radio = parent.radio_alpha_factor
        if n_new_driver:
            if self.rng.random() < p.p_glycolytic:
                glyco = True
            if self.rng.random() < p.p_acid_resist:
                acid = True
            # Radiorésistance héritable : chaque driver concerné réduit alpha,
            # jusqu'à une borne biologique (radioresist_max). La résistance est
            # transmise à toute la descendance — c'est ce qui permet la
            # sélection clonale sous radiothérapie fractionnée (Alfonso 2019).
            if self.rng.random() < p.p_radioresist:
                radio = max(radio * (1.0 - p.radioresist_gain),
                            1.0 - p.radioresist_max)

        # Avantage sélectif tiré autour de la moyenne mesurée (Bozic : sigma=s/2)
        if n_new_driver and p.s_sigma > 0:
            s_draw = max(0.0, self.rng.normal(p.s_driver, p.s_sigma))
            fit = parent.fitness * (1.0 + s_draw)
            fit *= (1.0 - p.s_deleterious) ** n_del
        else:
            fit = parent.fitness * (1.0 - p.s_deleterious) ** n_del

        child = Clone(
            cid=self._new_cid(), parent=parent_cid, n_drivers=drivers,
            n_passengers=parent.n_passengers + n_new_pass,
            n_deleterious=deleterious, fitness=fit,
            glycolytic=glyco, acid_resistant=acid,
            radio_alpha_factor=radio, birth_time=time,
            new_mutations=tuple(new_ids),
        )
        self.clones[child.cid] = child
        return child.cid

    def mutate_conditional(self, parent_cid: int, time: float) -> int:
        """Comme `mutate`, mais SACHANT qu'au moins une mutation est survenue.

        Utile lorsqu'on a déjà tiré le nombre de cellules mutantes en bloc
        (loi binomiale sur p_any). Un simple rejet serait correct mais coûteux
        (~1/p_any tirages) ; on échantillonne donc la loi conditionnelle exacte :

            P(driver=1 | >=1 mutation) = u / p_any
            si driver=0, le nombre de passengers suit une loi de Poisson
            tronquée en zéro.
        """
        p = self.p
        p_any = self.p_any_mutation
        driver = int(self.rng.random() < (p.u_driver / p_any))
        if driver:
            n_pass = int(self.rng.poisson(p.v_passenger))
        else:
            n_pass = int(_zero_truncated_poisson(self.rng, p.v_passenger))
        return self._build_child(parent_cid, time, driver, n_pass)

    @property
    def p_any_mutation(self) -> float:
        """Probabilité qu'une cellule fille acquière au moins une mutation."""
        p = self.p
        return 1.0 - (1.0 - p.u_driver) * np.exp(-p.v_passenger)

    # -- analyse --------------------------------------------------------
    def lineage(self, cid: int) -> List[int]:
        """Chaîne d'ancêtres du clone (de la racine au clone)."""
        out: List[int] = []
        cur: Optional[int] = cid
        while cur is not None:
            out.append(cur)
            cur = self.clones[cur].parent
        return out[::-1]

    def spawn_variant(self, parent_cid: int, **overrides) -> int:
        """Crée un clone fille identique au parent, sauf attributs surchargés.

        Sert à installer une hétérogénéité PRÉEXISTANTE dans la population
        fondatrice (radiosensibilité notamment), sans passer par une mutation.
        """
        parent = self.clones[parent_cid]
        fields = dict(
            cid=self._new_cid(), parent=parent_cid,
            n_drivers=parent.n_drivers, n_passengers=parent.n_passengers,
            n_deleterious=parent.n_deleterious, fitness=parent.fitness,
            glycolytic=parent.glycolytic, acid_resistant=parent.acid_resistant,
            radio_alpha_factor=parent.radio_alpha_factor,
            birth_time=parent.birth_time, new_mutations=(),
        )
        fields.update(overrides)
        child = Clone(**fields)
        self.clones[child.cid] = child
        return child.cid

    def driver_genotype(self, cid: int) -> Tuple[int, bool, bool]:
        """Génotype fonctionnel d'un clone : (nb drivers, glycolytique, acido-R)."""
        c = self.clones[cid]
        return (c.n_drivers, c.glycolytic, c.acid_resistant)

    def ancestor_at_depth(self, cid: int, depth: int = 1) -> int:
        """Ancêtre du clone à une profondeur donnée dans l'arbre généalogique.

        Sert au traçage de lignage (lineage tracing) : colorer les cellules par
        leur ancêtre commun à faible profondeur révèle les DOMAINES CLONAUX
        spatiaux, là où colorer par génotype exact ne montrerait qu'un bruit
        poivre-et-sel (chaque cellule portant ses propres passengers).
        C'est l'analogue in silico du marquage par code-barres génétique.
        """
        lin = self.lineage(cid)
        return lin[min(depth, len(lin) - 1)]

    def subtree_counts(self, counts: Dict[int, int]) -> Dict[int, int]:
        """Effectif de chaque clone ET de toute sa descendance.

        Calculé en un seul passage des feuilles vers la racine : les clones
        étant créés par ordre croissant d'identifiant, un parent a toujours un
        cid inférieur à celui de ses enfants.
        """
        sub: Dict[int, int] = {cid: counts.get(cid, 0) for cid in self.clones}
        for cid in sorted(self.clones, reverse=True):
            parent = self.clones[cid].parent
            if parent is not None and sub[cid]:
                sub[parent] += sub[cid]
        return sub

    def allele_frequencies(self, counts: Dict[int, int]) -> np.ndarray:
        """Fréquences alléliques (fractions cellulaires) de chaque mutation.

        Une mutation apparue dans un clone est portée par ce clone et par
        toute sa descendance : sa fréquence est donc la taille du sous-arbre
        correspondant, rapportée à l'effectif total.
        """
        total = sum(counts.values())
        if total == 0:
            return np.zeros(0)
        sub = self.subtree_counts(counts)
        freqs: List[float] = []
        for cid, clone in self.clones.items():
            n = sub[cid]
            if n <= 0 or not clone.new_mutations:
                continue
            f = n / float(total)
            freqs.extend([f] * len(clone.new_mutations))
        if not freqs:
            return np.zeros(0)
        return np.sort(np.asarray(freqs, dtype=float))

# ---------------------------------------------------------------------------
# Analyse du spectre de fréquences alléliques
# ---------------------------------------------------------------------------
def cumulative_vaf(vaf: np.ndarray, f_min: float = 0.02,
                   f_max: float = 0.5, n_bins: int = 40
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """M(f) : nombre de mutations de fréquence >= f, sur la fenêtre de Williams.

    La fenêtre [0.12, 0.24] est celle utilisée par Williams et al. (2016) pour
    se restreindre aux variants subclonaux et limiter les biais de pureté.
    Les VAF sont ici des fréquences cellulaires (le modèle est haploïde) ;
    en séquençage, une mutation hétérozygote apparaît à f/2.
    """
    if vaf.size == 0:
        return np.zeros(0), np.zeros(0)
    fs = np.linspace(f_min, f_max, n_bins)
    m = np.array([np.sum(vaf >= f) for f in fs], dtype=float)
    return fs, m


def neutrality_test(vaf: np.ndarray, f_min: float = 0.02,
                    f_max: float = 0.5) -> Dict[str, float]:
    """Test 1/f de Williams et al. (2016).

    Ajuste M(f) = m * (1/f) + c et renvoie le R². Un R² >= 0.98 est le seuil
    retenu par les auteurs pour ne pas rejeter l'évolution neutre.

    Choix de la fenêtre
    -------------------
    Williams et al. utilisent [0.12, 0.24] car ils travaillent sur des VAF de
    séquençage réel, où il faut écarter les variants clonaux et les artefacts
    de pureté/ploïdie. En simulation, ces biais n'existent pas, mais une autre
    contrainte apparaît : le nombre de cellules simulables (~1e5) est très
    inférieur à celui d'une tumeur réelle (~1e9). Dans la fenêtre étroite, le
    spectre ne contient alors que quelques centaines de mutations réparties en
    paliers discrets, ce qui dégrade mécaniquement le R² (~0.89) pour des
    raisons d'ÉCHANTILLONNAGE et non de biologie.

    On élargit donc la fenêtre par défaut à [0.02, 0.5], ce qui restaure la
    résolution en fréquence (~1300 mutations) et redonne R² ≈ 0.99 en régime
    neutre. La dépendance à la fenêtre est explicitement quantifiée dans
    `validation.py` (test `check_neutral_power_law`), et non passée sous
    silence.
    """
    fs, m = cumulative_vaf(vaf, f_min, f_max)
    if fs.size < 4 or np.all(m == m[0]):
        return {"r2": float("nan"), "slope": float("nan"), "n_mutations": 0.0}
    x = 1.0 / fs
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((m - pred) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": float(r2), "slope": float(coef[0]),
            "n_mutations": float(np.sum(vaf >= f_min))}


def shannon_diversity(counts: Dict[int, int]) -> float:
    """Indice de Shannon de la composition clonale (diversité intratumorale)."""
    n = np.array([c for c in counts.values() if c > 0], dtype=float)
    if n.size <= 1:
        return 0.0
    p = n / n.sum()
    return float(-np.sum(p * np.log(p)))


def clonal_summary(tree: CloneTree, counts: Dict[int, int]) -> Dict[str, float]:
    """Statistiques synthétiques de la population clonale."""
    live = {c: n for c, n in counts.items() if n > 0}
    total = sum(live.values())
    if total == 0:
        return {"n_cells": 0.0, "n_clones": 0.0, "shannon": 0.0,
                "max_drivers": 0.0, "mean_drivers": 0.0,
                "mean_fitness": 1.0, "dominant_fraction": 0.0,
                "mean_passengers": 0.0}
    drivers = np.array([tree.clones[c].n_drivers for c in live])
    fits = np.array([tree.clones[c].fitness for c in live])
    passe = np.array([tree.clones[c].n_passengers for c in live])
    w = np.array([live[c] for c in live], dtype=float)
    return {
        "n_cells": float(total),
        "n_clones": float(len(live)),
        "shannon": shannon_diversity(live),
        "max_drivers": float(drivers.max()),
        "mean_drivers": float(np.average(drivers, weights=w)),
        "mean_fitness": float(np.average(fits, weights=w)),
        "mean_passengers": float(np.average(passe, weights=w)),
        "dominant_fraction": float(w.max() / total),
    }


# ---------------------------------------------------------------------------
# Croissance exponentielle asynchrone (référence pour le spectre 1/f)
# ---------------------------------------------------------------------------
def exponential_growth_process(params: EvolutionParams, n_target: int = 60_000,
                               seed: Optional[int] = None,
                               rng: Optional[np.random.Generator] = None,
                               planted_subclone_at: Optional[int] = None,
                               planted_advantage: float = 0.0
                               ) -> "BranchingResult":
    """Croissance exponentielle en temps continu (algorithme de Gillespie).

    Contrairement au processus à générations synchrones, les divisions sont
    ASYNCHRONES : à chaque évènement, une cellule tirée au hasard se divise
    (probabilité b/(b+d)) ou meurt. C'est le régime dans lequel Williams et al.
    (2016) établissent la loi de puissance M(f) ∝ 1/f, car les fréquences
    alléliques y varient continûment au lieu de progresser par paliers de
    génération.

    La sélection agit sur le TAUX NET de croissance en réduisant la mortalité :
    d_k = d_base * (1-s)^k. C'est la formulation de Bozic et al. ; augmenter la
    seule probabilité de division ne suffirait pas à produire une expansion
    subclonale détectable, car le taux net resterait dominé par la mortalité.

    `planted_subclone_at` permet d'introduire délibérément un sous-clone
    avantagé lorsque la population atteint une taille donnée : c'est le
    contrôle positif du test de neutralité (une expansion subclonale doit
    faire CHUTER le R² du fit 1/f sous le seuil de 0.98).

    Utilisé comme référence de validation du spectre de fréquences alléliques.
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    tree = CloneTree(params, rng, founder_drivers=1)
    cells: List[int] = [tree.root]          # identifiant de clone, par cellule
    p_any = tree.p_any_mutation
    d_base = params.d_base
    p_div = (1.0 - d_base)                   # probabilité de division (s petit)

    n_events = 0
    max_events = 80 * n_target
    planted = False
    while len(cells) < n_target and cells and n_events < max_events:
        n_events += 1

        if (planted_subclone_at is not None and not planted
                and len(cells) >= planted_subclone_at):
            j = int(rng.integers(len(cells)))
            new_cid = tree._build_child(cells[j], float(n_events), 1, 0)
            tree.clones[new_cid].n_drivers = max(tree.clones[new_cid].n_drivers, 2)
            cells[j] = new_cid
            planted = True

        i = int(rng.integers(len(cells)))
        cid = cells[i]
        k = tree.clones[cid].n_drivers
        if planted_subclone_at is not None:
            d_k = d_base * (1.0 - planted_advantage) if k >= 2 else d_base
        else:
            d_k = d_base * (1.0 - params.s_driver) ** k
        if rng.random() < (1.0 - d_k):
            # Division : deux filles, chacune peut muter
            daughters = []
            for _ in range(2):
                if rng.random() < p_any:
                    daughters.append(tree.mutate_conditional(cid, float(n_events)))
                else:
                    daughters.append(cid)
            cells[i] = daughters[0]
            cells.append(daughters[1])
        else:
            cells[i] = cells[-1]
            cells.pop()

    counts: Dict[int, int] = {}
    for cid in cells:
        counts[cid] = counts.get(cid, 0) + 1
    return BranchingResult(len(cells), tree, counts, len(cells) == 0,
                           n_events, [])


# ---------------------------------------------------------------------------
# Processus de branchement bien mélangé (référence analytique)
# ---------------------------------------------------------------------------
@dataclass
class BranchingResult:
    n_cells: int
    tree: CloneTree
    counts: Dict[int, int]
    extinct: bool
    generations: int
    history: List[dict] = field(default_factory=list)


def branching_process(params: EvolutionParams, n_generations: int = 40,
                      n_max: int = 200_000, seed: Optional[int] = None,
                      rng: Optional[np.random.Generator] = None,
                      record: bool = False) -> BranchingResult:
    """Processus de branchement de Bozic et al. (2010), population bien mélangée.

    À chaque génération, une cellule portant k drivers se divise avec la
    probabilité b_k = 1 - d_k et meurt avec d_k = (1-s)^k / 2. Ce modèle sert
    de RÉFÉRENCE ANALYTIQUE : il permet de vérifier que le moteur de mutation
    et de sélection reproduit les résultats publiés (probabilité d'extinction
    d'un driver, spectre 1/f en régime neutre).
    """
    rng = rng if rng is not None else np.random.default_rng(seed)
    tree = CloneTree(params, rng, founder_drivers=1)
    counts: Dict[int, int] = {tree.root: 1}
    hist: List[dict] = []

    for gen in range(n_generations):
        total = sum(counts.values())
        if total == 0:
            return BranchingResult(0, tree, counts, True, gen, hist)
        if total >= n_max:
            break

        new_counts: Dict[int, int] = {}
        for cid, n in list(counts.items()):
            if n <= 0:
                continue
            k = tree.clones[cid].n_drivers
            d_k = params.d_base * (1.0 - params.s_driver) ** k
            b_k = 1.0 - d_k
            # Nombre de cellules qui se divisent (le reste meurt)
            n_div = int(rng.binomial(n, b_k))
            if n_div == 0:
                continue
            # Chaque division donne 2 filles ; chacune peut muter
            p_any = tree.p_any_mutation
            for _ in range(2):
                # On tire en bloc le nombre de filles mutantes, puis on
                # échantillonne la loi conditionnelle exacte pour chacune.
                n_mut = int(rng.binomial(n_div, p_any))
                new_counts[cid] = new_counts.get(cid, 0) + (n_div - n_mut)
                for _ in range(n_mut):
                    child = tree.mutate_conditional(cid, float(gen))
                    new_counts[child] = new_counts.get(child, 0) + 1
        counts = {c: n for c, n in new_counts.items() if n > 0}
        if record:
            hist.append({"generation": gen, **clonal_summary(tree, counts)})

    total = sum(counts.values())
    return BranchingResult(total, tree, counts, total == 0,
                           n_generations, hist)
