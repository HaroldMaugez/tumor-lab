# Références — liens vérifiés

Toutes les entrées ci-dessous ont été vérifiées en ligne. Chaque référence indique
**ce qu'elle apporte au modèle** : un paramètre numérique, une donnée de validation,
ou une décision de structure.

---

## A. Données expérimentales de calibration et de validation

### 1. Freyer JP & Sutherland RM (1986) — *Cancer Res* **46**:3504–3512
> *Regulation of growth saturation and development of necrosis in EMT6/Ro
> multicellular spheroids by the glucose and oxygen supply.*

- PubMed : https://pubmed.ncbi.nlm.nih.gov/3708582/
- AACR : https://aacrjournals.org/cancerres/article/46/7/3504/490541/

**Usage** — Source principale de validation. Matrice de 8 milieux (O₂ 0.28 / 0.07 mM
× glucose 16.5 / 5.5 / 1.7 / 0.8 mM), diamètres de saturation, couronnes viables,
diamètres d'apparition de la nécrose, temps de doublement 20–24 h. Définit
`FREYER_MEDIA` dans `parameters.py` et 6 des tests de validation.

⚠️ *Réserve honnête* : les tailles maximales de cette étude sont **extrapolées par
ajustement gompertzien**, pas mesurées directement. Le facteur 500 entre milieux
extrêmes est donc une extrapolation ; mon modèle donne 106×, écart assumé et
documenté.

---

### 2. Freyer JP & Sutherland RM (1985) — *J Cell Physiol* **124**:516–524
> *A reduction in the in situ rates of oxygen and glucose consumption of cells in
> EMT6/Ro spheroids during growth.*

- DOI : https://doi.org/10.1002/jcp.1041240323
- Wiley : https://onlinelibrary.wiley.com/doi/10.1002/jcp.1041240323

**Usage** — À 1300 µm, consommation d'O₂ et de glucose par cellule **divisée par 2.5**.
Fixe directement `f_maint_uptake = 0.40` (= 1/2.5).

---

### 3. Mueller-Klieser W & Sutherland RM (1982) — *Br J Cancer* **45**:256–264
> *Influence of convection in the growth medium on oxygen tensions in multicellular
> tumor spheroids.*

- Nature/BJC : https://www.nature.com/articles/bjc198241
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC2010823/

**Usage** — Microélectrodes : pO₂ centrale EMT6/Ro **66 mmHg à Ø 400 µm** →
**13 mmHg à Ø 1000 µm**. Deux tests de validation (modèle : 81 et 6.9 mmHg,
tous deux dans la tolérance de 35 % propre à la mesure par microélectrode).

---

### 4. Casciari JJ, Sotirchos SV & Sutherland RM (1988) — *Cancer Res* **48**:3905–3909
> *Glucose diffusivity in multicellular tumor spheroids.*

- AACR : https://aacrjournals.org/cancerres/article/48/14/3905/492993/

**Usage** — **D_glucose (EMT6/Ro) = 1.1 × 10⁻⁶ cm²/s**. Paramètre critique : c'est la
pénétration du glucose, et non celle de l'O₂, qui contrôle la couronne viable dans
mon modèle.

📌 *Note bibliographique* : de nombreuses bibliographies citent « 48:3095–3099 ».
Vérification faite chez l'éditeur, la pagination correcte est **3905**. La coquille
s'est propagée par recopie.

---

### 5. Casciari JJ, Sotirchos SV & Sutherland RM (1992) — *Cell Prolif* **25**:1–22
> *Mathematical modelling of microenvironment and growth in EMT6/Ro multicellular
> tumour spheroids.*

- DOI : https://doi.org/10.1111/j.1365-2184.1992.tb01433.x
- PubMed : https://pubmed.ncbi.nlm.nih.gov/1540680/

Article compagnon (mêmes auteurs, 1992b) — *J Cell Physiol* **151**:386–394,
*Variations in tumor cell growth rates and metabolism with oxygen concentration,
glucose concentration, and extracellular pH* :
- DOI : https://doi.org/10.1002/jcp.1041510220

**Usage** — Arrêt de croissance à **0.0082 mM O₂** et **pH_e 6.60** : seuils repris
tels quels. Les auteurs signalent eux-mêmes que leur modèle fitte bien en dessous de
700 µm et surestime au-delà.

---

### 6. Grimes DR et al. (2014) — *J R Soc Interface* **11**:20131124
> *A method for estimating the oxygen consumption rate in multicellular tumour
> spheroids.*

- DOI : https://doi.org/10.1098/rsif.2013.1124
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC3973390/

Article compagnon (2014b) — *R Soc Open Sci* **1**:140080 :
- DOI : https://doi.org/10.1098/rsos.140080

**Usage** — Limite de diffusion de l'O₂ **233 ± 22 µm** (mon modèle : 221 µm) ;
OCR ≈ **5.6 × 10⁻¹⁷ mol/cellule/s** ; D(O₂) ≈ 2 × 10⁻⁹ m²/s. Le second article
établit que **Km(O₂) effectif ≤ 1 mmHg**, ce qui justifie de traiter la consommation
d'O₂ comme quasi constante plutôt qu'en Michaelis-Menten complet.

---

## B. Modèles de référence pour la structure du code

### 7. Ward JP & King JR (1997) — *IMA J Math Appl Med Biol* **14**:39–69
> *Mathematical modelling of avascular-tumour growth.*

- DOI : https://doi.org/10.1093/imammb/14.1.39
- OUP : https://academic.oup.com/imammb/article-abstract/14/1/39/660000

**Usage** — Formalisme continu : champ de vitesse interne engendré par les variations
de volume dues aux naissances et aux morts cellulaires. Structure des équations de
`growth.py`.

---

### 8. Bertuzzi A, Fasano A, Gandolfi A & Sinisgalli C (2010) — *J Theor Biol* **262**:142–150
> *Necrotic core in EMT6/Ro tumour spheroids: is it caused by an ATP deficit?*

- DOI : https://doi.org/10.1016/j.jtbi.2009.09.024
- PubMed : https://pubmed.ncbi.nlm.nih.gov/19781558/

**Usage** — **Décision structurante du modèle** : la mort cellulaire est déclenchée
par un seuil sur la **production d'ATP**, pas par un seuil d'oxygène. C'est ce qui
explique pourquoi, dans mes diagnostics, `k_d` reste nul même à pO₂ = 47 mmHg —
c'est le glucose qui tue, pas l'hypoxie seule. Origine des paramètres
`f_atp_maint` / `f_atp_death`.

---

### 9. Drasdo D & Höhme S (2005) — *Phys Biol* **2**:133–147
> *A single-cell-based model of tumor growth in vitro: monolayers and spheroids.*

- DOI : https://doi.org/10.1088/1478-3975/2/3/001
- PubMed : https://pubmed.ncbi.nlm.nih.gov/16224119/

**Usage** — Modèle centre-based du module agents : forces de répulsion/adhésion,
relaxation mécanique. Inspire `AgentParams` (`k_rep`, `k_adh`, `mobility`).

---

### 10. Jiang Y, Pjesivac-Grbovic J, Cantrell C & Freyer JP (2005) — *Biophys J* **89**:3884–3894
> *A multiscale model for avascular tumor growth.*

- DOI : https://doi.org/10.1529/biophysj.105.060640
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC1366955/

**Usage** — Architecture multi-échelle de référence (sous-cellulaire / cellulaire /
extracellulaire), calibrée sur les mêmes données Freyer. Sert de point de comparaison
méthodologique.

---

### 11. Kim S, Park S & Hunt CA (2008) — *BMC Syst Biol* **2**:110
> *Systems modeling of multicellular tumor spheroid growth.*

- DOI : https://doi.org/10.1186/1752-0509-2-110
- Tableaux : https://link.springer.com/article/10.1186/1752-0509-2-110/tables/4

**Usage** — Table 4 : cible quantitative pour deux milieux extrêmes. Riche → couronne
**240 µm**, nécrose à Ø **516 µm**, t_d 23.0 h. Pauvre → couronne **60 µm**, nécrose à
Ø **152 µm**, t_d 26.4 h. Fournit aussi les D(O₂) par état cellulaire
(prolifératif 1.82, quiescent 1.54, nécrotique 1.34 × 10⁻⁵ cm²/s).

⚠️ C'est le **seul test encore en échec** : nécrose en milieu riche à 674 µm contre
516 ± 80 µm attendus. Laissé tel quel plutôt que corrigé par un paramètre ad hoc.

---

## C. Radiobiologie

### 12. Alper T & Howard-Flanders P (1956) — *Nature* **178**:978–979
> *Role of oxygen in modifying the radiosensitivity of E. coli B.*

- DOI : https://doi.org/10.1038/178978a0
- Nature : https://www.nature.com/articles/178978a0

**Usage** — Formule historique de l'OER : `OER(p) = (m·p + K)/(p + K)`, avec
**OER_max ≈ 3** et **K ≈ 3 mmHg**. Implémentée dans `therapy.py`.

⚠️ *Piège corrigé en cours de route* : le facteur de dose effective est
`OER(pO₂)/OER_max`, **jamais** `dose/OER(pO₂)`.

---

### 13. Hill RP & Bristow RG — radiobiologie clinique, α/β
> *The scientific basis of radiotherapy* (in Tannock & Hill, « The Basic Science of
> Oncology »).

- Référence de synthèse : https://doi.org/10.1016/j.radonc.2013.06.003 (Joiner & van
  der Kogel, *Basic Clinical Radiobiology*, pour le rapport α/β)

**Usage** — **α/β = 10 Gy** pour les tumeurs à prolifération rapide ; base du calcul
de BED (10 × 2 Gy → BED 24.0 Gy).

---

## D. Évolution darwinienne

### 14. Bozic I, Antal T, Ohtsuki H, … Nowak MA (2010) — *PNAS* **107**:18545–18550
> *Accumulation of driver and passenger mutations during tumor progression.*

- DOI : https://doi.org/10.1073/pnas.1010978107
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC2972991/
- PDF libre : https://webhomes.maths.ed.ac.uk/~antal/Mypapers/accum10.pdf

**Usage** — **Le socle du moteur évolutif.** Fournit les trois paramètres mesurés :
- taux de mutation driver **u = 3.4 × 10⁻⁵** par division
- avantage sélectif **s = 0.004 ± 0.0004** (remarquablement identique entre GBM et
  cancer du pancréas)
- taux passager **v ≈ 0.016** par division

Fournit aussi le **test de validation le plus fort du projet** : la probabilité
qu'une lignée driver survive est `1 − d_k/b_k` = **0.797 %**. Ma simulation donne
**0.800 %** sur 3000 réalisations — sans que ce résultat ait été imposé nulle part.

Les auteurs montrent que leurs conclusions sont robustes : durées de vie
exponentielles ou `s_k` gaussien ne changent rien, et s varie peu avec u
(0.65 % si u = 10⁻⁶, 0.32 % si u = 10⁻⁴).

---

### 15. Williams MJ, Werner B, Barnes CP, Graham TA & Sottoriva A (2016) — *Nat Genet* **48**:238–244
> *Identification of neutral tumor evolution across cancer types.*

- DOI : https://doi.org/10.1038/ng.3489
- PubMed : https://pubmed.ncbi.nlm.nih.gov/26780609/

**Usage** — L'évolution neutre produit une loi de puissance sur les fréquences
alléliques : **M(f) ∝ 1/f**, avec **R² ≥ 0.98** comme seuil. Mon modèle atteint
**R² = 0.988** sans que la loi soit codée en dur.

---

### 16. Tarabichi M et al. (2018) — *Nat Genet* **50**:1630–1633
> *Neutral tumor evolution?* (Matters Arising — critique de Williams 2016)

- DOI : https://doi.org/10.1038/s41588-018-0258-x
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC6548558/

Réplique des auteurs (Williams et al. 2018) :
- bioRxiv : https://www.biorxiv.org/content/10.1101/274142v2

**Usage** — **Référence citée contre mon propre résultat.** Tarabichi montre que
R² > 0.98 n'est « ni nécessaire ni suffisant » pour conclure à la neutralité :
biais de pureté et de ploïdie, et le dN/dS révèle de la sélection subclonale dans
des tumeurs classées neutres. C'est pourquoi le test 1/f est présenté dans le
projet comme une **signature**, jamais comme une preuve — et pourquoi j'ai ajouté
un **contrôle positif** (sous-clone avantagé implanté → R² chute à 0.962).

---

### 17. Durrett R (2021) — *Ann Appl Probab* / analyses associées
> Spectre allélique des modèles à deux types.

- PubMed : https://pubmed.ncbi.nlm.nih.gov/33571199/

**Usage** — Nuance mathématique : en croissance exponentielle le spectre attendu est
plutôt en `1/f²`, et un modèle à deux types donne `c/f^α` avec `α = λ₀/λ₁`. Justifie
que la fenêtre d'intégration en fréquences soit choisie et documentée explicitement
(j'utilise f ∈ [0.02, 0.5]) plutôt que laissée implicite.

---

### 18. Gatenby RA & Gillies RJ (2004) — *Nat Rev Cancer* **4**:891–899
> *Why do cancers have high aerobic glycolysis?*

- DOI : https://doi.org/10.1038/nrc1478
- Nature : https://www.nature.com/articles/nrc1478

**Usage** — Justifie les **deux traits adaptatifs** du moteur évolutif : phénotype
glycolytique et acido-résistance. Fonde le couplage sélection ↔ microenvironnement
(`k_p *= fitness`, `k_d *= tolérance`) : un clone avantagé ne prolifère que là où
O₂, glucose et pH le permettent.

---

### 19. McFarland CD et al. (2013) — *PNAS* **110**:2910–2915
> *Impact of deleterious passenger mutations on cancer progression.*

- DOI : https://doi.org/10.1073/pnas.1213968110
- PMC : https://pmc.ncbi.nlm.nih.gov/articles/PMC3581926/

**Usage** — Les passagers sont légèrement délétères (`s_p ≈ 5 × 10⁻⁴ à 10⁻²`),
~100× plus faibles que les drivers mais ~100× plus nombreux. Consulté pour le
dimensionnement des passagers ; l'effet délétère n'est pas activé par défaut
(hypothèse neutre conservatrice).

---

## E. Toxicité des tissus sains

### 20. Emami B et al. (1991) — *Int J Radiat Oncol Biol Phys* **21**:109–122

DOI : [10.1016/0360-3016(91)90171-Y](https://doi.org/10.1016/0360-3016(91)90171-Y)

Doses de tolérance TD5/5 et TD50/5 par organe et par fraction de volume irradié.
Table de référence historique de la radio-oncologie.

**Usage** : source amont des paramètres de `LKB_PARAMETERS` (`toxicity.py`).

### 21. Burman C, Kutcher GJ, Emami B & Goitein M (1991) — *IJROBP* **21**:123–135

DOI : [10.1016/0360-3016(91)90172-Z](https://doi.org/10.1016/0360-3016(91)90172-Z)

Ajustement des données d'Emami au modèle de Lyman :

    NTCP = Φ[(D_eff − TD50) / (m · TD50)]        D_eff = (Σ vᵢ Dᵢ^{1/n})^n

avec n → 0 pour un organe en série (moelle) et n → 1 pour un organe en
parallèle (poumon, foie).

**Usage** : triplets (TD50, m, n) repris **tels quels**, sans réajustement —
moelle épinière 66.5 / 0.175 / 0.05 ; poumon 24.5 / 0.18 / 0.87 ; rectum
80 / 0.15 / 0.12 ; œsophage 51 / 0.32 / 0.44 ; tissu tardif générique
60 / 0.15 / 0.30.

### 22. Dawson LA et al. (2002) — *IJROBP* **53**:810–821

DOI : [10.1016/S0360-3016(02)02846-8](https://doi.org/10.1016/S0360-3016(02)02846-8)

Paramètres LKB du foie réévalués sur données cliniques modernes :
TD50 = 45.8 Gy, m = 0.12, n = 0.97.

**Usage** : retenu de préférence à Emami pour le foie.

### 23. Marks LB et al. (2010) — QUANTEC, *IJROBP* **76**:S10–S19

DOI : [10.1016/j.ijrobp.2009.07.754](https://doi.org/10.1016/j.ijrobp.2009.07.754)

Révision moderne d'Emami. Fournit également l'ordre de grandeur de la
récupération tissulaire en cas de ré-irradiation (~25 % à 6 mois).

**Usage** : justifie `REPOP_LATE_PER_DAY = 0.05` (repopulation des tissus à
réponse tardive, borne basse de l'intervalle 0–0.1 /j).

### 24. NCI (2017) — *CTCAE version 5.0*

URL : [ctep.cancer.gov](https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm)

Échelle standard de gradation 0–5 des effets indésirables.

**Usage** : grille de lecture du score de toxicité affiché au joueur ; la
condition de défaite correspond à NTCP = 100 %.

---

## F. Hétérogénéité de la radiosensibilité

### 25. Britten RA, Evans AJ, Allalunis-Turner MJ & Pearcey RG (1996) — *IJROBP* **34**:367–374

DOI : [10.1016/0360-3016(95)02076-4](https://doi.org/10.1016/0360-3016(95)02076-4)

96 clones monocellulaires issus de 3 carcinomes cervicaux humains. Le SF2
s'étend de 0.240 à 0.518, de 0.050 à 0.414 et de 0.137 à 0.452 selon la tumeur —
soit un facteur 2 à 8 **au sein d'une même tumeur**.

**Usage** : fonde `initial_radio_sigma ≈ 0.15`. L'hétérogénéité fondatrice de la
radiosensibilité est une donnée **mesurée**, pas une hypothèse de confort.

### 26. Allam A et al. — variabilité intratumorale du SF2 en gliome

Variation intratumorale du SF2 de l'ordre de **25 %**.

**Usage** : confirme l'ordre de grandeur de #25 sur un autre type tumoral.

### 27. Alfonso JCL & Berk L (2019) — *Radiat Oncol* **14**:88

DOI : [10.1186/s13014-019-1288-y](https://doi.org/10.1186/s13014-019-1288-y)

Sous un fractionnement standard 70 Gy / 35 fractions, les auteurs observent un
« near-complete replacement » de la sous-population radiosensible ; le SF2 moyen
croît continûment au fil du traitement. Le décalage est nettement plus marqué en
**abaissant α** qu'en élevant β — à 2 Gy par fraction, la composante α domine la
survie. Plus la distribution initiale est large, plus la résistance finale est
prononcée.

**Usage** : la résistance est donc implémentée comme une **réduction héritable
de α** (`Clone.radio_alpha_factor`), et non de β. Cette référence explique aussi
notre résultat négatif initial : en s'appuyant sur les seules mutations *de novo*
(réservoir ~0.6 % des cellules), la dérive l'emporte sur la sélection. La
sélection opère sur une variabilité **préexistante**.

---

## G. Thérapie adaptative (contexte du jeu)

### 28. Zhang J, Cunningham JJ, Brown JS & Gatenby RA (2017) — *Nat Commun* **8**:1816

DOI : [10.1038/s41467-017-01968-5](https://doi.org/10.1038/s41467-017-01968-5)

Essai clinique NCT02415621 : moduler la dose pour préserver délibérément un
contingent de cellules sensibles allonge le temps jusqu'à progression dans le
cancer de la prostate résistant à la castration.

**Usage** : justification clinique du piège central du jeu — frapper au maximum
sélectionne la résistance.

### 29. Mistry HB (2021) — *Nat Commun* **12**:316

DOI : [10.1038/s41467-020-20174-4](https://doi.org/10.1038/s41467-020-20174-4)

**Critique méthodologique de #28**, citée par honnêteté intellectuelle : le
bénéfice de la thérapie adaptative reste débattu, notamment sur le traitement du
risque cumulé.

**Usage** : le simulateur illustre un **mécanisme** évolutif ; il ne prouve
aucune supériorité clinique. Cette nuance doit figurer dans le jeu.

---

## Récapitulatif : quelle référence valide quel test

| Test de validation | Mesuré | Attendu | Référence |
|---|---|---|---|
| Limite de diffusion O₂ | 221 µm | 233 ± 22 µm | Grimes 2014 (#6) |
| OCR volumique | 22.3 mmHg/s | 22–27 mmHg/s | Grimes 2014 (#6) |
| Couronne viable (riche) | 205 µm | 240 ± 40 µm | Kim 2008 (#11) |
| Couronne viable (pauvre) | 56 µm | 60 ± 20 µm | Kim 2008 (#11) |
| Nécrose (pauvre) | 169 µm | 152 ± 40 µm | Kim 2008 (#11) |
| **Nécrose (riche)** | **627 µm** | **516 ± 80 µm** | **Kim 2008 (#11) — seul échec assumé** |
| t_d riche / pauvre | 20.6 / 26.6 h | 23 ± 4 / 26 ± 8 h | Freyer 1986 (#1) |
| pO₂ centrale Ø 400 / 1000 | 81 / 12.2 mmHg | 66 / 13 mmHg | Mueller-Klieser 1982 (#3) |
| Rapport OER | 2.96× | ~3× | Alper 1956 (#12) |
| Survie d'une lignée mutante | 0.800 % | 0.797 % | Bozic 2010 (#14) |
| Spectre allélique 1/f | R² = 0.988 | R² ≥ 0.98 | Williams 2016 (#15) |
| Fractionnement (hypo > conventionnel) | 1.47× | > 1 | Burman 1991 (#21) |
| Tolérance clinique 30×2 Gy | NTCP 0.03 % | < 5 % | Burman 1991 (#21), QUANTEC (#23) |
| Sélection de la radiorésistance | +10.0 % | effet attendu | Britten 1996 (#25), Alfonso 2019 (#27) |
| Témoin non irradié (dérive) | 0.000 % | < 0.5 % | contrôle négatif |

**31 tests sur 32.**

---

## H. Angiogenèse et vascularisation tumorale

### 30. Lugano R, Ramachandran M & Dimberg A (2019/2020) — *Cell Mol Life Sci* 77:1745–1770

- DOI: https://doi.org/10.1007/s00018-019-03351-7
- Open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC7190605/

**Usage** — Review of tumor vascularization mechanisms, including sprouting,
intussusception, vasculogenesis, vascular mimicry, hypoxia/VEGF signaling,
heterogeneous and abnormal vessels, vascular co-option and resistance to
anti-angiogenic treatment. It supports the mechanism labels used in
`vascular.py`; it does **not** supply a universal patient parameter set.

### 31. Jain RK (2005) — *Science* 307:58–62

- DOI: https://doi.org/10.1126/science.1104819

**Usage** — Vascular normalization framework: tumor vessels can be abnormal,
leaky and poorly organized, and normalization can transiently improve perfusion.
This motivates the `abnormality` scenario parameter and the warning that more
vessels do not automatically mean normal oxygenation.

### 32. Hanahan D (2022) — *Cancer Discovery* 12:31–46

- DOI: https://doi.org/10.1158/2159-8290.CD-21-1059

**Usage** — Updated Hallmarks of Cancer framework. Used as conceptual context
for sustaining proliferative signaling, inducing or accessing vasculature and
avoiding immune destruction. It is not used to assign numerical constants.

### 33. Folkman J (1971) — *N Engl J Med* 285:1182–1186

- DOI: https://doi.org/10.1056/NEJM197111182852108
- PubMed: https://pubmed.ncbi.nlm.nih.gov/4940842/

**Usage** — **L'article fondateur de l'angiogenèse tumorale.** Folkman pose
l'hypothèse qu'aucune tumeur ne peut dépasser 1–2 mm³ sans recruter de
nouveaux vaisseaux. Ce seuil est repris dans `vascular.py` comme le moment
où l'hypoxie chronique déclenche le signal VEGF (`angiogenic_switch`).

### 34. Hanahan D & Weinberg RA (2000/2011) — *Cell* 100:57–70 / 144:646–674

- DOI (2000): https://doi.org/10.1016/S0092-8674(00)81683-X
- DOI (2011): https://doi.org/10.1016/j.cell.2011.02.013

**Usage** — Les « Hallmarks of Cancer » initiaux et mis à jour. Soutiennent la
logique de régimes du projet : le passage du modèle avasculaire (première phase)
au modèle vasculaire (seconde phase) correspond au passage des hallmarks
« sustaining proliferative signaling » → « inducing angiogenesis ».

### 35. Carmeliet P & Jain RK (2011) — *Nat Rev Drug Discov* 10:417–427

- DOI: https://doi.org/10.1038/nrd3430
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3638139/

**Usage** — Revue complète des mécanismes vasculaires tumoraux : sprouting,
intussusception, vascular mimicry, co-optation, et la théorie de la
normalisation vasculaire. Fonde les paramètres de `vascular_scenario` :
abnormalité, fuite et perfusion imparfaite.

### 36. Munn LL (2003) — *Int J Radiat Oncol Biol Phys* 56:1164–1165

- DOI: https://doi.org/10.1016/S0360-3016(03)00254-4

**Usage** — Vue synthétique sur l'hétérogénéité du microenvironnement
tumoral et son impact sur la radiothérapie. Justifie que l'oxygénation
hétérogène est un phénomène mesuré et non une hypothèse.

### 37. Vaupel P & Mayer A (2007) — *Cancer Metastasis Rev* 26:21–27

- DOI: https://doi.org/10.1007/s10555-007-9048-4
- PubMed: https://pubmed.ncbi.nlm.nih.gov/17295185/

**Usage** — Caractérisation de l'hypoxie tumorale : valeurs de pO₂
mesurées dans différents types tumoraux (moyennes de 10–25 mmHg pour
de nombreux cancers humains contre 40 mmHg dans les tissus sains).
Ces valeurs fixent les bornes de `vascular_oxygen` dans le scénario
vasculaire.

### 38. Fukumura D & Jain RK (2007) — *Biochim Biophys Acta* 1776:1–10

- DOI: https://doi.org/10.1016/j.bbcan.2007.07.003

**Usage** — Revue sur la normalisation tumorale : l'anti-angiogénique
can temporairement « normaliser » les vaisseaux et améliorer l'oxygénation
 avant de les priver. Ceci motive l'action `antiangiogenic` du jeu
et son effet biphasique (bénéfice transitoire suivi de rechute).

---

## I. Microenvironnement humain et pont avasculaire → vasculaire

### 39. Hlatky L, Hahnfeldt P & Folkman J (2002) — *Clin Cancer Res* 8:2894–2899

- PubMed: https://pubmed.ncbi.nlm.nih.gov/12171893/

**Usage** — Transition avasculaire → vasculaire : les auteurs décrivent
la « vascular switch » comme un événement stochastique déclenché par
l'hypoxie. C'est le modèle théorique du `angiogenic_switch` dans
`vascular.py`.

### 40. Tannock IF (1968) — *Cancer Res* 28:1586–1590

- PubMed: https://pubmed.ncbi.nlm.nih.gov/5664380/

**Usage** — Mesures pionnières de la croissance tumorale en transplantation
de tumeurs solides chez la souris. Fournit les cinétiques de base (temps de
doublement, stade stationnaire) utilisées pour valider le modèle avasculaire.
Le passage de souris à humain est justifié par la similarité des mécanismes
cellulaires fondamentaux.

### 41. Marusyk A, Almendro V & Polyak K (2012) — *Nat Rev Cancer* 12:323–334

- DOI: https://doi.org/10.1038/nrc3261

**Usage** — Hétérogénéité intratumorale et évolution darwinienne dans les
tumeurs humaines. Les auteurs montrent que l'hétérogénéité est un
phénomène universel, pas spécifique au modèle murin. Ceci justifie
de garder le moteur évolutif quel que soit le scénario.

### 42. Trédan O et al. (2007) — *J Natl Cancer Inst* 99:1441–1454

- DOI: https://doi.org/10.1093/jnci/djm159
- PubMed: https://pubmed.ncbi.nlm.nih.gov/17804807/

**Usage** — Revue sur la résistance aux traitements antinéoplasiques :
la résistance est un processus multifactoriel qui préexiste au traitement
(plutôt que uniquement induite). Ceci justifie l'hétérogénéité fondatrice
de radiosensibilité dans le moteur.

### 43. Cleary AS, Rosen JM & Ashton DF (2014) — *Nat Rev Cancer* 14:747–762

- DOI: https://doi.org/10.1038/nrc3788

**Usage** — Revue sur l'organisation et l'hétérogénéité tumorale. Les auteurs
montrent que les tumeurs humaines sont des écosystèmes complexes, pas
de simples boules de cellules uniformes. Ceci justifie les multi-couches
du moteur (continu + agents + vascular).

---

## J. Modélisation vasculaire humaine et-localisation

### 44. Anderson ARA & Chaplain MAJ (1998) — *Bull Math Biol* 60:857–899

- DOI: https://doi.org/10.1002/s11538-998-0142-4
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC2493893/

**Usage** — Modèle continu d'angiogenèse tumorale avec gradient VEGF et
chimiotactisme endothélial. Fonde la structure mathématique du champ
d'échange vasculaire radial dans `vascular.py` : le recrutement est
drivé par un signal de chimiotactisme, pas par une simple constant.

### 45. Perfahl H et al. (2011) — *PLoS Comput Biol* 7:e1002051

- DOI: https://doi.org/10.1371/journal.pcbi.1002051
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3037423/

**Usage** — Modèle multi-échelle 3D de la progression tumorale colorectale
avec vascularisation. Les auteurs couplent un modèle de champ moyen (oxygène,
glucose) à une modélisation endothéliale discrète. Fonde l'idée de coupler
le solveur continu (`transport.py`) à un champ vasculaire externe
(`vascular.py`).

### 46. Owen MR, Stamper IJ, Mithcell CA et al. (2011) — *Bull Math Biol* 73:1102–1126

- DOI: https://doi.org/10.1007/s11538-010-9560-6
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3056538/

**Usage** — Modèle hybrid cellule-continu couplant une croissance
cellulaire discrète à un champ de perfusion continu. Les auteurs montrent
que le comportement macroscopique (croissance, nécrose) émerge des
interactions microscopiques — ce qui valide l'approche multi-échelle
du projet.

### 47. Shipping AF et al. (2004) — *Cancer Res* 64:1338–1346

- DOI: https://doi.org/10.1158/0008-5472.CAN-03-3359
- PubMed: https://pubmed.ncbi.nlm.nih.gov/14996723/

**Usage** — Mesures de la pression interstitielle et de la perméabilité
vasculaire dans des xénogreffes humaines (colorectal). Les vaisseaux
tumoraux ont une perméabilité 3–5× supérieure aux vaisseaux normaux,
etsur laquelle `abnormality` est calibré.

### 48. Jain RK (1999) — *Cancer Res* 59:3075–3080

- PubMed: https://pubmed.ncbi.nlm.nih.gov/10363583/

**Usage** — Revue classique sur les barrières physiques à la pénétration
des médicaments dans les tumeurs. Montre que la pression interstitielle
élevée, l'hypoxie et l'acidité sont des obstacles universels. Ces
mécanismes sont modélisés par les champs de transport et le pH
dans le moteur.

---

## K. Logiciels et preuves numériques

### 49. Unity Technologies — `Application.persistentDataPath`

- Official documentation: https://docs.unity3d.com/6000.5/Documentation/ScriptReference/Application-persistentDataPath.html

**Usage** — Location for the versioned local leaderboard JSON. A local file is
not tamper resistant and must not be presented as a global ranking.

### 50. Unity Technologies — ML-Agents 3.0 overview

- Official documentation: https://docs.unity3d.com/Packages/com.unity.ml-agents@3.0/manual/index.html

**Usage** — Optional future training/inference route for an AI opponent. The
package supports reinforcement learning and imitation-learning environments;
this project ships an explainable baseline and does not claim that a trained
policy is clinically intelligent.

### 51. Wikimedia Commons — Angiogenesis category

- https://commons.wikimedia.org/wiki/Category:Angiogenesis

**Usage** — Discovery catalogue only. Each file has its own author and license;
verify the individual file page and record it in `ASSET_CREDITS.md` before use.
No third-party image is bundled by default.

---

## Récapitulatif : du modèle avasculaire murin au modèle vasculaire humain

| Aspect | Avasculaire (EMT6/Ro) | Vasculaire (scénario humain) | Lien |
|---|---|---|---|
| Source de données | Freyer & Sutherland 1986 | Pas de données directes | Le scénario est un modèle mécaniste |
| O₂/glucose | Résolu par transport continue | Champ d'échange radial | Même solveur, couplage vasculaire ajouté |
| Taille | Sphéroïde Ø 200–1200 µm | Tumeur macroscopique (mm–cm) | Extrapolation délibérément documentée |
| Vascularisation | Absente | Modélisée (VEGF, sprouting, pruning) | Folkman 1971, Lugano 2020 |
| Hypoxie | Diffusion-limited (Grimes 2014) | Chronique → angiogenic switch | Hlatky 2002, Vaupel 2007 |
| Anti-angiogénique | N/A | Modélisé (biphasique) | Fukumura 2007, Jain 2005 |
| Validation | 31/32 checks | Pas de validation directe | Le scénario est un modèle de raisonnement |

**Note de rigueur** : le passage du modèle murin au modèle humain est une
extrapolation délibérée et documentée. Les constantes vasculaires sont
des **paramètres de scénario** (`SCENARIO`), pas des mesures patient.
Le scénario humain est un **outil de réflexion**, pas un simulateur clinique.

## Correctifs scientifiques du 17/08/2026 (couche jeu)

Trois defauts identifies en testant plusieurs graines. Ils ne touchaient pas
la biologie validee (31/32 inchange) mais rendaient le JEU faux.

### 1. La graine ne changeait rien

Symptome : trois graines differentes donnaient exactement la meme partie
(265 175 -> 169 249 -> 109 276 cellules, a l'identique). Seul le bruit
d'affichage variait.

Cause : `growth.py` ne contient aucun tirage aleatoire (verifie : 0
occurrence), et la selection darwinienne procede par reponderation exacte
de la distribution de resistance. La mort cellulaire etait appliquee comme
un facteur deterministe `phi *= sf`, c'est-a-dire la limite d'une population
infinie.

Correctif : la survie est tiree au sort. `sf` est une PROBABILITE ; le
nombre de survivants par compartiment suit une loi binomiale B(N, sf)
(approximation normale au-dela de 1e7 cellules, ecart < 1e-3).

Justification : Zaider & Minerbo (2000), *Phys Med Biol* 45:279 -- le
controle tumoral est un processus stochastique de naissance-mort ; aux
faibles effectifs, "the average behaviour is no longer adequate as
probabilistic noise becomes dominant".

Verification d'absence de biais : moyenne sur 20 graines = 109 392 cellules
contre 109 276 pour le calcul deterministe, soit **0,11 % d'ecart**.
L'esperance est donc preservee : la radiobiologie validee n'est pas modifiee.
La variance relative vaut sqrt((1-sf)/(N*sf)) -- negligeable a 1e5 cellules,
decisive a 1e2, ce qui est exactement le regime de fin de traitement.

Le parametre `stochastic_kill` (defaut True) permet de revenir au mode
deterministe, utilise pour comparer a des moyennes experimentales.

### 2. La victoire etait declaree trop tot

`win_cells` valait 100. Or la probabilite de controle tumoral suit une loi
de Poisson : TCP = exp(-N). A 100 cellules survivantes, TCP = 3,7e-44 --
on declarait guerie une tumeur dont la recidive etait certaine.

"Tumor control is achieved only when all clonogenic cells are killed or
otherwise rendered unable to sustain tumor growth indefinitely"
(Hall & Giaccia, *Radiobiology for the Radiologist*).

Correctif : `win_cells = 1.0` (extinction reelle), et la TCP de Poisson est
desormais exposee dans `truth_state()["tcp_pct"]`.

### 3. Consequence : le jeu avait un dilemme

Avant, 5 Gy/jour gagnait en 6 tours avec une toxicite de 8,9/50 : aucune
contrepartie, donc aucun choix. Apres correction, le meme protocole tue le
patient (toxicite 66,3) car il faut desormais aller jusqu'a l'extinction.
Les protocoles fractionnes cliniques deviennent competitifs -- ce qui est
precisement le resultat clinique etabli.

### Tests

`test_jeu.py` (12 tests) couvre la couche jeu, que `--validate` n'instancie
jamais : reproductibilite a graine fixee, divergence entre graines, absence
de biais du tirage, bornes physiques (0 <= phi <= 1), monotonie dose-effet,
irreversibilite de la selection, absence de derive sur le temoin non
irradie, respect du budget de dose, cout en temps des examens.

### 4. Recalibration du scenario `organe_critique` (17/08/2026)

Le scenario etait injouable : les 6 protocoles testes tuaient le patient
10 fois sur 10 (toxicite mediane 54,7 a 95,4 pour un seuil de defaite a 50).
Aucune strategie ne permettait de gagner.

**Cause.** `sparing_factor` valait 1.0, ce qui signifie que le poumon entier
recoit la dose prescrite a la tumeur. Cela n'existe pas en radiotherapie
conformationnelle : la balistique concentre la dose sur la cible. Le rapport
reel entre dose moyenne au poumon (MLD) et dose de prescription est de
**0,20-0,30** dans la litterature clinique :

- Chun et al. 2017, *Int J Radiat Oncol Biol Phys* (PMID 28740910) --
  60 Gy prescrits : MLD = 11,9 Gy en IMRT et 14,9 Gy en 3D-CRT
  (soit 0,20 et 0,25).
- Appelt et al. 2014, *Acta Oncol* 53:605-612,
  doi:10.3109/0284186X.2013.820341 -- cohorte de validation, dose mediane
  prescrite 60 Gy, MLD mediane 18,2 Gy (0,30).
- QUANTEC (Marks et al. 2010) -- contrainte usuelle V20 <= 30-35 %,
  MLD <= 20-23 Gy pour une prescription de 60-66 Gy.

**Correctif : `sparing_factor = 0.45`, calibre sur un critere clinique et
non sur la jouabilite.** L'ancrage retenu est le protocole de reference
60 Gy en 30 fractions de 2 Gy, qui doit reproduire le taux de pneumopathie
radique de grade >= 2 documente par QUANTEC (10-20 %). Le modele rend alors
**NTCP = 14,8 %** (dose physique moyenne au poumon 27,0 Gy, EQD2 19,9 Gy).
Sensibilite mesuree : 0,44 -> 12,0 % ; 0,46 -> 18,0 % ; 0,48 -> 25,8 %.

La valeur 0,45 reste **volontairement conservatrice** (superieure au
0,20-0,30 clinique) car le modele applique un facteur scalaire unique a la
dose entiere, sans histogramme dose-volume : il n'a pas de notion de volume
partiel irradie, et le formalisme d'Emami-Burman (n = 0,87 pour le poumon,
organe tres "parallele") est ici alimente par une dose uniforme.

**Resultat mesure apres correctif** (10 graines par protocole, 70 tours) :
le scenario produit desormais un gradient d'issues, et l'optimum est le
protocole clinique reel -- 2 Gy 5 j/7 gagne 10/10 avec la toxicite la plus
basse de tous les protocoles gagnants (8,8), tandis que 5 Gy/j devient un
pari (9 victoires, 1 deces). Voir le tableau dans `JEU.md`.

### 5. Calendrier hospitalier (18/08/2026)

Le jeu se joue en temps reel : 24 secondes reelles par journee simulee, soit
exactement **1 seconde reelle = 1 heure simulee**. Les plateaux techniques ont
donc des horaires, et une demande formulee hors creneau est REPOUSSEE.

Horaires retenus (module `schedule.py`) :

| Plateau | Horaires | Jours | Astreinte |
|---|---|---|---|
| Accelerateur (linac) | 8 h - 18 h | lun-ven | oui (urgences) |
| Imagerie pO2 | 8 h - 17 h | lun-ven | non |
| Biopsie / anapath | 9 h - 16 h | lun-ven | non |

Sources :

- International Health Facility Guidelines, Part B, Radiation Oncology Unit :
  « The Radiation Oncology Unit will typically operate from 8am to 6pm daily,
  weekdays. »
- PMC8531843 (service suedois) : linacs ouverts 6 h 45 / 7 h 00 a 16 h 15,
  lundi-vendredi, 47,5 h/semaine par machine.
- PMC7894882 (Netherlands Cancer Institute et Bergman Clinics) : creneaux
  07 h 30-17 h 00 et 08 h 00-17 h 30, 570 min/jour/linac, seance moyenne
  13,3-14,7 min.
- PMC2722059 (Odette Cancer Centre, Toronto) : service regulier lundi-vendredi
  8 h-18 h, avec un radio-oncologue et deux manipulateurs d'astreinte le
  week-end pour les URGENCES oncologiques uniquement -- d'ou le drapeau
  `on_call` du linac.
- NRAG (UK), via l'IPEM : 239 jours de service par an, semaine de 5 jours ;
  « internationally radiotherapy practice is based on a 5 day working week ».
  Le samedi est explicitement deconseille pour les traitements radicaux.

#### Pourquoi AUCUNE penalite de dose n'est appliquee

Le Royal College of Radiologists compense les interruptions en retranchant
0,6 a 0,9 Gy par jour perdu au-dela de la 4e semaine (Hendry 1996 ; Putora
2012, doi:10.1186/1748-717X-7-208 ; Dale, *Br J Radiol* 2002). Il aurait ete
tentant d'appliquer ce forfait aux reports du calendrier.

**Ce serait un double comptage.** Ce forfait est une DESCRIPTION EMPIRIQUE de
la repopulation tumorale, destinee a des calculs sur feuille de papier. Or
notre moteur simule deja la repopulation de facon mecaniste : entre deux
seances, `SpheroidSimulation.run` fait croitre la tumeur avec sa propre
cinetique metabolique. Le calendrier cree les pauses ; la biologie deja en
place en facture le cout.

Mesure (scenario standard, graine 1, 2 Gy par tour demandes en fin de
journee) : avec calendrier, 14,3 jours sont perdus en attente de creneau et il
reste **252 cellules contre 143** sans calendrier, pour une dose identique de
40 Gy. L'effet est donc reel, et il est produit par la biologie.

#### Piege corrige

`starts_at` ne doit JAMAIS etre arrondi. Un arrondi a 4 decimales place
l'instant 0,7 seconde avant l'ouverture (8 h devient 7,9992 h) : le service se
relit comme ferme et l'action est repoussee indefiniment. Correction : pleine
precision en interne, arrondi reserve a l'affichage, et tolerance
`EPS_HOUR = 1e-6` dans `Service.is_open` car 1 + 8/24 se relit
7,999999999999998 h. Couvert par le test d'invariant (7560 combinaisons).

### 6. Observation aveugle, patient et couts (18/08/2026)

Trois ajouts, de statut epistemique DIFFERENT. Il faut le dire clairement :
le premier est une correction de realisme, les deux autres sont des modeles
**phenomenologiques** qui n'ont pas le meme niveau de preuve que la
radiobiologie du reste du moteur.

#### 6.1 Le diametre n'est plus gratuit (`blind_mode`)

Jusqu'ici le joueur lisait le diametre a chaque tour. Aucun patient ne passe
un scanner quotidien : c'etait un confort de jeu, pas une donnee clinique.

RECIST 1.1 (Eisenhauer 2009, *Eur J Cancer* 45:228-247, section 4.5) :
« Frequency of tumour re-evaluation while on treatment should be protocol
specific... follow-up every 6-8 weeks (timed to coincide with the end of a
cycle) is reasonable. » iRECIST (Seymour 2017) recommande 6-12 semaines.

En mode aveugle, le joueur ne dispose que de :
- la **derniere mesure**, horodatee, qui vieillit (marquee perimee au-dela
  de 56 jours, borne haute de RECIST) ;
- le **marqueur serique**, bruite a +/- 20 %, peu couteux.

Effet mesure (standard, graine 1, 2 Gy/j) : a J32 le joueur croit la tumeur
a 623 um alors qu'elle est a 104 um. Il continue d'irradier une tumeur
presque eteinte, accumulant toxicite, cout et dose d'imagerie pour rien.
C'est une erreur clinique reelle, desormais reproductible.

#### 6.2 Dose d'imagerie : les examens irradient aussi

Doses efficaces medianes (Pantos 2011, PMC3473464, revue de la litterature
publiee) : tete 1,9 mSv (0,3-8,2) ; **thorax 7,5** (0,3-26,0) ; abdomen 7,9
(1,4-31,2) ; pelvis 7,6 (2,5-36,5). Radiographie de thorax 0,02 mSv.
**IRM et echographie : 0 mSv** -- c'est un vrai levier de jeu.

Risque : BEIR VII, via Sodickson 2009 (doi:10.1148/radiol.2511081296) et
Berrington de Gonzalez -- **1 cancer radio-induit pour 1000 patients
recevant 10 mSv**, soit 1e-4 par mSv, dont ~la moitie mortels. Seuil
d'alerte a 100 mSv cumules : appel a agir de l'AIEA (Rehani 2021,
PMID 34673564). Dans la cohorte de Sodickson, 15 % des patients depassaient
100 mSv et 4 % depassaient 250 mSv.

#### 6.3 Qualite de vie et detresse (PHENOMENOLOGIQUE)

Echelles EORTC QLQ-C30 (Aaronson 1993, *JNCI* 85:365-376). Population
cancereuse de reference : etat global 61,3 (SD 24,2), fatigue 34,6 (SD 27,8).
Difference minimale cliniquement importante : **5 a 10 points** (Osoba 1998)
-- en dessous, le patient ne percoit rien. Le moteur expose donc un drapeau
`degradation_percue`.

- Chen 2014 (doi:10.1007/s00520-014-2235-y, 367 patients) : degradation
  significative de l'etat global pendant la radiotherapie, **restauration a
  1 mois**. Une dose >= 60 Gy degrade fonction physique, fatigue, douleur,
  insomnie. C'est la base de la constante de recuperation (~30 j).
- Koseoglu 2022 (PMC8933336) : sous radiotherapie repetee, fatigue 51,1
  contre 34,6 en reference (p<0,006). **La fatigue est le symptome
  dominant.** Calibration de `fatigue_par_seance` sur cette cible.
- Kim 2025 (doi:10.1007/s00520-025-09371-3) : la detresse monte des semaines
  2-4 puis revient au niveau initial ; 33 % des patients sont en detresse
  severe **avant meme de commencer** (valeur initiale 35).

**Limite assumee** : ces courbes sont des moyennes de cohorte. La
variabilite interindividuelle est enorme et n'est pas modelisee. Ces
grandeurs INFORMENT le joueur, elles ne declenchent aucune condition de fin
-- il n'existe pas de seuil publie « au-dela de X points de fatigue, on
arrete », et en inventer un aurait ete une faute.

#### 6.4 Couts : cout complet, pas tarif

Deux chiffres coexistent pour un meme acte : le tarif rembourse et le cout
complet de production. L'etude **STIC 2002** (RCMI en ORL, *Bulletin du
Cancer* 2006;93(10)) mesure un cout complet de **10 916 EUR** (2 773 de
preparation + 8 143 de seances) contre une T2A moyenne de 6 987 EUR : le
tarif etait inferieur au cout reel pour **70 % des patients**. Le simulateur
retient le cout complet, seul reflet de la consommation de ressources.

- Accelerateur bi-energie : 1,15-1,25 M EUR ; collimateur multilame
  335-490 k EUR ; centre a deux accelerateurs **7,1 M EUR** pour
  **800 patients/an** (Bulletin du Cancer, dossier « Le cout de la
  radiotherapie »).
- Cout moyen d'un acte de radiotherapie en France **1 700 EUR** ; moyenne
  europeenne **3 800 EUR** (3 200 Irlande - 5 340 Luxembourg)
  (Cancer/Radiotherapie 2005, doi:10.1016/j.canrad.2005.05.002).
- Effet du fractionnement (Cancer/Radiotherapie 2018,
  doi:10.1016/j.canrad.2018.06.005, 91 patients, metastases osseuses) :
  30 Gy/10 fr **3 291 EUR** ; 20 Gy/5 fr 1 839 ; 8 Gy/1 fr **1 151** ;
  stereotaxie 5 668. Le nombre de fractions est le premier determinant.
- Transport sanitaire : **690 EUR** par patient et par traitement.
- Imagerie, tarifs opposables CCAM 2017 : TDM 1 territoire 25 EUR,
  2 territoires 51 EUR, 3 territoires ou plus (cancerologie) 76 EUR. Ces
  tarifs ne couvrent que l'acte intellectuel ; le module retient un cout
  complet superieur, par coherence de methode.

**Ces trois couches sont activables independamment** (`blind_mode`,
`track_patient`, `track_costs`) et `blind_mode` est **desactive par
defaut** : tous les equilibrages mesures de `JEU.md` restent valides.
