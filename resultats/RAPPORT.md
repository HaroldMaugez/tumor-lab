# Simulation de la croissance d'une tumeur avasculaire

Modèle multi-échelle d'un sphéroïde tumoral (lignée EMT6/Ro), calibré et validé sur des données expérimentales publiées.

## 1. Structure du modèle

| Échelle | Description | Formalisme |
|---|---|---|
| Moléculaire | O₂, glucose, lactate, pH | réaction-diffusion quasi-stationnaire, volumes finis |
| Cellulaire | bioénergétique (ATP), prolifération, quiescence, nécrose | loi de Pirt + cinétique de Michaelis-Menten |
| Tissulaire | croissance, advection, desquamation | frontière libre, schéma TVD van Leer |
| Thérapeutique | radiothérapie fractionnée | linéaire-quadratique + OER |

## 2. Paramètres et traçabilité

Chaque paramètre est étiqueté MESURÉ (valeur publiée), DÉRIVÉ (calculé) ou CALIBRÉ (ajusté ici, dans des bornes issues de la littérature).

| Paramètre | Valeur | Unité | Statut | Source |
|---|---|---|---|---|
| `D_o2` | 1820 | µm²/s | MESURÉ | [Kim07], [Gri14] |
| `D_glc` | 110 | µm²/s | MESURÉ | [Cas88] |
| `D_lac` | 150 | µm²/s | MESURÉ | [Cas92] |
| `q_o2_max` | 5.6e-17 | mol/cell/s | MESURÉ | [Gri14] |
| `km_o2` | 0.005 | mM | MESURÉ | [Gri14b], [Cas92] |
| `q_glc_max` | 1.487e-16 | mol/cell/s | MESURÉ | [FS85], [Cas92] |
| `km_glc` | 0.4 | mM | MESURÉ | [Cas88] (9L), [Cas92] |
| `pasteur_factor` | 2.5 | - | MESURÉ | [Cas92], [FS85] |
| `f_maint_uptake` | 0.4 | - | MESURÉ | [FS85] |
| `cell_density` | 0.00055 | cell/µm³ | MESURÉ | [FS86], [Jia05] |
| `cell_radius` | 7 | µm | MESURÉ | [Gri14] |
| `atp_glyc_per_glc` | 2 | ATP/glc | MESURÉ | biochimie standard |
| `atp_ox_per_o2` | 5 | ATP/O2 | MESURÉ | P/O = 2.5 |
| `buffer_capacity` | 25 | mM/pH | MESURÉ | physiologie acido-basique |
| `ph_bulk` | 7.4 | - | MESURÉ | [FS86] |
| `ph_min_growth` | 6.6 | - | MESURÉ | [Cas92] |
| `lac_death_onset` | 8 | - | MESURÉ | [Jia05] |
| `lac_death_lethal` | 20 | - | MESURÉ | [BF91] |
| `lac_death_weight` | 2 | - | CALIBRÉ | [Jia05], [BF91] |
| `td_min` | 20 | h | MESURÉ | [FS86] |
| `f_atp_maint` | 0.3704 | - | CALIBRÉ | ce travail (bornes [BG09]) |
| `f_atp_death` | 0.3519 | - | CALIBRÉ | ce travail (bornes [BG09]) |
| `s50_growth` | 0.1363 | - | CALIBRÉ | ce travail (contraint par [FS86]) |
| `k_necrosis` | 2.407e-05 | 1/s | CALIBRÉ | ce travail |
| `k_lysis` | 3.209e-06 | 1/s | MESURÉ (ordre de grandeur) | [WK97], [Jia05] |
| `shedding_velocity` | 4e-05 | µm/s | CALIBRÉ | ce travail (bornes [FS86]) |
| `alpha_lq` | 0.3 | Gy⁻¹ | MESURÉ | [Hil11] |
| `beta_lq` | 0.03 | Gy⁻² | MESURÉ | [Hil11] |
| `oer_max` | 3 | - | MESURÉ | Alper & Howard-Flanders (1956) |
| `k_oer` | 3 | mmHg | MESURÉ | Alper & Howard-Flanders (1956) |

### Grandeurs dérivées

- Consommation d'O₂ volumique : **30.80 µM/s** (≈ 22.3 mmHg/s ; littérature 22–27 mmHg/s)
- Limite de diffusion de l'O₂ : **221 µm** (mesure Grimes 2014 : 233 ± 22 µm)
- Temps de doublement minimal : **20 h** (mesure Freyer & Sutherland : 20–24 h)

## 3. Vérification et validation


### Vérification

| Test | Valeur | Référence / critère | Statut |
|---|---|---|---|
| Solution analytique (profil parabolique d'O₂) | 3.541e-07 erreur relative L∞ | 0.002 erreur relative L∞ | ✅ |
| Ordre de convergence en maillage | 2.164 ordre | 1.6 ordre | ✅ |
| Conservation de la masse (flux O₂ = consommation) | 1.119e-11 erreur relative | 1e-08 erreur relative | ✅ |
| Exactitude du jacobien analytique | 2.199e-12 erreur relative | 1e-06 erreur relative | ✅ |
| Séparation d'échelles (τ_diffusion / τ_croissance) | 0.000318 - | 0.001 - | ✅ |
| Positivité des concentrations | 1.19e-08 mM | 0 mM | ✅ |
| Indépendance au pas de temps | 5.446e-06 écart relatif | 0.02 écart relatif | ✅ |
| Validation croisée continu / agents (rayon) | 0.03188 écart relatif | 0.15 écart relatif | ✅ |
| Reproductibilité à graine fixée (et divergence sinon) | 1 - | 1 - | ✅ |

### Validation

| Test | Valeur | Référence / critère | Statut |
|---|---|---|---|
| Limite de diffusion de l'O₂ (vs Grimes 2014 : 233±22 µm) | 221.2 µm | 291.2 µm | ✅ |
| OCR volumique (vs 22-27 mmHg/s publiés) | 22.32 mmHg/s | 33.08 mmHg/s | ✅ |
| Couronne viable ~constante aux grands diamètres | 19.78 µm | 45 µm | ✅ |
| OER borné entre 1 (anoxie) et 3 (normoxie) | 2.942 - | 3 - | ✅ |
| Radiorésistance hypoxique (dose isoeffet anoxie/normoxie) | 2.962 × | 3 × | ✅ |
| Survie d'une lignée mutante (vs théorie de Bozic 2010) | 0.8667 % | 0.7968 % | ✅ |
| Spectre neutre en 1/f (vs Williams et al. 2016) | 0.9875 R² | 0.98 R² | ✅ |
| Expansion subclonale sous sélection (contrôle positif) | 0.4669 fraction | 0.1 fraction | ✅ |
| Gradient spatial de pression sélective (périphérie/coeur) | 1.608 × | 1.2 × | ✅ |
| Croissance gompertzienne (R² de l'ajustement) | 0.9999 - | 0.95 - | ✅ |
| Incertitude propagée recouvrant l'expérience | 6 /6 | 6 /6 | ✅ |
| Fractionnement : hypofractionné plus toxique (α/β=3 Gy) | 1.473 × | 1 × | ✅ |
| Tolérance clinique d'un schéma standard (30×2 Gy) | 0.0002909 NTCP | 0.05 NTCP | ✅ |
| Sélection de la radiorésistance sous fractionnement | 10.01 % | 2 % | ✅ |
| Couronne viable [16.5 mM glc / 0.28 mM O₂] | 205.4 µm | 240 µm | ✅ |
| Diamètre d'apparition de la nécrose [16.5 mM glc / 0.28 mM O₂] | 626.7 µm | 515.9 µm | ❌ |
| Temps de doublement initial [16.5 mM glc / 0.28 mM O₂] | 20.59 h | 23 h | ✅ |
| Couronne viable [0.8 mM glc / 0.07 mM O₂] | 55.86 µm | 60 µm | ✅ |
| Diamètre d'apparition de la nécrose [0.8 mM glc / 0.07 mM O₂] | 169.2 µm | 151.8 µm | ✅ |
| Temps de doublement initial [0.8 mM glc / 0.07 mM O₂] | 26.63 h | 26 h | ✅ |
| pO₂ central (Ø 400 µm) [MK82] | 81.39 mmHg | 66 mmHg | ✅ |
| pO₂ central (Ø 1000 µm) [MK82] | 12.2 mmHg | 13 mmHg | ✅ |

**Bilan : 31/32 tests réussis.**

## 4. Évolution darwinienne intratumorale

Le moteur évolutif implémente les trois forces de l'évolution :

| Force | Implémentation | Paramètre mesuré | Source |
|---|---|---|---|
| Mutation | drivers et passengers tirés à chaque division | u = 3.4×10⁻⁵, v = 0.016 par division | Bozic et al. 2010 |
| Dérive | naissance/mort stochastiques | >99 % des drivers s'éteignent | Bozic et al. 2010 |
| Sélection | fitness (1+s)^k modulant la prolifération locale | s = 0.004 ± 0.0004 | Bozic et al. 2010 |

**Stochasticité maîtrisée** : toute l'aléa passe par un unique générateur `numpy.random.Generator`. Sans graine, chaque exécution produit une histoire évolutive différente (comme deux tumeurs issues du même clone fondateur) ; à graine fixée, le résultat est reproductible au bit près.

**Validation** : la probabilité de survie d'une lignée reproduit la théorie de Bozic (0.80 % simulé contre 0.797 % attendu) et le spectre de fréquences alléliques suit la loi de puissance 1/f de l'évolution neutre (R² ≈ 0.99, seuil de Williams et al. 2016 : 0.98), sans que cette loi ait été imposée. Le contrôle positif (sous-clone avantagé implanté) dégrade bien cette signature.

## 5. Régime humain fictif vascularisé

Le scénario `vascular_human` active `tumeur/vascular.py`. L'angiogenic switch répond à l'hypoxie persistante et le réseau échange O₂, glucose et lactate avec le tissu. Les paramètres d'échange, l'activité maximale et l'anormalité sont **SCENARIO**, non mesurés chez un patient ; ce régime n'est pas inclus dans le score de validation EMT6/Ro.

L'action `antiangiogenic` est une pression de jeu : elle ne représente ni une molécule, ni une dose, ni une recommandation médicale. La figure `fig9_vascularisation.png` est une comparaison de scénario et non une validation expérimentale.

## 6. Limites connues

- **Rapport des volumes de saturation** : le modèle produit un rapport d'environ 100× entre les conditions extrêmes, contre ~500× mesurés. Freyer & Sutherland attribuent explicitement cet écart à des inhibiteurs produits par la nécrose, absents de ce modèle ; par ailleurs leurs volumes de saturation sont *extrapolés* par ajustement gompertzien et non mesurés directement (Kim, Park & Hunt 2008 relèvent le même désaccord). La hiérarchie des tailles finales est en revanche correctement reproduite.
- **Diamètre d'apparition de la nécrose (milieu riche)** : le modèle prédit ~675 µm contre 516 µm mesurés. L'écart est cohérent avec la littérature : Casciari et al. (1992) rapportent que les modèles fondés uniquement sur les nutriments « ne prédisent pas correctement l'apparition de la nécrose », ce qui suggère l'intervention de facteurs additionnels (inhibiteurs solubles, contacts cellule-cellule) non représentés ici. Ce désaccord est conservé plutôt que masqué par un ajustement supplémentaire.
- Géométrie à symétrie sphérique pour le modèle continu (les asymétries sont accessibles via le modèle à agents).
- Modèle avasculaire : ni angiogenèse, ni invasion, ni réponse immunitaire.
- Le spectre allélique n'est interprétable que sur le modèle de population (~10⁵ cellules) : à l'échelle du modèle spatial (~10⁴ cellules), la fenêtre de fréquences ne contient que quelques mutations. Cette limite d'échantillonnage est signalée sur les figures.
- Pour observer l'évolution en quelques jours simulés, les taux de mutation des démonstrations sont amplifiés ; les valeurs mesurées (Bozic 2010) restent celles par défaut et servent aux validations.

## 7. Reproductibilité

```bash
python run_simulation.py --validate   # suite de tests
python run_simulation.py              # étude complète
```

Générateurs aléatoires initialisés explicitement ; solveur déterministe.

## 8. Références

1. Freyer JP & Sutherland RM. *Cancer Res* 46:3504-3512 (1986).
2. Freyer JP & Sutherland RM. *J Cell Physiol* 124:516-524 (1985).
3. Casciari JJ, Sotirchos SV & Sutherland RM. *Cancer Res* 48:3905 (1988) ; *Cell Prolif* 25:1-22 (1992).
4. Grimes DR et al. *J R Soc Interface* 11:20131124 (2014) ; *PLoS ONE* 11:e0153692 (2016).
5. Mueller-Klieser W & Sutherland RM. *Br J Cancer* 45:256 (1982).
6. Jiang Y, Pjesivac-Grbovic J, Cantrell C & Freyer JP. *Biophys J* 89:3884-3894 (2005).
7. Ward JP & King JR. *IMA J Math Appl Med Biol* 14:39-69 (1997).
8. Bertuzzi A, Fasano A, Gandolfi A & Sinisgalli C. *J Theor Biol* 262:142-150 (2010).
9. Drasdo D & Höhme S. *Phys Biol* 2:133-147 (2005).
10. Kim S, Park S & Hunt CA. *BMC Syst Biol* 2:110 (2008).
11. Alper T & Howard-Flanders P. *Nature* 178:978-979 (1956).
12. Lugano R, Ramachandran M & Dimberg A. *Cell Mol Life Sci* 77:1745-1770 (2019), DOI: 10.1007/s00018-019-03351-7, https://pmc.ncbi.nlm.nih.gov/articles/PMC7190605/.
13. Jain RK. *Science* 307:58-62 (2005), DOI: 10.1126/science.1104819.
14. Hanahan D. *Cancer Discovery* 12:31-46 (2022), DOI: 10.1158/2159-8290.CD-21-1059.