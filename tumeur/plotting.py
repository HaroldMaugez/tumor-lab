"""Figures scientifiques (matplotlib, sans dépendance à un backend interactif)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import benchmarks as bm
from .growth import Snapshot, SpheroidSimulation
from .parameters import Parameters
from .units import conc_to_po2

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})

C_O2, C_GLC, C_LAC, C_PH = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"


def plot_growth_curves(sim: SpheroidSimulation, path: str,
                       targets: dict | None = None) -> str:
    """Cinétique de croissance : diamètre, cellules, structure, microenvironnement."""
    a = sim.as_arrays()
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].plot(a["t_days"], a["diameter_um"], color="k", lw=2)
    ax[0, 0].set_xlabel("Temps (jours)"); ax[0, 0].set_ylabel("Diamètre (µm)")
    ax[0, 0].set_title("Croissance du sphéroïde")

    ax[0, 1].semilogy(a["t_days"], np.maximum(a["n_cells"], 1), "k-", lw=2,
                      label="total")
    ax[0, 1].semilogy(a["t_days"], np.maximum(a["n_prolif"], 1), "-", color=C_GLC,
                      label="prolifératives")
    ax[0, 1].semilogy(a["t_days"], np.maximum(a["n_quiescent"], 1), "-",
                      color="#ff7f0e", label="quiescentes")
    ax[0, 1].set_xlabel("Temps (jours)"); ax[0, 1].set_ylabel("Nombre de cellules")
    ax[0, 1].set_title("Populations cellulaires"); ax[0, 1].legend()

    ax[1, 0].plot(a["t_days"], a["radius_um"], "k-", lw=2, label="rayon total")
    ax[1, 0].plot(a["t_days"], a["necrotic_radius_um"], "-", color=C_LAC,
                  label="rayon nécrotique")
    ax[1, 0].fill_between(a["t_days"], a["necrotic_radius_um"], a["radius_um"],
                          color=C_GLC, alpha=0.25, label="couronne viable")
    if targets and "viable_rim_um" in targets:
        ax[1, 0].axhline(targets["viable_rim_um"], ls=":", color="gray")
    ax[1, 0].set_xlabel("Temps (jours)"); ax[1, 0].set_ylabel("Rayon (µm)")
    ax[1, 0].set_title("Structure radiale"); ax[1, 0].legend()

    ax2 = ax[1, 1]
    ax2.plot(a["t_days"], a["po2_center_mmHg"], color=C_O2, label="pO₂ centre (mmHg)")
    ax2.plot(a["t_days"], a["glc_center_mM"], color=C_GLC, label="glucose centre (mM)")
    ax2.set_xlabel("Temps (jours)"); ax2.set_ylabel("Concentration")
    ax3 = ax2.twinx(); ax3.plot(a["t_days"], a["ph_center"], color=C_PH, ls="--")
    ax3.set_ylabel("pH au centre", color=C_PH); ax3.grid(False)
    ax2.set_title("Microenvironnement central"); ax2.legend(loc="upper right")

    fig.suptitle(f"Croissance avasculaire — {sim.p.label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_radial_profiles(snapshots: Sequence[Snapshot], path: str) -> str:
    """Profils radiaux à plusieurs instants."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))

    for c, s in zip(cmap, snapshots):
        lbl = f"{s.t_days:.0f} j (Ø {s.diameter:.0f} µm)"
        ax[0, 0].plot(s.r, s.po2, color=c, label=lbl)
        ax[0, 1].plot(s.r, s.c_glc, color=c)
        ax[1, 0].plot(s.r, s.ph, color=c)
        ax[1, 1].plot(s.r, s.phi_live, color=c)

    ax[0, 0].axhline(10, ls=":", color="gray")
    ax[0, 0].text(0.02, 0.06, "seuil d'hypoxie (10 mmHg)", transform=ax[0, 0].transAxes,
                  fontsize=7, color="gray")
    ax[0, 0].set_ylabel("pO₂ (mmHg)"); ax[0, 0].set_title("Oxygène")
    ax[0, 1].set_ylabel("Glucose (mM)"); ax[0, 1].set_title("Glucose")
    ax[1, 0].set_ylabel("pH extracellulaire"); ax[1, 0].set_title("Acidité")
    ax[1, 1].set_ylabel("Fraction de cellules vivantes")
    ax[1, 1].set_title("Structure (viable / nécrotique)")
    for a_ in ax.flat:
        a_.set_xlabel("Rayon (µm)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Profils radiaux au cours de la croissance", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_spheroid_map(snap: Snapshot, path: str) -> str:
    """Coupe 2D du sphéroïde : oxygène et structure histologique."""
    n = 340
    L = snap.radius * 1.12
    x = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, x)
    Rr = np.sqrt(X ** 2 + Y ** 2)

    po2 = np.interp(Rr, snap.r, snap.po2, right=np.nan)
    phi = np.interp(Rr, snap.r, snap.phi_live, right=np.nan)
    outside = Rr > snap.radius
    po2 = np.where(outside, np.nan, po2)
    phi = np.where(outside, np.nan, phi)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    im0 = ax[0].imshow(po2, extent=[-L, L, -L, L], origin="lower",
                       cmap="inferno", vmin=0)
    ax[0].set_title("Pression partielle d'O₂ (mmHg)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)

    histo = LinearSegmentedColormap.from_list(
        "histo", ["#3b2410", "#a9713f", "#e8c39e", "#c94f4f"])
    im1 = ax[1].imshow(phi, extent=[-L, L, -L, L], origin="lower",
                       cmap=histo, vmin=0, vmax=1)
    ax[1].set_title("Fraction viable (aspect histologique)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)

    for a_ in ax:
        a_.set_xlabel("x (µm)"); a_.set_ylabel("y (µm)"); a_.grid(False)
        a_.add_patch(plt.Circle((0, 0), snap.radius, fill=False,
                                color="white", lw=1.0, alpha=0.8))
    fig.suptitle(f"Coupe équatoriale à t = {snap.t_days:.1f} j "
                 f"(Ø {snap.diameter:.0f} µm, couronne viable {snap.viable_rim:.0f} µm)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_media_comparison(results: Dict[str, dict], path: str) -> str:
    """Croissance sous différentes conditions de culture (protocole [FS86])."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(results)))
    for c, (label, r) in zip(cmap, results.items()):
        ax[0].plot(r["t_days"], r["diameter_um"], color=c, label=label)
        ax[1].semilogy(r["t_days"], np.maximum(r["n_cells"], 1), color=c, label=label)
    ax[0].set_xlabel("Temps (jours)"); ax[0].set_ylabel("Diamètre (µm)")
    ax[0].set_title("Diamètre du sphéroïde")
    ax[1].set_xlabel("Temps (jours)"); ax[1].set_ylabel("Nombre de cellules")
    ax[1].set_title("Nombre de cellules")
    ax[1].legend(fontsize=7, title="milieu (O₂ / glucose)")
    fig.suptitle("Effet de l'apport en nutriments (protocole Freyer & Sutherland 1986)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_therapy(control, treated, log: List[dict], path: str) -> str:
    """Réponse à la radiothérapie fractionnée."""
    a0, a1 = control.as_arrays(), treated.as_arrays()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    ax[0].plot(a0["t_days"], a0["diameter_um"], "k-", lw=2, label="témoin")
    ax[0].plot(a1["t_days"], a1["diameter_um"], "-", color=C_LAC, lw=2, label="irradié")
    for e in log:
        ax[0].axvline(e["day"], color="gray", alpha=0.3, lw=0.8)
    ax[0].set_xlabel("Temps (jours)"); ax[0].set_ylabel("Diamètre (µm)")
    ax[0].set_title("Réponse tumorale"); ax[0].legend()

    ax[1].semilogy(a0["t_days"], np.maximum(a0["n_cells"], 1), "k-", lw=2,
                   label="témoin")
    ax[1].semilogy(a1["t_days"], np.maximum(a1["n_cells"], 1), "-", color=C_LAC,
                   lw=2, label="irradié")
    ax[1].set_xlabel("Temps (jours)"); ax[1].set_ylabel("Cellules vivantes")
    ax[1].set_title("Charge tumorale"); ax[1].legend()

    if log:
        days = [e["day"] for e in log]
        ax[2].plot(days, [e["mean_SF"] for e in log], "o-", color=C_O2,
                   label="survie moyenne")
        ax[2].plot(days, [e["min_SF"] for e in log], "v--", color=C_GLC,
                   label="min (bien oxygéné)")
        ax[2].plot(days, [e["max_SF"] for e in log], "^--", color=C_LAC,
                   label="max (hypoxique)")
        ax[2].set_yscale("log")
    ax[2].set_xlabel("Jour de la fraction"); ax[2].set_ylabel("Fraction survivante")
    ax[2].set_title("Survie par fraction (effet oxygène)"); ax[2].legend(fontsize=7)

    fig.suptitle("Radiothérapie fractionnée — modèle linéaire-quadratique avec OER",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_validation(checks, path: str) -> str:
    """Synthèse graphique modèle vs expérience."""
    exp = [c for c in checks if c.category == "validation" and c.criterion > 0
           and np.isfinite(c.value)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(exp))
    ratio = [c.value / c.criterion for c in exp]
    colors = ["#2ca02c" if c.passed else "#d62728" for c in exp]
    ax.barh(y, ratio, color=colors, alpha=0.8)
    ax.axvline(1.0, color="k", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([c.name[:52] for c in exp], fontsize=7)
    ax.set_xlabel("Modèle / mesure expérimentale (1 = accord parfait)")
    ax.set_title("Validation face aux données EMT6/Ro publiées", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_agent_model(abm, path: str) -> str:
    """Visualisation 3D du modèle à agents + validation croisée."""
    from .agents import ALIVE, NECROTIC, QUIESCENT
    fig = plt.figure(figsize=(11, 4.4))

    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    pos, st = abm.pos, abm.state
    keep = np.abs(pos[:, 2]) < abm.p.cell_radius * 2.5     # coupe centrale
    cols = np.array(["#2ca02c", "#ff7f0e", "#8b4513"])[st[keep]]
    ax0.scatter(pos[keep, 0], pos[keep, 1], pos[keep, 2], c=cols, s=6, alpha=0.85)
    ax0.set_title("Coupe du sphéroïde (agents)"); ax0.grid(False)
    ax0.set_xlabel("x (µm)"); ax0.set_ylabel("y (µm)")

    h = abm.history
    t = [r["t_days"] for r in h]
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.semilogy(t, [r["n_cells"] for r in h], "k-", label="total")
    ax1.semilogy(t, [max(r["n_alive"], 1) for r in h], color="#2ca02c",
                 label="prolifératives")
    ax1.semilogy(t, [max(r["n_quiescent"], 1) for r in h], color="#ff7f0e",
                 label="quiescentes")
    ax1.semilogy(t, [max(r["n_necrotic"], 1) for r in h], color="#8b4513",
                 label="nécrotiques")
    ax1.set_xlabel("Temps (jours)"); ax1.set_ylabel("Cellules")
    ax1.set_title("Populations (modèle à agents)"); ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(t, [r["radius_um"] for r in h], "k-", label="agents")
    ax2.set_xlabel("Temps (jours)"); ax2.set_ylabel("Rayon (µm)")
    ax2.set_title("Rayon du sphéroïde"); ax2.legend(fontsize=7)

    fig.suptitle("Modèle à agents 3D (validation croisée du modèle continu)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_evolution(abm, path: str, replicates: List[dict] | None = None,
                   vaf_reference: np.ndarray | None = None) -> str:
    """Évolution darwinienne : diversité clonale, spectre allélique, carte spatiale."""
    from .evolution import cumulative_vaf, neutrality_test

    fig = plt.figure(figsize=(12.5, 7.4))
    h = abm.history
    t = [r["t_days"] for r in h]

    # (a) diversité clonale au cours du temps
    ax = fig.add_subplot(2, 3, 1)
    ax.plot(t, [r.get("n_clones", 0) for r in h], color="#1f77b4")
    ax.set_xlabel("Temps (jours)"); ax.set_ylabel("Nombre de clones")
    ax.set_title("Diversité clonale")

    # (b) indice de Shannon
    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, [r.get("shannon", 0) for r in h], color="#2ca02c")
    ax.set_xlabel("Temps (jours)"); ax.set_ylabel("Indice de Shannon")
    ax.set_title("Hétérogénéité intratumorale")

    # (c) fitness et drivers
    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, [r.get("mean_fitness", 1) for r in h], color="#d62728",
            label="fitness moyenne")
    ax.set_xlabel("Temps (jours)"); ax.set_ylabel("Fitness moyenne", color="#d62728")
    ax2 = ax.twinx()
    ax2.plot(t, [r.get("mean_drivers", 0) for r in h], color="#9467bd", ls="--")
    ax2.set_ylabel("Drivers par cellule", color="#9467bd"); ax2.grid(False)
    ax.set_title("Sélection : fitness et drivers")

    # (d) spectre de fréquences alléliques (test 1/f)
    #
    # Le spectre est calculé sur le modèle de POPULATION (1e5 cellules) et non
    # sur le modèle spatial : à 1e4 cellules, la fenêtre [0.02, 0.5] ne contient
    # que quelques mutations et le R² n'est pas interprétable. C'est une limite
    # d'échantillonnage assumée, indiquée sur la figure.
    ax = fig.add_subplot(2, 3, 4)
    vaf = vaf_reference if vaf_reference is not None else abm.allele_frequencies()
    n_win = int(np.sum(vaf >= 0.02)) if vaf.size else 0
    if n_win >= 20:
        fs, m = cumulative_vaf(vaf)
        ax.plot(1.0 / fs, m, "o", ms=3, color="#1f77b4", label="simulation")
        A = np.vstack([1.0 / fs, np.ones_like(fs)]).T
        coef, *_ = np.linalg.lstsq(A, m, rcond=None)
        ax.plot(1.0 / fs, A @ coef, "-", color="#d62728",
                label=f"ajustement 1/f (R²={neutrality_test(vaf)['r2']:.3f})")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, f"échantillon insuffisant\n({n_win} mutations dans la "
                          "fenêtre)\nvoir le modèle de population",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="gray")
    ax.set_xlabel("1/f"); ax.set_ylabel("M(f) : mutations de fréquence ≥ f")
    ax.set_title("Spectre allélique (Williams 2016)")

    # (e) carte spatiale des domaines clonaux (traçage de lignage)
    ax = fig.add_subplot(2, 3, 5)
    pos, cid = abm.pos, abm.clone_id
    keep = np.abs(pos[:, 2]) < abm.p.cell_radius * 2.0
    if keep.sum() and abm.tree is not None:
        # On colore par l'ancêtre commun à faible profondeur : chaque teinte
        # est une lignée, ce qui révèle les domaines clonaux spatiaux.
        anc = np.array([abm.tree.ancestor_at_depth(int(c), depth=2)
                        for c in cid[keep]])
        uniq, inv = np.unique(anc, return_inverse=True)
        ax.scatter(pos[keep, 0], pos[keep, 1], c=plt.cm.tab20(inv % 20),
                   s=5, alpha=0.9)
        ax.set_title(f"Domaines clonaux ({uniq.size} lignées)")
    else:
        ax.set_title("Domaines clonaux")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_aspect("equal"); ax.grid(False)

    # (f) variabilité entre réalisations
    ax = fig.add_subplot(2, 3, 6)
    if replicates:
        for i, rep in enumerate(replicates):
            ax.plot(rep["t_days"], rep["n_clones"], alpha=0.8,
                    label=f"graine {rep['seed']}")
        ax.legend(fontsize=6)
        ax.set_title("Réalisations indépendantes")
    else:
        ax.set_title("Réalisations indépendantes (non calculées)")
    ax.set_xlabel("Temps (jours)"); ax.set_ylabel("Nombre de clones")

    fig.suptitle("Évolution darwinienne intratumorale — mutation, dérive, sélection",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def plot_vascular_comparison(avasim, vascularsim, path: str) -> str:
    """Compare un run avasculaire et le scénario vascularisé.

    La figure distingue une validation (avasculaire) d'un scénario mécaniste
    (vascularisé), afin d'éviter de présenter une simulation fictive comme une
    mesure clinique.
    """
    a0, a1 = avasim.as_arrays(), vascularsim.as_arrays()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(a0["t_days"], a0["diameter_um"], label="avasculaire validé", color="#777777")
    ax[0].plot(a1["t_days"], a1["diameter_um"], label="vascularisé — scénario", color="#c43c39")
    ax[0].set(xlabel="Temps (jours)", ylabel="Diamètre (µm)", title="Croissance")
    ax[0].legend(fontsize=7)
    ax[1].plot(a0["t_days"], a0["po2_center_mmHg"], label="avasculaire", color="#377eb8")
    ax[1].plot(a1["t_days"], a1["po2_center_mmHg"], label="vascularisé", color="#e41a1c")
    ax[1].set(xlabel="Temps (jours)", ylabel="pO₂ centrale (mmHg)", title="Oxygénation")
    ax[1].legend(fontsize=7)
    if hasattr(vascularsim, "vascular") and vascularsim.vascular.history:
        vh = vascularsim.vascular.history
        ax[2].plot(np.cumsum([x["day_step"] for x in vh]),
                   [x["activity"] for x in vh], color="#984ea3")
    ax[2].set(xlabel="Temps (jours)", ylabel="Activité relative", title="Angiogenic switch")
    fig.suptitle("Comparaison de régimes — mécanisme vascularisé non calibré patient",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
