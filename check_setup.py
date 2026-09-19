#!/usr/bin/env python3
"""
Vérification de l'installation, à lancer AVANT d'ouvrir Unity.

    python3 check_setup.py

Si tout est vert, le moteur fonctionnera sous Unity. Si quelque chose casse
ici, ce sera beaucoup plus pénible à diagnostiquer depuis la console Unity.
"""

import os
import subprocess
import sys
import time

OK = "  [OK]  "
KO = "  [KO]  "
problems = []


def check(label, fn):
    try:
        detail = fn()
        print(f"{OK}{label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"{KO}{label} — {e}")
        problems.append((label, str(e)))
        return False


print("=" * 68)
print("  Vérification du moteur tumoral")
print("=" * 68)

# 1. Python -------------------------------------------------------------
def _py():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} trop ancien, il faut 3.10+")
    return f"Python {v.major}.{v.minor}.{v.micro}"


check("Version de Python", _py)

# 2. Dépendances --------------------------------------------------------
def _numpy():
    import numpy
    return f"numpy {numpy.__version__}"


def _scipy():
    import scipy
    return f"scipy {scipy.__version__}"


ok_deps = check("numpy installé", _numpy) and check("scipy installé", _scipy)

if not ok_deps:
    print("\n  -> Installez les dépendances :")
    print(f"     {sys.executable} -m pip install numpy scipy\n")
    sys.exit(1)

# 3. Le paquet s'importe ------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _import():
    from tumeur.game import TumorGame, make_config, SCENARIOS
    return f"{len(SCENARIOS)} scénarios : {', '.join(sorted(SCENARIOS))}"


if not check("Import du moteur", _import):
    print("\n  -> Lancez ce script DEPUIS le dossier tumorsim/.\n")
    sys.exit(1)

# 4. Une partie tourne --------------------------------------------------
from tumeur.game import TumorGame, make_config


def _run():
    g = TumorGame(make_config("standard", seed=1))
    t0 = time.time()
    r = g.step({"type": "irradiate", "dose_gy": 2.0})
    dt = time.time() - t0
    if not (r.truth["diameter_um"] > 0):
        raise RuntimeError("diamètre nul")
    return f"1 tour en {dt:.2f} s (attendu ~1 s)"


check("Simulation d'un tour", _run)

# 5. Le serveur répond --------------------------------------------------
def _server():
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.Popen(
        [sys.executable, "-u", os.path.join(here, "server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=here,
    )
    try:
        p.stdin.write('{"cmd":"ping"}\n')
        p.stdin.write('{"cmd":"reset","scenario":"standard"}\n')
        p.stdin.write('{"cmd":"step","action":{"type":"irradiate","dose_gy":2.0}}\n')
        p.stdin.write('{"cmd":"quit"}\n')
        p.stdin.flush()
        import json
        lines = []
        for _ in range(4):
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("le serveur n'a pas répondu")
            lines.append(json.loads(line))
        bad = [l for l in lines if not l.get("ok")]
        if bad:
            raise RuntimeError(f"réponse en erreur : {bad[0].get('error')}")
        return "ping, reset, step, quit — 4/4"
    finally:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()


check("Serveur JSON (comme l'appellera Unity)", _server)

# 6. Chemin absolu à copier dans Unity ----------------------------------
here = os.path.dirname(os.path.abspath(__file__))
print("\n" + "=" * 68)
if problems:
    print(f"  {len(problems)} PROBLÈME(S) — à régler avant de passer à Unity")
    for lbl, err in problems:
        print(f"    - {lbl} : {err}")
    print("=" * 68)
    sys.exit(1)

print("  TOUT EST BON. Le moteur est prêt pour Unity.")
print("=" * 68)
print("\n  Chemin de l'exécutable Python à renseigner dans Unity :")
print(f"    {sys.executable}")
print("\n  Dossier à copier dans Assets/StreamingAssets/ :")
print(f"    {here}")
print()
