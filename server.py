#!/usr/bin/env python3
"""
Serveur du moteur tumoral pour Unity — protocole JSON Lines sur stdin/stdout.

Pourquoi stdin/stdout et pas HTTP
---------------------------------
Unity lance ce script comme processus enfant. Pas de port à ouvrir, pas de
pare-feu, pas de conflit de port, arrêt automatique quand le jeu se ferme, et
la latence est celle d'un tube (~0.1 ms) contre ~1 ms pour une boucle HTTP
locale. Le mode HTTP reste disponible (`--http PORT`) pour déboguer depuis un
navigateur ou pour un build WebGL, qui ne peut pas lancer de sous-processus.

Protocole
---------
Une requête JSON par ligne, une réponse JSON par ligne :

    -> {"cmd": "reset", "config": {"seed": 1}}
    <- {"ok": true, "observation": {...}}

    -> {"cmd": "step", "action": {"type": "irradiate", "dose_gy": 2.0}}
    <- {"ok": true, "observation": {...}, "truth": {...}, "reward": -0.3, ...}

Commandes : ping, reset, step, state, profile, provenance, actions, ai_action, history, quit.
`provenance` expose le statut SCENARIO des paramètres vasculaires; il ne transforme
pas une fermeture phénoménologique en mesure clinique.

IMPORTANT — un tour coûte ~1 s de calcul (mesuré : médiane 1.15 s, max 1.7 s).
Unity DOIT appeler ce serveur sur un thread de fond, jamais dans Update(),
sinon l'image se fige. Le script C# fourni dans UNITY.md le fait déjà.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from tumeur.game import (TumorGame, GameConfig, SCENARIOS, make_config,
                           GLUCOSE_LEVELS, OXYGEN_LEVELS)
from tumeur.parameters import Parameters
from tumeur.ai import recommend_action


def _jsonable(o):
    """Numpy -> types Python natifs (json ne sait pas sérialiser np.float64)."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


class Engine:
    def __init__(self):
        self.game: TumorGame | None = None

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd", "")
        if cmd == "ping":
            return {"ok": True, "pong": True, "version": "2.0", "features": [
                "vascular_human", "ai_action", "leaderboard_score"]}

        if cmd == "scenarios":
            return {"ok": True, "scenarios": _jsonable(SCENARIOS)}

        if cmd == "reset":
            cfg_in = dict(req.get("config") or {})
            scenario = req.get("scenario")
            if scenario:
                try:
                    cfg = make_config(scenario, **cfg_in)
                except (ValueError, TypeError) as e:
                    return {"ok": False, "error": str(e)}
                self.game = TumorGame(cfg)
                return {"ok": True, "scenario": scenario,
                        "observation": _jsonable(self.game.observe()),
                        "config": _jsonable(asdict(cfg))}
            valid = {f for f in GameConfig.__dataclass_fields__}
            unknown = set(cfg_in) - valid
            if unknown:
                return {"ok": False, "error": f"champs de config inconnus: {sorted(unknown)}",
                        "valid_fields": sorted(valid)}
            cfg = GameConfig(**cfg_in)
            self.game = TumorGame(cfg)
            return {"ok": True, "observation": _jsonable(self.game.observe()),
                    "config": _jsonable(asdict(cfg))}

        if self.game is None:
            return {"ok": False, "error": "moteur non initialisé : appeler reset d'abord"}

        if cmd == "step":
            r = self.game.step(req.get("action"))
            return {"ok": True, "turn": r.turn, "day": r.day,
                    "observation": _jsonable(r.observation),
                    "truth": _jsonable(r.truth), "reward": r.reward,
                    "done": r.done, "outcome": r.outcome, "info": _jsonable(r.info)}

        if cmd == "state":
            return {"ok": True, "observation": _jsonable(self.game.observe()),
                    "truth": _jsonable(self.game.truth_state())}

        if cmd == "profile":
            # profils radiaux pour l'affichage 3D / la carte de chaleur
            s = self.game._snapshot()
            from tumeur.units import conc_to_po2
            profile = {
                "r_um": s.r, "phi_live": s.phi_live,
                "po2_mmHg": conc_to_po2(s.c_o2), "glucose_mM": s.c_glc,
                "lactate_mM": s.c_lac, "ph": s.ph,
                "radius_um": s.radius, "necrotic_radius_um": s.necrotic_radius,
                "model_regime": self.game.cfg.model_regime,
                "vascular_activity": self.game.vascular.activity,
            }
            if self.game.vascular.enabled:
                vp = self.game.vascular.profiles(s.xi)
                profile["vascular_density"] = vp["relative_density"]
            return {"ok": True, "profile": _jsonable(profile)}

        if cmd == "ai_action":
            return {"ok": True, "action": _jsonable(recommend_action(self.game)),
                    "model_regime": self.game.cfg.model_regime}

        if cmd == "provenance":
            return {"ok": True, "model_regime": self.game.cfg.model_regime,
                    "vascular": self.game.vascular.provenance()}

        if cmd == "actions":
            return {"ok": True, "actions": [
                {"type": "wait"},
                {"type": "irradiate", "dose_gy": [0.0, self.game.cfg.dose_max_per_turn]},
                {"type": "set_medium", "glucose_mM": list(GLUCOSE_LEVELS),
                 "oxygen_mM": list(OXYGEN_LEVELS)},
                {"type": "biopsy"},
                {"type": "po2_imaging"},
                {"type": "antiangiogenic", "intensity": [0.0, 1.0],
                 "available_in": ["vascular_human"]},
            ]}

        if cmd == "history":
            return {"ok": True, "history": _jsonable(self.game.history)}

        if cmd == "quit":
            return {"ok": True, "bye": True}

        return {"ok": False, "error": f"commande inconnue: {cmd}"}


def run_stdio():
    eng = Engine()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = eng.handle(req)
        except Exception as e:
            resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        if isinstance(req, dict) and req.get("cmd") == "quit":
            break


def run_http(port: int):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    eng = Engine()

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_OPTIONS(self):
            self._send({"ok": True})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                self._send(eng.handle(req))
            except Exception as e:
                self._send({"ok": False, "error": str(e)}, 500)

        def do_GET(self):
            self._send(eng.handle({"cmd": "state"} if eng.game else {"cmd": "ping"}))

        def log_message(self, *a):
            pass

    print(f"[moteur] HTTP sur 0.0.0.0:{port}", file=sys.stderr, flush=True)
    HTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Moteur tumoral pour Unity")
    ap.add_argument("--http", type=int, default=None,
                    help="servir en HTTP sur ce port au lieu de stdin/stdout")
    a = ap.parse_args()
    if a.http:
        run_http(a.http)
    else:
        run_stdio()
