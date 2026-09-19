"""
Site web Tumor Lab — serveur Flask.

Lance avec :  python site/app.py
Ouvre        :  http://localhost:5000
"""
from __future__ import annotations

import csv
import io
import json
import os
import pickle
import sys
import time
import threading
import sqlite3
from dataclasses import asdict
from functools import wraps
from hashlib import pbkdf2_hmac
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, Response,
)
from werkzeug.exceptions import HTTPException

# --- Python engine path ---------------------------------------------------
ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

from tumeur.game import TumorGame, GameConfig, SCENARIOS, make_config
from tumeur.ai import recommend_action
import numpy as np

# --- App setup -------------------------------------------------------------
app = Flask(__name__)
# Une clé stable est indispensable : os.urandom() invalide toutes les sessions
# à chaque reload du worker PythonAnywhere. Elle doit être définie en production.
app.secret_key = os.environ.get("TUMORLAB_SECRET_KEY", "dev-only-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("TUMORLAB_COOKIE_SECURE", "0") == "1",
)


@app.errorhandler(Exception)
def handle_api_error(error):
    """Keep API failures JSON so the browser can show the real HTTP error.

    Flask's default 500 page is HTML, which otherwise causes a misleading
    ``JSON.parse: unexpected character`` error in the game interface.
    """
    if not request.path.startswith("/api/"):
        if isinstance(error, HTTPException):
            return error
        app.logger.exception("Erreur page %s", request.path)
        return "Une erreur interne est survenue.", 500
    if isinstance(error, HTTPException):
        status = error.code or 500
        message = error.description
    else:
        status = 500
        message = "Erreur interne pendant le calcul."
        app.logger.exception("Erreur API %s", request.path)
    return jsonify({"ok": False, "error": message}), status


DB_PATH = os.path.join(os.path.dirname(__file__), "tumorlab.db")


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            salt BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            scenario TEXT NOT NULL,
            score REAL DEFAULT 0,
            outcome TEXT DEFAULT '',
            turns INTEGER DEFAULT 0,
            dose_gy REAL DEFAULT 0,
            toxicity REAL DEFAULT 0,
            diameter_end REAL DEFAULT 0,
            trace_json TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS active_games (
            user_id INTEGER PRIMARY KEY,
            scenario TEXT NOT NULL,
            state BLOB NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Migration for databases created before trace_json was introduced.
    try:
        conn.execute("ALTER TABLE games ADD COLUMN trace_json TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def hash_password(password: str, salt: bytes = None) -> tuple:
    if salt is None:
        salt = os.urandom(16)
    h = pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return h, salt


def verify_password(password: str, stored_hash: bytes, salt: bytes) -> bool:
    h, _ = hash_password(password, salt)
    return h == stored_hash


# ═══════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    if "user_id" in session:
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        db.close()
        return user
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES — PAGES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    user = get_current_user()
    scenarios_info = {
        "tutoriel": {"name": "Tutoriel", "desc": "Apprenez les bases. Difficulté facile.", "icon": "📖"},
        "standard": {"name": "Standard", "desc": "Mode classique. Budget de dose limité.", "icon": "⚔️"},
        "organe_critique": {"name": "Organe critique", "desc": "L'organe est sensible. Aucune stratégie brutale ne passe.", "icon": "💔"},
        "vascular_human": {"name": "Humain fictif", "desc": "Angiogenèse, vaisseaux anormaux, action anti-angiogénique.", "icon": "🩸"},
        "in_vitro_strict": {"name": "Laboratoire", "desc": "EMT6/Ro strict. Le fractionnement n'y contrôle rien.", "icon": "🔬"},
    }
    return render_template("home.html", user=user, scenarios=scenarios_info)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 2:
            error = "Le nom d'utilisateur doit faire au moins 2 caractères."
        elif len(password) < 4:
            error = "Le mot de passe doit faire au moins 4 caractères."
        else:
            db = get_db()
            existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                error = "Ce nom d'utilisateur est déjà pris."
            else:
                pw_hash, salt = hash_password(password)
                db.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                           (username, pw_hash, salt))
                db.commit()
                user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                db.close()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect(url_for("home"))
            db.close()
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()
        if user and verify_password(password, user["password_hash"], user["salt"]):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        error = "Nom d'utilisateur ou mot de passe incorrect."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/play/<scenario>")
@login_required
def play(scenario):
    if scenario not in SCENARIOS:
        return redirect(url_for("home"))
    return render_template("game.html", scenario=scenario, user=get_current_user())


@app.route("/api/game/export")
@login_required
def api_game_export():
    """Export the current or last active game for analysis."""
    game = _get_game(session["user_id"])
    if game is None:
        return jsonify({"ok": False, "error": "Aucune partie disponible."}), 404
    fmt = request.args.get("format", "json").lower()
    payload = {
        "scenario": game.cfg.model_regime,
        "config": _jsonable(vars(game.cfg)),
        "observation": _jsonable(game.observe()),
        "history": _jsonable(game.history),
    }
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["tour", "jour", "action", "diametre_um", "toxicite", "score", "resultat"])
        for item in game.history:
            truth = item.get("truth", {})
            writer.writerow([
                item.get("turn", ""), truth.get("day", ""),
                item.get("action", {}).get("type", ""),
                truth.get("diameter_um", ""), truth.get("toxicity_score", ""),
                item.get("score", ""), game.outcome,
            ])
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=tumor-lab-partie.csv"},
        )
    if fmt != "json":
        return jsonify({"ok": False, "error": "Format demandé inconnu."}), 400
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=tumor-lab-partie.json"},
    )


@app.route("/learn")
def learn():
    return render_template("learn.html", user=get_current_user())


@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    rows = db.execute("""
        SELECT g.*, u.username
        FROM games g JOIN users u ON g.user_id = u.id
        WHERE g.scenario != 'tutoriel'
        ORDER BY g.score DESC
        LIMIT 50
    """).fetchall()
    db.close()
    return render_template("leaderboard.html", rows=rows, user=get_current_user())


@app.route("/tests")
def tests_page():
    return render_template("tests.html", user=get_current_user())


@app.route("/references")
def references():
    refs = [
        {"id": 1, "authors": "Freyer JP & Sutherland RM", "year": 1986, "title": "Regulation of growth saturation and development of necrosis in EMT6/Ro multicellular spheroids by the glucose and oxygen supply", "journal": "Cancer Res", "volume": "46", "pages": "3504–3512", "doi": "10.1111/j.1365-2184.1992.tb01433.x", "usage": "Source principale de validation. Matrice de 8 milieux, diamètres de saturation, couronnes viables, diamètres d'apparition de la nécrose.", "category": "Calibration"},
        {"id": 2, "authors": "Freyer JP & Sutherland RM", "year": 1985, "title": "A reduction in the in situ rates of oxygen and glucose consumption of cells in EMT6/Ro spheroids during growth", "journal": "J Cell Physiol", "volume": "124", "pages": "516–524", "doi": "10.1002/jcp.1041240323", "usage": "Consommation d'O₂ et de glucose par cellule divisée par 2.5 à 1300 µm.", "category": "Calibration"},
        {"id": 3, "authors": "Mueller-Klieser W & Sutherland RM", "year": 1982, "title": "Influence of convection in the growth medium on oxygen tensions in multicellular tumor spheroids", "journal": "Br J Cancer", "volume": "45", "pages": "256–264", "doi": "10.1038/bjc198241", "usage": "Microélectrodes : pO₂ centrale EMT6/Ro 66 mmHg à Ø 400 µm → 13 mmHg à Ø 1000 µm.", "category": "Calibration"},
        {"id": 4, "authors": "Casciari JJ, Sotirchos SV & Sutherland RM", "year": 1988, "title": "Glucose diffusivity in multicellular tumor spheroids", "journal": "Cancer Res", "volume": "48", "pages": "3905–3909", "doi": "10.1158/0008-5472.CAN-88-3905", "usage": "D_glucose (EMT6/Ro) = 1.1 × 10⁻⁶ cm²/s.", "category": "Calibration"},
        {"id": 5, "authors": "Grimes DR et al.", "year": 2014, "title": "A method for estimating the oxygen consumption rate in multicellular tumour spheroids", "journal": "J R Soc Interface", "volume": "11", "pages": "20131124", "doi": "10.1098/rsif.2013.1124", "usage": "Limite de diffusion de l'O₂ 233 ± 22 µm ; OCR ≈ 5.6 × 10⁻¹⁷ mol/cellule/s.", "category": "Calibration"},
        {"id": 6, "authors": "Ward JP & King JR", "year": 1997, "title": "Mathematical modelling of avascular-tumour growth", "journal": "IMA J Math Appl Med Biol", "volume": "14", "pages": "39–69", "doi": "10.1093/imammb/14.1.39", "usage": "Formalisme continu : champ de vitesse interne engendré par les variations de volume.", "category": "Modèle"},
        {"id": 7, "authors": "Bertuzzi A et al.", "year": 2010, "title": "Necrotic core in EMT6/Ro tumour spheroids: is it caused by an ATP deficit?", "journal": "J Theor Biol", "volume": "262", "pages": "142–150", "doi": "10.1016/j.jtbi.2009.09.024", "usage": "La mort cellulaire est déclenchée par un seuil sur la production d'ATP, pas par un seuil d'oxygène.", "category": "Modèle"},
        {"id": 8, "authors": "Bozic I et al.", "year": 2010, "title": "Accumulation of driver and passenger mutations during tumor progression", "journal": "PNAS", "volume": "107", "pages": "18545–18550", "doi": "10.1073/pnas.1010978107", "usage": "Socle du moteur évolutif : u = 3.4 × 10⁻⁵, s = 0.004 ± 0.0004, v ≈ 0.016.", "category": "Évolution"},
        {"id": 9, "authors": "Williams MJ et al.", "year": 2016, "title": "Identification of neutral tumor evolution across cancer types", "journal": "Nat Genet", "volume": "48", "pages": "238–244", "doi": "10.1038/ng.3489", "usage": "Loi 1/f : M(f) ∝ 1/f, R² ≥ 0.98 comme seuil. Mon modèle atteint R² = 0.988.", "category": "Évolution"},
        {"id": 10, "authors": "Alper T & Howard-Flanders P", "year": 1956, "title": "Role of oxygen in modifying the radiosensitivity of E. coli B", "journal": "Nature", "volume": "178", "pages": "978–979", "doi": "10.1038/178978a0", "usage": "Formule historique de l'OER : OER_max ≈ 3, K ≈ 3 mmHg.", "category": "Radiobiologie"},
        {"id": 11, "authors": "Emami B et al.", "year": 1991, "title": "Tolerance of normal tissue to therapeutic irradiation", "journal": "Int J Radiat Oncol Biol Phys", "volume": "21", "pages": "109–122", "doi": "10.1016/0360-3016(91)90171-Y", "usage": "Doses de tolérance TD5/5 et TD50/5 par organe. Source des paramètres LKB.", "category": "Toxicité"},
        {"id": 12, "authors": "Burman C et al.", "year": 1991, "title": "Exact calculation of lung inhomogeneity dose in radiation therapy", "journal": "Int J Radiat Oncol Biol Phys", "volume": "21", "pages": "123–135", "doi": "10.1016/0360-3016(91)90172-Z", "usage": "Modèle de Lyman pour le calcul du NTCP. Triplets (TD50, m, n) repris tels quels.", "category": "Toxicité"},
        {"id": 13, "authors": "Britten RA et al.", "year": 1996, "title": "Radiosensitivity of cervical carcinoma clonogens: evidence of the lack of correlation with radiation dose", "journal": "IJROBP", "volume": "34", "pages": "367–374", "doi": "10.1016/0360-3016(95)02076-4", "usage": "96 clones : le SF2 s'étend de 0.240 à 0.518 au sein d'une même tumeur.", "category": "Hétérogénéité"},
        {"id": 14, "authors": "Gatenby RA & Gillies RJ", "year": 2004, "title": "Why do cancers have high aerobic glycolysis?", "journal": "Nat Rev Cancer", "volume": "4", "pages": "891–899", "doi": "10.1038/nrc1478", "usage": "Justifie les deux traits adaptatifs : phénotype glycolytique et acido-résistance.", "category": "Évolution"},
        {"id": 15, "authors": "Zhang J et al.", "year": 2017, "title": "Stringent inhibition of targeted cancer cells promotes immune activation and durable control of tumor growth", "journal": "Nat Commun", "volume": "8", "pages": "1816", "doi": "10.1038/s41467-017-01968-5", "usage": "Justification clinique du piège central du jeu — frapper au maximum sélectionne la résistance.", "category": "Thérapie"},
        {"id": 16, "authors": "Lugano R, Ramachandran M & Dimberg A", "year": 2020, "title": "Tumor-associated angiogenesis", "journal": "Cell Mol Life Sci", "volume": "77", "pages": "1745–1770", "doi": "10.1007/s00018-019-03351-7", "usage": "Review of tumor vascularization mechanisms. Supports vascular.py mechanism labels.", "category": "Angiogenèse"},
        {"id": 17, "authors": "Jain RK", "year": 2005, "title": "Normalization of tumor vasculature", "journal": "Science", "volume": "307", "pages": "58–62", "doi": "10.1126/science.1104819", "usage": "Vascular normalization framework. Motivates abnormality and normalization warning.", "category": "Angiogenèse"},
        {"id": 18, "authors": "Folkman J", "year": 1971, "title": "Tumor angiogenesis: therapeutic implications", "journal": "N Engl J Med", "volume": "285", "pages": "1182–1186", "doi": "10.1056/NEJM197111182852108", "usage": "L'article fondateur. Seuil 1–2 mm³ pour le recrutement vasculaire.", "category": "Angiogenèse"},
        {"id": 19, "authors": "Carmeliet P & Jain RK", "year": 2011, "title": "Molecular mechanisms and clinical applications of angiogenesis", "journal": "Nat Rev Drug Discov", "volume": "10", "pages": "417–427", "doi": "10.1038/nrd3430", "usage": "Revue complète des mécanismes vasculaires tumoraux. Fonde vascular_scenario.", "category": "Angiogenèse"},
        {"id": 20, "authors": "Vaupel P & Mayer A", "year": 2007, "title": "Hypoxia in cancer: impact and involvement", "journal": "Cancer Metastasis Rev", "volume": "26", "pages": "21–27", "doi": "10.1007/s10555-007-9048-4", "usage": "pO₂ mesurées dans les tumeurs humaines (10–25 mmHg). Borne de vascular_oxygen.", "category": "Angiogenèse"},
        {"id": 21, "authors": "Fukumura D & Jain RK", "year": 2007, "title": "Tumor microenvironment abnormalities", "journal": "Biochim Biophys Acta", "volume": "1776", "pages": "1–10", "doi": "10.1016/j.bbcan.2007.07.003", "usage": "Revue sur la normalisation tumorale. Effet biphasique de l'anti-angiogénique.", "category": "Angiogenèse"},
        {"id": 22, "authors": "Hanahan D", "year": 2022, "title": "Hallmarks of Cancer: New Dimensions", "journal": "Cancer Discovery", "volume": "12", "pages": "31–46", "doi": "10.1158/2159-8290.CD-21-1059", "usage": "Updated Hallmarks. Conceptual context for sustaining proliferative signaling.", "category": "Général"},
        {"id": 23, "authors": "Drasdo D & Höhme S", "year": 2005, "title": "A single-cell-based model of tumor growth in vitro", "journal": "Phys Biol", "volume": "2", "pages": "133–147", "doi": "10.1088/1478-3975/2/3/001", "usage": "Modèle centre-based du module agents. Forces de répulsion/adhésion.", "category": "Modèle"},
        {"id": 24, "authors": "Anderson ARA & Chaplain MAJ", "year": 1998, "title": "Continuous and discrete mathematical models of tumor-induced angiogenesis", "journal": "Bull Math Biol", "volume": "60", "pages": "857–899", "doi": "10.1002/s11538-998-0142-4", "usage": "Modèle continu d'angiogenèse avec gradient VEGF. Fonde vascular.py.", "category": "Vascular"},
        {"id": 25, "authors": "Perfahl H et al.", "year": 2011, "title": "3D simulation of multi-scale tumour growth", "journal": "PLoS Comput Biol", "volume": "7", "pages": "e1002051", "doi": "10.1371/journal.pcbi.1002051", "usage": "Modèle multi-échelle 3D couplant champ moyen et modélisation endothéliale.", "category": "Vascular"},
        {"id": 26, "authors": "Owen MR et al.", "year": 2011, "title": "Hybrid models of tumour growth", "journal": "Bull Math Biol", "volume": "73", "pages": "1102–1126", "doi": "10.1007/s11538-010-9560-6", "usage": "Modèle hybrid cellule-continu. Valide l'approche multi-échelle.", "category": "Vascular"},
        {"id": 27, "authors": "Alfonso JCL & Berk L", "year": 2019, "title": "A unified theory of radioresistance in tumours", "journal": "Radiat Oncol", "volume": "14", "pages": "88", "doi": "10.1186/s13014-019-1288-y", "usage": "La résistance est une réduction héritable de α (Clone.radio_alpha_factor).", "category": "Radiobiologie"},
        {"id": 28, "authors": "Marks LB et al.", "year": 2010, "title": "QUANTEC Recommendations", "journal": "IJROBP", "volume": "76", "pages": "S10–S19", "doi": "10.1016/j.ijrobp.2009.07.754", "usage": "Révision moderne d'Emami. Récupération tissulaire ~25 % à 6 mois.", "category": "Toxicité"},
        {"id": 29, "authors": "Marusyk A et al.", "year": 2012, "title": "Intratumoral heterogeneity: a driving force for tumor progression", "journal": "Nat Rev Cancer", "volume": "12", "pages": "323–334", "doi": "10.1038/nrc3261", "usage": "L'hétérogénéité intratumorale est universelle, pas spécifique au modèle murin.", "category": "Évolution"},
        {"id": 30, "authors": "Hlatky L et al.", "year": 2002, "title": "The price of tumor hypoxia", "journal": "Clin Cancer Res", "volume": "8", "pages": "2894–2899", "usage": "Transition avasculaire → vasculaire comme événement stochastique.", "category": "Vascular"},
    ]
    categories = {
        "Calibration": "Données expérimentales",
        "Modèle": "Modèles de référence",
        "Radiobiologie": "Radiobiologie",
        "Évolution": "Évolution darwinienne",
        "Toxicité": "Toxicité des tissus sains",
        "Hétérogénéité": "Hétérogénéité",
        "Thérapie": "Thérapie adaptative",
        "Angiogenèse": "Angiogenèse et vascularisation",
        "Vascular": "Modélisation vasculaire",
        "Général": "Général",
    }
    return render_template("references.html", refs=refs, categories=categories, user=get_current_user())


# ═══════════════════════════════════════════════════════════════════════════
#  API — GAME ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Game instances per session (simple in-memory store).
# PythonAnywhere doit utiliser un worker pour conserver l'état de jeu en mémoire;
# le verrou évite que deux clics simultanés corrompent une partie.
_games: dict[int, TumorGame] = {}
_game_lock = threading.RLock()


def _get_game(user_id: int) -> TumorGame | None:
    """Load the game from SQLite so it survives worker changes/reloads."""
    game = _games.get(user_id)
    if game is not None:
        return game
    db = get_db()
    row = db.execute(
        "SELECT state FROM active_games WHERE user_id = ?", (user_id,)
    ).fetchone()
    db.close()
    if row is None:
        return None
    try:
        game = pickle.loads(row["state"])
    except (pickle.PickleError, EOFError, AttributeError, ImportError, ValueError):
        app.logger.exception("Impossible de restaurer la partie de l'utilisateur %s", user_id)
        return None
    _games[user_id] = game
    return game


def _save_active_game(user_id: int, scenario: str, game: TumorGame) -> None:
    state = sqlite3.Binary(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
    db = get_db()
    db.execute("""
        INSERT INTO active_games (user_id, scenario, state, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            scenario = excluded.scenario,
            state = excluded.state,
            updated_at = CURRENT_TIMESTAMP
    """, (user_id, scenario, state))
    db.commit()
    db.close()


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


@app.route("/api/game/reset", methods=["POST"])
@login_required
def api_reset():
    data = request.get_json() or {}
    scenario = data.get("scenario", "standard")
    seed = data.get("seed", int(time.time()) % 100000)
    organ = data.get("organ")
    sparing = data.get("sparing_factor")
    overrides = {}
    if organ is not None:
        overrides["organ"] = organ
    if sparing is not None:
        overrides["sparing_factor"] = float(sparing)
    try:
        cfg = make_config(scenario, seed=seed, **overrides)
    except (ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    game = TumorGame(cfg)
    with _game_lock:
        _games[session["user_id"]] = game
        _save_active_game(session["user_id"], scenario, game)
        obs = game.observe()
    return jsonify({"ok": True, "observation": _jsonable(obs),
                    "scenarios": list(SCENARIOS.keys())})


@app.route("/api/game/step", methods=["POST"])
@login_required
def api_step():
    game = _get_game(session["user_id"])
    if game is None:
        return jsonify({"ok": False, "error": "Partie non initialisée."}), 400
    data = request.get_json() or {}
    action = data.get("action", {"type": "wait"})
    # Le solveur est volontairement exécuté côté serveur, mais jamais deux fois
    # en parallèle pour le même état. L'état truth reste strictement serveur.
    try:
        with _game_lock:
            result = game.step(action)
            _save_active_game(session["user_id"], game.cfg.model_regime, game)
    except Exception as exc:
        app.logger.exception("Erreur pendant le calcul d'un tour")
        return jsonify({
            "ok": False,
            "error": "Le calcul du tour a échoué côté serveur.",
            "detail": str(exc) or type(exc).__name__,
        }), 500
    resp = {        "ok": True,
        "turn": result.turn,
        "day": round(result.day, 2),
        "observation": _jsonable(result.observation),
        "reward": result.reward,
        "done": result.done,
        "outcome": result.outcome,
        "info": _jsonable(result.info),
    }
    # Save to leaderboard if game over
    if result.done and "user_id" in session:
        _save_game_result(session["user_id"], game, result)
    return jsonify(resp)


@app.route("/api/game/ai", methods=["POST"])
@login_required
def api_ai():
    game = _get_game(session["user_id"])
    if game is None:
        return jsonify({"ok": False, "error": "Partie non initialisée."}), 400
    with _game_lock:
        action = recommend_action(game)
    return jsonify({"ok": True, "action": _jsonable(action)})


@app.route("/api/game/state", methods=["GET"])
@login_required
def api_state():
    game = _get_game(session["user_id"])
    if game is None:
        return jsonify({"ok": False, "error": "Partie non initialisée."}), 400
    return jsonify({"ok": True,
                    "observation": _jsonable(game.observe())})


@app.route("/api/game/actions", methods=["GET"])
@login_required
def api_actions():
    game = _get_game(session["user_id"])
    if game is None:
        return jsonify({"ok": False, "error": "Partie non initialisée."}), 400
    from tumeur.game import GLUCOSE_LEVELS, OXYGEN_LEVELS
    return jsonify({"ok": True, "actions": [
        {"type": "wait"},
        {"type": "irradiate", "dose_gy": [0.0, game.cfg.dose_max_per_turn]},
        {"type": "set_medium", "glucose_mM": list(GLUCOSE_LEVELS),
         "oxygen_mM": list(OXYGEN_LEVELS)},
        {"type": "biopsy"},
        {"type": "po2_imaging"},
        {"type": "antiangiogenic", "intensity": [0.0, 1.0],
         "available_in": ["vascular_human"]},
    ]})


# ═══════════════════════════════════════════════════════════════════════════
#  API — LEADERBOARD (SSE)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/leaderboard")
def api_leaderboard():
    db = get_db()
    rows = db.execute("""
        SELECT g.*, u.username
        FROM games g JOIN users u ON g.user_id = u.id
        WHERE g.scenario != 'tutoriel'
        ORDER BY g.score DESC LIMIT 50
    """).fetchall()
    db.close()
    return jsonify([dict(r) for r in rows])


# ═══════════════════════════════════════════════════════════════════════════
#  API — VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

# Cache for validation tests (computed once, ~3 min)
_tests_cache = None
_tests_status = "idle"  # idle | running | done

def _run_tests():
    global _tests_cache, _tests_status
    try:
        _tests_status = "running"
        from tumeur.validation import run_all
        checks = run_all(verbose=False)
        results = []
        for c in checks:
            results.append({
                "name": c.name,
                "category": c.category,
                "value": round(c.value, 6) if np.isfinite(c.value) else None,
                "criterion": round(c.criterion, 6),
                "passed": bool(c.passed),
                "unit": c.unit,
                "detail": c.detail,
            })
        n_ok = sum(1 for r in results if r["passed"])
        _tests_cache = {"total": len(results), "passed": n_ok, "checks": results}
        _tests_status = "done"
    except Exception as exc:
        # Ne jamais laisser l'interface en « running » indéfiniment.
        _tests_cache = {"total": 0, "passed": 0, "checks": [],
                        "status": "error", "error": str(exc)}
        _tests_status = "error"

import threading as _threading

@app.route("/api/tests/run")
def api_tests_run():
    """Returns cached test results, or kicks off computation if not ready."""
    global _tests_status
    if _tests_cache is not None:
        return jsonify(_tests_cache)
    if _tests_status == "idle":
        _threading.Thread(target=_run_tests, daemon=True).start()
    return jsonify({"status": "running", "message": "Les tests tournent (~3 min)... rechargez dans un instant."}), 202


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save_game_result(user_id: int, game: TumorGame, result):
    try:
        truth = result.truth
        db = get_db()
        db.execute("""
            INSERT INTO games (user_id, scenario, score, outcome, turns, dose_gy,
                               toxicity, diameter_end, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, game.cfg.model_regime,
              round(truth.get("score", 0), 1),
              result.outcome,
              result.turn,
              round(truth.get("dose_delivered_gy", 0), 1),
              round(truth.get("toxicity_score", 0), 1),
              round(truth.get("diameter_um", 0), 0),
              json.dumps(_jsonable(game.history), ensure_ascii=False)))
        db.commit()
        db.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("TUMORLAB_DEBUG", "0") == "1")
