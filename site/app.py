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
import random
import sys
import time
import threading
import sqlite3
from functools import wraps
from hashlib import pbkdf2_hmac
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, Response,
)
from itsdangerous import URLSafeSerializer, BadSignature
from werkzeug.exceptions import HTTPException

# --- Python engine & i18n path --------------------------------------------
SITE_DIR = str(Path(__file__).resolve().parent)
ENGINE_DIR = str(Path(__file__).resolve().parent.parent)
if SITE_DIR not in sys.path:
    sys.path.insert(0, SITE_DIR)
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

from tumeur.game import TumorGame, GameConfig, SCENARIOS, make_config
from tumeur.ai import recommend_action
from i18n import SCENARIOS_I18N, FIGURES_I18N, CHECK_NAMES_EN
import numpy as np

# --- App setup -------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("TUMORLAB_SECRET_KEY", "dev-only-change-me")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("TUMORLAB_COOKIE_SECURE", "0") == "1",
)

DB_PATH = os.path.join(SITE_DIR, "tumorlab.db")
VALIDATION_JSON_PATH = os.path.join(ENGINE_DIR, "resultats", "validation_checks.json")
AI_METRICS_JSON_PATH = os.path.join(ENGINE_DIR, "resultats", "ai_training_metrics.json")


def _token_serializer() -> URLSafeSerializer:
    return URLSafeSerializer(app.secret_key, salt="tumorlab-player-v1")


def make_player_token(user_id: int) -> str:
    return _token_serializer().dumps({"uid": int(user_id)})


def verify_player_token(token: str | None) -> int | None:
    if not token:
        return None
    try:
        payload = _token_serializer().loads(token)
        uid = int(payload.get("uid"))
        return uid if uid > 0 else None
    except (BadSignature, ValueError, TypeError, AttributeError):
        return None


def get_lang() -> str:
    q = request.args.get("lang")
    if q in ("en", "fr"):
        session["lang"] = q
        return q
    hdr = request.headers.get("X-Lang")
    if hdr in ("en", "fr"):
        return hdr
    lang = session.get("lang", "en")
    return lang if lang in ("en", "fr") else "en"


def _with_lang_param(target_url: str, lang_code: str) -> str:
    parsed = urlparse(target_url)
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["lang"] = lang_code
    return urlunparse(parsed._replace(query=urlencode(params)))


@app.context_processor
def inject_globals():
    lang = get_lang()
    return {
        "lang": lang,
        "is_en": lang == "en",
        "is_guest": bool(session.get("is_guest", False)),
    }


@app.route("/set-lang/<lang_code>")
def set_lang(lang_code: str):
    chosen = "fr" if lang_code == "fr" else "en"
    session["lang"] = chosen
    nxt = request.args.get("next") or request.referrer or url_for("home")
    return redirect(_with_lang_param(nxt, chosen))


@app.errorhandler(Exception)
def handle_api_error(error):
    """Keep API failures JSON so the browser can show the real HTTP error."""
    if not request.path.startswith("/api/"):
        if isinstance(error, HTTPException):
            return error
        app.logger.exception("Erreur page %s", request.path)
        return "Internal server error / Une erreur interne est survenue.", 500
    if isinstance(error, HTTPException):
        status = error.code or 500
        message = error.description
    else:
        status = 500
        message = "Internal calculation error." if get_lang() == "en" else "Erreur interne pendant le calcul."
        app.logger.exception("Erreur API %s", request.path)
    return jsonify({"ok": False, "error": message}), status


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
#  AUTH & STATELESS TOKEN / GUEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_user_by_id(uid: int):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    db.close()
    return user


def get_current_user():
    if "user_id" in session:
        user = _get_user_by_id(int(session["user_id"]))
        if user is not None:
            return user
        session.pop("user_id", None)
    tok = request.args.get("token") or request.headers.get("X-Player-Token")
    uid = verify_player_token(tok)
    if uid is not None:
        return _get_user_by_id(uid)
    return None


def resolve_api_user_id(json_body: dict | None = None) -> int | None:
    """Resolve user_id via signed token first (iframe-safe), then Flask session."""
    tok = (
        request.headers.get("X-Player-Token")
        or (json_body.get("player_token") if isinstance(json_body, dict) else None)
        or request.args.get("token")
    )
    uid = verify_player_token(tok)
    if uid is not None and _get_user_by_id(uid) is not None:
        return uid
    if "user_id" in session:
        s_uid = int(session["user_id"])
        if _get_user_by_id(s_uid) is not None:
            return s_uid
    return None


def ensure_user_or_guest(existing_uid: int | None = None):
    """Return logged-in/token user, or create a lightweight guest session in 1 click."""
    if existing_uid is not None:
        u = _get_user_by_id(existing_uid)
        if u is not None:
            return u
    user = get_current_user()
    if user is not None:
        return user
    db = get_db()
    for _ in range(15):
        guest_name = f"Guest_{random.randint(1000, 9999)}"
        existing = db.execute("SELECT id FROM users WHERE username = ?", (guest_name,)).fetchone()
        if not existing:
            pw_hash, salt = hash_password(os.urandom(8).hex())
            db.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                       (guest_name, pw_hash, salt))
            db.commit()
            user = db.execute("SELECT * FROM users WHERE username = ?", (guest_name,)).fetchone()
            db.close()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["is_guest"] = True
            return user
    db.close()
    return None


def api_auth_required(f):
    """Ensure API endpoints NEVER return HTML 302 redirects."""
    @wraps(f)
    def decorated(*args, **kwargs):
        data = request.get_json(silent=True) if request.is_json else None
        uid = resolve_api_user_id(data)
        if uid is None:
            msg = ("Game session missing — please start a new game."
                   if get_lang() == "en"
                   else "Session introuvable — veuillez relancer une partie.")
            return jsonify({"ok": False, "error": msg}), 401
        request.api_user_id = uid
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES — PAGES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    lang = get_lang()
    user = get_current_user()
    return render_template(
        "home.html",
        user=user,
        scenarios=SCENARIOS_I18N[lang],
        figures=FIGURES_I18N[lang],
    )


@app.route("/guest")
def guest_start():
    """1-click guest session so anyone can try the app without registering."""
    ensure_user_or_guest()
    scenario = request.args.get("scenario")
    lang = get_lang()
    if scenario and scenario in SCENARIOS:
        return redirect(url_for("play", scenario=scenario, lang=lang))
    return redirect(url_for("home", lang=lang))


@app.route("/register", methods=["GET", "POST"])
def register():
    lang = get_lang()
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 2:
            error = ("Username must be at least 2 characters." if lang == "en"
                     else "Le nom d'utilisateur doit faire au moins 2 caractères.")
        elif len(password) < 4:
            error = ("Password must be at least 4 characters." if lang == "en"
                     else "Le mot de passe doit faire au moins 4 caractères.")
        else:
            db = get_db()
            existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                error = ("This username is already taken." if lang == "en"
                         else "Ce nom d'utilisateur est déjà pris.")
            else:
                pw_hash, salt = hash_password(password)
                db.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                           (username, pw_hash, salt))
                db.commit()
                user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                db.close()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["is_guest"] = False
                return redirect(url_for("home", lang=lang))
            db.close()
    return render_template("register.html", error=error, user=get_current_user())


@app.route("/login", methods=["GET", "POST"])
def login():
    lang = get_lang()
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
            session["is_guest"] = False
            return redirect(url_for("home", lang=lang))
        error = ("Incorrect username or password." if lang == "en"
                 else "Nom d'utilisateur ou mot de passe incorrect.")
    return render_template("login.html", error=error, user=get_current_user())


@app.route("/logout")
def logout():
    keep_lang = get_lang()
    session.clear()
    session["lang"] = keep_lang
    return redirect(url_for("home", lang=keep_lang))


@app.route("/play/<scenario>")
def play(scenario):
    """Automatically let new visitors play in 1 click via Guest session."""
    lang = get_lang()
    if scenario not in SCENARIOS:
        return redirect(url_for("home", lang=lang))
    user = ensure_user_or_guest()
    player_token = make_player_token(user["id"]) if user else ""
    scenario_meta = SCENARIOS_I18N[lang].get(scenario, {"name": scenario, "desc": "", "badge": ""})
    return render_template(
        "game.html",
        scenario=scenario,
        scenario_meta=scenario_meta,
        user=user,
        player_token=player_token,
    )


@app.route("/api/game/export")
@api_auth_required
def api_game_export():
    """Export the current or last active game for analysis."""
    game = _get_game(request.api_user_id)
    if game is None:
        return jsonify({"ok": False, "error": "No active game / Aucune partie disponible."}), 404
    fmt = request.args.get("format", "json").lower()
    scenario_name = getattr(game, "scenario_name", game.cfg.model_regime)
    payload = {
        "scenario": scenario_name,
        "model_regime": game.cfg.model_regime,
        "config": _jsonable(vars(game.cfg)),
        "observation": _jsonable(game.observe()),
        "history": _jsonable(game.history),
    }
    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["turn", "day", "action", "diameter_um", "toxicity", "score", "outcome"])
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
            headers={"Content-Disposition": "attachment; filename=tumor-lab-game.csv"},
        )
    if fmt != "json":
        return jsonify({"ok": False, "error": "Unknown format / Format inconnu."}), 400
    return Response(
        json.dumps(payload, ensure_ascii=False, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=tumor-lab-game.json"},
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
    ai_meta = None
    if os.path.exists(AI_METRICS_JSON_PATH):
        try:
            with open(AI_METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                ai_meta = json.load(f).get("metadata")
        except Exception:
            ai_meta = None
    return render_template("tests.html", user=get_current_user(), ai_meta=ai_meta)


@app.route("/references")
def references():
    refs = [
        {"id": 1, "authors": "Freyer JP & Sutherland RM", "year": 1986, "title": "Regulation of growth saturation and development of necrosis in EMT6/Ro multicellular spheroids by the glucose and oxygen supply", "journal": "Cancer Res", "volume": "46", "pages": "3504–3512", "doi": "10.1111/j.1365-2184.1992.tb01433.x", "usage": "Primary validation source: 8 nutrient media, saturation diameters, viable rim thicknesses, necrosis onset.", "usage_fr": "Source principale de validation. Matrice de 8 milieux, diamètres de saturation, couronnes viables, diamètres d'apparition de la nécrose.", "category": "Calibration"},
        {"id": 2, "authors": "Freyer JP & Sutherland RM", "year": 1985, "title": "A reduction in the in situ rates of oxygen and glucose consumption of cells in EMT6/Ro spheroids during growth", "journal": "J Cell Physiol", "volume": "124", "pages": "516–524", "doi": "10.1002/jcp.1041240323", "usage": "Per-cell O2 and glucose uptake drops 2.5x at 1300 µm diameter.", "usage_fr": "Consommation d'O2 et de glucose par cellule divisée par 2.5 à 1300 µm.", "category": "Calibration"},
        {"id": 3, "authors": "Mueller-Klieser W & Sutherland RM", "year": 1982, "title": "Influence of convection in the growth medium on oxygen tensions in multicellular tumor spheroids", "journal": "Br J Cancer", "volume": "45", "pages": "256–264", "doi": "10.1038/bjc198241", "usage": "Microelectrode pO2 scans: central pO2 drops from 66 mmHg (400 µm) to 13 mmHg (1000 µm).", "usage_fr": "Microélectrodes : pO2 centrale EMT6/Ro 66 mmHg à 400 µm vers 13 mmHg à 1000 µm.", "category": "Calibration"},
        {"id": 4, "authors": "Casciari JJ, Sotirchos SV & Sutherland RM", "year": 1988, "title": "Glucose diffusivity in multicellular tumor spheroids", "journal": "Cancer Res", "volume": "48", "pages": "3905–3909", "doi": "10.1158/0008-5472.CAN-88-3905", "usage": "Measured glucose diffusivity in EMT6/Ro = 1.1 x 10^-6 cm²/s.", "usage_fr": "D_glucose (EMT6/Ro) = 1.1 x 10^-6 cm²/s.", "category": "Calibration"},
        {"id": 5, "authors": "Grimes DR et al.", "year": 2014, "title": "A method for estimating the oxygen consumption rate in multicellular tumour spheroids", "journal": "J R Soc Interface", "volume": "11", "pages": "20131124", "doi": "10.1098/rsif.2013.1124", "usage": "O2 diffusion limit 233 ± 22 µm; OCR ≈ 5.6 x 10^-17 mol/cell/s.", "usage_fr": "Limite de diffusion de l'O2 233 ± 22 µm ; OCR ≈ 5.6 x 10^-17 mol/cellule/s.", "category": "Calibration"},
        {"id": 6, "authors": "Ward JP & King JR", "year": 1997, "title": "Mathematical modelling of avascular-tumour growth", "journal": "IMA J Math Appl Med Biol", "volume": "14", "pages": "39–69", "doi": "10.1093/imammb/14.1.39", "usage": "Continuous free-boundary formalism with internal velocity field.", "usage_fr": "Formalisme continu : champ de vitesse interne engendré par les variations de volume.", "category": "Modèle"},
        {"id": 7, "authors": "Bertuzzi A et al.", "year": 2010, "title": "Necrotic core in EMT6/Ro tumour spheroids: is it caused by an ATP deficit?", "journal": "J Theor Biol", "volume": "262", "pages": "142–150", "doi": "10.1016/j.jtbi.2009.09.024", "usage": "Cell death is triggered by an ATP production threshold rather than O2 alone.", "usage_fr": "La mort cellulaire est déclenchée par un seuil sur la production d'ATP, pas par un seuil d'oxygène.", "category": "Modèle"},
        {"id": 8, "authors": "Bozic I et al.", "year": 2010, "title": "Accumulation of driver and passenger mutations during tumor progression", "journal": "PNAS", "volume": "107", "pages": "18545–18550", "doi": "10.1073/pnas.1010978107", "usage": "Evolutionary parameters: u = 3.4 x 10^-5, s = 0.004 ± 0.0004, v ≈ 0.016.", "usage_fr": "Socle du moteur évolutif : u = 3.4 x 10^-5, s = 0.004 ± 0.0004, v ≈ 0.016.", "category": "Évolution"},
        {"id": 9, "authors": "Williams MJ et al.", "year": 2016, "title": "Identification of neutral tumor evolution across cancer types", "journal": "Nat Genet", "volume": "48", "pages": "238–244", "doi": "10.1038/ng.3489", "usage": "Neutral 1/f power law (R² ≥ 0.98 threshold; our engine achieves R² = 0.988).", "usage_fr": "Loi 1/f : M(f) ∝ 1/f, R² ≥ 0.98 comme seuil. Mon modèle atteint R² = 0.988.", "category": "Évolution"},
        {"id": 10, "authors": "Alper T & Howard-Flanders P", "year": 1956, "title": "Role of oxygen in modifying the radiosensitivity of E. coli B", "journal": "Nature", "volume": "178", "pages": "978–979", "doi": "10.1038/178978a0", "usage": "Classic OER equation for hypoxic radioresistance (OER_max ≈ 3, K ≈ 3 mmHg).", "usage_fr": "Formule historique de l'OER : OER_max ≈ 3, K ≈ 3 mmHg.", "category": "Radiobiologie"},
        {"id": 11, "authors": "Emami B et al.", "year": 1991, "title": "Tolerance of normal tissue to therapeutic irradiation", "journal": "Int J Radiat Oncol Biol Phys", "volume": "21", "pages": "109–122", "doi": "10.1016/0360-3016(91)90171-Y", "usage": "Healthy organ tolerance doses (TD5/5 and TD50/5) per organ.", "usage_fr": "Doses de tolérance TD5/5 et TD50/5 par organe. Source des paramètres LKB.", "category": "Toxicité"},
        {"id": 12, "authors": "Burman C et al.", "year": 1991, "title": "Fitting of normal tissue tolerance data to an analytic function", "journal": "Int J Radiat Oncol Biol Phys", "volume": "21", "pages": "123–135", "doi": "10.1016/0360-3016(91)90172-Z", "usage": "Lyman-Kutcher-Burman NTCP toxicity model parameters (TD50, m, n).", "usage_fr": "Modèle de Lyman pour le calcul du NTCP. Triplets (TD50, m, n) repris tels quels.", "category": "Toxicité"},
        {"id": 13, "authors": "Britten RA et al.", "year": 1996, "title": "Radiosensitivity of cervical carcinoma clonogens: evidence of the lack of correlation with radiation dose", "journal": "IJROBP", "volume": "34", "pages": "367–374", "doi": "10.1016/0360-3016(95)02076-4", "usage": "96 clones showing SF2 heterogeneity (0.240–0.518) inside a single tumor — grounds biopsy logic.", "usage_fr": "96 clones : le SF2 s'étend de 0.240 à 0.518 au sein d'une même tumeur.", "category": "Hétérogénéité"},
        {"id": 14, "authors": "Gatenby RA & Gillies RJ", "year": 2004, "title": "Why do cancers have high aerobic glycolysis?", "journal": "Nat Rev Cancer", "volume": "4", "pages": "891–899", "doi": "10.1038/nrc1478", "usage": "Glycolytic switch and acid resistance in hypoxic microenvironments.", "usage_fr": "Justifie les deux traits adaptatifs : phénotype glycolytique et acido-résistance.", "category": "Évolution"},
        {"id": 15, "authors": "Zhang J et al.", "year": 2017, "title": "Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer", "journal": "Nat Commun", "volume": "8", "pages": "1816", "doi": "10.1038/s41467-017-01968-5", "usage": "Clinical basis for adaptive therapy: maximum-dose blasts select resistant clones.", "usage_fr": "Justification clinique du piège central du jeu — frapper au maximum sélectionne la résistance.", "category": "Thérapie"},
        {"id": 16, "authors": "Lugano R, Ramachandran M & Dimberg A", "year": 2020, "title": "Tumor-associated angiogenesis", "journal": "Cell Mol Life Sci", "volume": "77", "pages": "1745–1770", "doi": "10.1007/s00018-019-03351-7", "usage": "Review of tumor vascularization mechanisms supporting vascular.py.", "usage_fr": "Revue des mécanismes de vascularisation tumorale supportant vascular.py.", "category": "Angiogenèse"},
        {"id": 17, "authors": "Jain RK", "year": 2005, "title": "Normalization of tumor vasculature", "journal": "Science", "volume": "307", "pages": "58–62", "doi": "10.1126/science.1104819", "usage": "Vascular normalization window and abnormal tumor perfusion.", "usage_fr": "Cadre de normalisation vasculaire et perfusion anormale.", "category": "Angiogenèse"},
        {"id": 18, "authors": "Folkman J", "year": 1971, "title": "Tumor angiogenesis: therapeutic implications", "journal": "N Engl J Med", "volume": "285", "pages": "1182–1186", "doi": "10.1056/NEJM197111182852108", "usage": "Foundational angiogenesis paper: ~1–2 mm³ diffusion limit triggers vessel recruitment.", "usage_fr": "L'article fondateur. Seuil 1–2 mm³ pour le recrutement vasculaire.", "category": "Angiogenèse"},
        {"id": 19, "authors": "Carmeliet P & Jain RK", "year": 2011, "title": "Molecular mechanisms and clinical applications of angiogenesis", "journal": "Nat Rev Drug Discov", "volume": "10", "pages": "417–427", "doi": "10.1038/nrd3430", "usage": "Comprehensive review of vascular tumor mechanisms.", "usage_fr": "Revue complète des mécanismes vasculaires tumoraux. Fonde vascular_scenario.", "category": "Angiogenèse"},
        {"id": 20, "authors": "Vaupel P & Mayer A", "year": 2007, "title": "Hypoxia in cancer: significance and impact on clinical outcome", "journal": "Cancer Metastasis Rev", "volume": "26", "pages": "225–239", "doi": "10.1007/s10555-007-9055-1", "usage": "Measured pO2 in human tumors (10–25 mmHg) and diagnostic prognostic value.", "usage_fr": "pO2 mesurées dans les tumeurs humaines (10–25 mmHg). Borne de vascular_oxygen.", "category": "Angiogenèse"},
        {"id": 21, "authors": "Alfonso JCL & Berk L", "year": 2019, "title": "Modeling the effect of intratumoral heterogeneity of radiosensitivity on tumor response over the course of fractionated radiation therapy", "journal": "Radiat Oncol", "volume": "14", "pages": "88", "doi": "10.1186/s13014-019-1288-y", "usage": "Analytical distribution of alpha radiosensitivity under fractionated radiation.", "usage_fr": "La résistance est une réduction héritable de alpha (ResistanceDistribution).", "category": "Radiobiologie"},
        {"id": 22, "authors": "Marks LB et al.", "year": 2010, "title": "Use of normal tissue complication probability models in the clinic (QUANTEC)", "journal": "IJROBP", "volume": "76", "pages": "S10–S19", "doi": "10.1016/j.ijrobp.2009.07.754", "usage": "Modern clinical QUANTEC normal-tissue dose-response guidelines.", "usage_fr": "Révision moderne d'Emami. Récupération tissulaire ~25 % à 6 mois.", "category": "Toxicité"},
        {"id": 23, "authors": "Drasdo D & Höhme S", "year": 2005, "title": "A single-cell-based model of tumor growth in vitro", "journal": "Phys Biol", "volume": "2", "pages": "133–147", "doi": "10.1088/1478-3975/2/3/001", "usage": "Center-based 3D agent mechanics used for cross-validation.", "usage_fr": "Modèle centre-based du module agents. Forces de répulsion/adhésion.", "category": "Modèle"},
        {"id": 24, "authors": "Hanahan D", "year": 2022, "title": "Hallmarks of Cancer: New Dimensions", "journal": "Cancer Discovery", "volume": "12", "pages": "31–46", "doi": "10.1158/2159-8290.CD-21-1059", "usage": "Conceptual oncology framework for proliferative signaling and microenvironment.", "usage_fr": "Cadre général des Hallmarks of Cancer.", "category": "Général"},
    ]
    lang = get_lang()
    if lang == "en":
        categories = {
            "Calibration": "Experimental Calibration Data",
            "Modèle": "Mathematical & Mechanistic Models",
            "Radiobiologie": "Radiobiology & Hypoxia",
            "Hétérogénéité": "Diagnostic Heterogeneity & Biopsy",
            "Évolution": "Darwinian Clonal Evolution",
            "Toxicité": "Healthy Organ Toxicity (NTCP)",
            "Thérapie": "Adaptive Therapy",
            "Angiogenèse": "Angiogenesis & Tumor Vasculature",
            "Général": "General Oncology",
        }
    else:
        categories = {
            "Calibration": "Données expérimentales",
            "Modèle": "Modèles de référence",
            "Radiobiologie": "Radiobiologie",
            "Hétérogénéité": "Hétérogénéité et biopsie",
            "Évolution": "Évolution darwinienne",
            "Toxicité": "Toxicité des tissus sains",
            "Thérapie": "Thérapie adaptative",
            "Angiogenèse": "Angiogenèse et vascularisation",
            "Général": "Général",
        }
    return render_template("references.html", refs=refs, categories=categories, user=get_current_user())


# ═══════════════════════════════════════════════════════════════════════════
#  API — GAME ENGINE
# ═══════════════════════════════════════════════════════════════════════════

_games: dict[int, TumorGame] = {}
_game_lock = threading.RLock()


def _get_game(user_id: int) -> TumorGame | None:
    """Load the game from SQLite so it survives worker changes/reloads."""
    game = _games.get(user_id)
    if game is not None:
        return game
    db = get_db()
    row = db.execute(
        "SELECT scenario, state FROM active_games WHERE user_id = ?", (user_id,)
    ).fetchone()
    db.close()
    if row is None:
        return None
    try:
        game = pickle.loads(row["state"])
        if not hasattr(game, "scenario_name"):
            game.scenario_name = row["scenario"]
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
def api_reset():
    data = request.get_json() or {}
    existing_uid = resolve_api_user_id(data)
    user = ensure_user_or_guest(existing_uid=existing_uid)
    uid = int(user["id"])
    player_token = make_player_token(uid)

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
    game.scenario_name = scenario
    with _game_lock:
        _games[uid] = game
        _save_active_game(uid, scenario, game)
        obs = game.observe()
    return jsonify({
        "ok": True,
        "player_token": player_token,
        "scenario": scenario,
        "observation": _jsonable(obs),
        "config": {
            "max_days": cfg.max_days,
            "dose_budget_gy": cfg.dose_budget_gy,
            "dose_max_per_turn": cfg.dose_max_per_turn,
            "control_diameter_um": cfg.control_diameter_um,
            "lose_diameter_um": cfg.lose_diameter_um,
            "lose_toxicity": cfg.lose_toxicity,
            "warn_toxicity": cfg.warn_toxicity,
            "organ": cfg.organ,
            "vascular_enabled": cfg.vascular_enabled,
        },
        "scenarios": list(SCENARIOS.keys()),
    })


@app.route("/api/game/step", methods=["POST"])
@api_auth_required
def api_step():
    uid = request.api_user_id
    game = _get_game(uid)
    if game is None:
        return jsonify({"ok": False, "error": "Game not initialized / Partie non initialisée."}), 400
    data = request.get_json() or {}
    action = data.get("action", {"type": "wait"})
    scenario_name = getattr(game, "scenario_name", game.cfg.model_regime)
    try:
        with _game_lock:
            result = game.step(action)
            _save_active_game(uid, scenario_name, game)
    except Exception as exc:
        app.logger.exception("Erreur pendant le calcul d'un tour")
        return jsonify({
            "ok": False,
            "error": "Turn calculation failed on server.",
            "detail": str(exc) or type(exc).__name__,
        }), 500
    # Never expose server-side `truth` to the browser; only public observation.
    public_info = {"action": result.info.get("action", action.get("type", "wait"))}
    if "biopsy" in result.info:
        public_info["biopsy"] = _jsonable(result.info["biopsy"])
    if "po2_imaging" in result.info:
        public_info["po2_imaging"] = _jsonable(result.info["po2_imaging"])
    if "radiation" in result.info:
        public_info["dose_applied_gy"] = float(result.info["radiation"].get("dose_gy", 0.0))
    if "antiangiogenic" in result.info:
        public_info["antiangiogenic"] = _jsonable(result.info["antiangiogenic"])

    resp = {
        "ok": True,
        "turn": result.turn,
        "day": round(result.day, 2),
        "observation": _jsonable(result.observation),
        "reward": result.reward,
        "done": result.done,
        "outcome": result.outcome,
        "info": public_info,
    }
    if result.done:
        _save_game_result(uid, game, result)
    return jsonify(resp)


@app.route("/api/game/ai", methods=["POST"])
@api_auth_required
def api_ai():
    game = _get_game(request.api_user_id)
    if game is None:
        return jsonify({"ok": False, "error": "Game not initialized / Partie non initialisée."}), 400
    with _game_lock:
        action = recommend_action(game)
    return jsonify({"ok": True, "action": _jsonable(action)})


@app.route("/api/game/state", methods=["GET"])
@api_auth_required
def api_state():
    game = _get_game(request.api_user_id)
    if game is None:
        return jsonify({"ok": False, "error": "Game not initialized / Partie non initialisée."}), 400
    return jsonify({"ok": True, "observation": _jsonable(game.observe())})


@app.route("/api/game/actions", methods=["GET"])
@api_auth_required
def api_actions():
    game = _get_game(request.api_user_id)
    if game is None:
        return jsonify({"ok": False, "error": "Game not initialized / Partie non initialisée."}), 400
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
#  API — LEADERBOARD
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
#  API — VALIDATION TESTS (Instant cached results + optional live recompute)
# ═══════════════════════════════════════════════════════════════════════════

_tests_cache = None
_tests_status = "idle"


def _load_saved_validation():
    if os.path.exists(VALIDATION_JSON_PATH):
        try:
            with open(VALIDATION_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _localize_checks(payload: dict, lang: str) -> dict:
    if not payload or "checks" not in payload or lang != "en":
        return payload
    out = dict(payload)
    translated = []
    for c in payload["checks"]:
        item = dict(c)
        item["name"] = CHECK_NAMES_EN.get(c["name"], c["name"])
        translated.append(item)
    out["checks"] = translated
    return out


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
                "value": round(float(c.value), 6) if np.isfinite(c.value) else None,
                "criterion": round(float(c.criterion), 6),
                "passed": bool(c.passed),
                "unit": c.unit,
                "detail": c.detail,
            })
        n_ok = sum(1 for r in results if r["passed"])
        _tests_cache = {"total": len(results), "passed": n_ok, "checks": results, "cached": False}
        _tests_status = "done"
    except Exception as exc:
        _tests_cache = {"total": 0, "passed": 0, "checks": [],
                        "status": "error", "error": str(exc)}
        _tests_status = "error"


@app.route("/api/tests/run")
def api_tests_run():
    """Returns pre-computed validation checks immediately, or recomputes live if requested."""
    global _tests_cache, _tests_status
    lang = get_lang()
    recompute = request.args.get("recompute") == "1"

    if recompute and _tests_status != "running":
        _tests_cache = None
        threading.Thread(target=_run_tests, daemon=True).start()
        return jsonify({"status": "running"}), 202

    if _tests_status == "running":
        return jsonify({"status": "running"}), 202

    if _tests_cache is None:
        saved = _load_saved_validation()
        if saved is not None:
            _tests_cache = saved
            _tests_status = "done"

    if _tests_cache is not None:
        return jsonify(_localize_checks(_tests_cache, lang))

    threading.Thread(target=_run_tests, daemon=True).start()
    return jsonify({"status": "running"}), 202


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save_game_result(user_id: int, game: TumorGame, result):
    try:
        truth = result.truth
        scenario_name = getattr(game, "scenario_name", game.cfg.model_regime)
        db = get_db()
        db.execute("""
            INSERT INTO games (user_id, scenario, score, outcome, turns, dose_gy,
                               toxicity, diameter_end, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, scenario_name,
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
