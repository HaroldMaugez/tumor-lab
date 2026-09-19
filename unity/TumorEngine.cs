// TumorEngine.cs — pont Unity <-> moteur Python.
//
// À placer dans Assets/Scripts/. Dépendance : Newtonsoft Json (Package Manager :
// "com.unity.nuget.newtonsoft-json"). Unity 2020.3+.
//
// POINT CRITIQUE DE CONCEPTION
// ----------------------------
// Un tour de simulation coûte ~1 s de CPU (mesuré : médiane 1.15 s, max 1.7 s).
// Appeler Step() depuis Update() gèlerait l'image pendant une seconde entière.
// Toutes les requêtes partent donc sur un thread de fond, et les réponses sont
// renvoyées au thread principal via une file. Le jeu reste à 60 fps pendant que
// la tumeur se calcule : on affiche une animation d'attente ("le temps passe...").
//
// Le processus Python est lancé en enfant : rien à installer côté joueur hormis
// Python 3 + numpy + scipy, et il s'arrête tout seul avec le jeu.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Debug = UnityEngine.Debug;

public class TumorEngine : MonoBehaviour
{
    [Header("Chemins")]
    [Tooltip("Exécutable Python. Laisser 'python3' si présent dans le PATH.")]
    public string pythonExecutable = "python3";

    [Tooltip("Chemin de server.py, relatif à StreamingAssets.")]
    public string serverScript = "tumorsim/server.py";

    [Header("Partie")]
    public int seed = 1;
    public int maxDays = 180;
    public float doseBudgetGy = 60f;
    public string initialScenario = "vascular_human";

    // --- état exposé au reste du jeu (thread principal uniquement) ---
    public bool IsReady { get; private set; }
    public bool IsBusy { get; private set; }
    public JObject LastObservation { get; private set; }
    public JObject LastTruth { get; private set; }
    public string Outcome { get; private set; } = "en_cours";
    public bool Done { get; private set; }

    public event Action<JObject> OnObservation;   // état joueur mis à jour
    public event Action<JObject> OnGameReset;
    public event Action<string> OnGameOver;
    public event Action<string> OnError;
    public event Action<JObject> OnAiRecommendation;

    Process _proc;
    StreamWriter _stdin;
    StreamReader _stdout;
    readonly object _ioLock = new object();
    readonly ConcurrentQueue<Action> _mainThread = new ConcurrentQueue<Action>();

    // ------------------------------------------------------------------
    void Awake()
    {
        DontDestroyOnLoad(gameObject);
        StartEngine();
    }

    void Update()
    {
        // exécute sur le thread principal ce que le thread de fond a déposé
        while (_mainThread.TryDequeue(out var act))
        {
            try { act(); }
            catch (Exception e) { Debug.LogException(e); }
        }
    }

    void OnDestroy() { StopEngine(); }
    void OnApplicationQuit() { StopEngine(); }

    // ------------------------------------------------------------------
    void StartEngine()
    {
        string path = Path.Combine(Application.streamingAssetsPath, serverScript);
        if (!File.Exists(path))
        {
            Fail($"server.py introuvable : {path}");
            return;
        }

        var psi = new ProcessStartInfo
        {
            FileName = pythonExecutable,
            Arguments = $"-u \"{path}\"",          // -u : sortie non tamponnée, indispensable
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(path)
        };

        try
        {
            _proc = Process.Start(psi);
            _stdin = _proc.StandardInput;
            _stdout = _proc.StandardOutput;
            _proc.ErrorDataReceived += (s, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data)) Debug.LogWarning("[moteur] " + e.Data);
            };
            _proc.BeginErrorReadLine();
        }
        catch (Exception e)
        {
            Fail("Impossible de lancer Python : " + e.Message);
            return;
        }

        // handshake puis reset, en tâche de fond
        Task.Run(() =>
        {
            var pong = Request(new JObject { ["cmd"] = "ping" });
            if (pong == null || !(bool)(pong["ok"] ?? false))
            {
                _mainThread.Enqueue(() => Fail("Le moteur ne répond pas au ping."));
                return;
            }
            // Le scénario fournit ses propres règles; le menu les sélectionne.
            var cfg = new JObject { ["seed"] = seed };
            var r = Request(new JObject { ["cmd"] = "reset", ["scenario"] = initialScenario,
                                          ["config"] = cfg });
            _mainThread.Enqueue(() =>
            {
                if (r != null && (bool)(r["ok"] ?? false))
                {
                    LastObservation = (JObject)r["observation"];
                    IsReady = true;
                    OnGameReset?.Invoke(LastObservation);
                    OnObservation?.Invoke(LastObservation);
                    Debug.Log("[moteur] prêt.");
                }
                else Fail("Échec du reset : " + r?["error"]);
            });
        });
    }

    void StopEngine()
    {
        try
        {
            if (_proc != null && !_proc.HasExited)
            {
                lock (_ioLock)
                {
                    _stdin?.WriteLine("{\"cmd\":\"quit\"}");
                    _stdin?.Flush();
                }
                if (!_proc.WaitForExit(1500)) _proc.Kill();
            }
        }
        catch { /* le processus est déjà mort : rien à faire */ }
        _proc = null;
    }

    void Fail(string msg)
    {
        Debug.LogError("[moteur] " + msg);
        OnError?.Invoke(msg);
    }

    // ------------------------------------------------------------------
    // Requête synchrone — À N'APPELER QUE DEPUIS UN THREAD DE FOND.
    JObject Request(JObject req)
    {
        lock (_ioLock)
        {
            try
            {
                _stdin.WriteLine(JsonConvert.SerializeObject(req));
                _stdin.Flush();
                string line = _stdout.ReadLine();
                if (string.IsNullOrEmpty(line)) return null;
                return JObject.Parse(line);
            }
            catch (Exception e)
            {
                _mainThread.Enqueue(() => Fail("E/S moteur : " + e.Message));
                return null;
            }
        }
    }

    // ------------------------------------------------------------------
    // API publique — appelable depuis l'UI, ne bloque jamais l'image.

    public void ResetGame(string scenario, bool aiOpponent = false, Action<JObject> onDone = null)
    {
        if (IsBusy) return;
        IsBusy = true;
        Task.Run(() =>
        {
            // Les préréglages de mode fournissent leurs propres règles; seul le
            // seed est remplacé ici pour garder les parties comparables.
            var cfg = new JObject { ["seed"] = seed };
            var r = Request(new JObject { ["cmd"] = "reset", ["scenario"] = scenario,
                                          ["config"] = cfg });
            _mainThread.Enqueue(() =>
            {
                IsBusy = false;
                Done = false;
                if (r == null || !(bool)(r["ok"] ?? false))
                {
                    Fail("reset : " + r?["error"]);
                    return;
                }
                LastObservation = (JObject)r["observation"];
                LastTruth = null;
                Outcome = "en_cours";
                IsReady = true;
                OnGameReset?.Invoke(LastObservation);
                OnObservation?.Invoke(LastObservation);
                onDone?.Invoke(r);
            });
        });
    }

    public void Step(JObject action, Action<JObject> onDone = null)
    {
        if (!IsReady || IsBusy || Done) return;
        IsBusy = true;
        Task.Run(() =>
        {
            var r = Request(new JObject { ["cmd"] = "step", ["action"] = action });
            _mainThread.Enqueue(() =>
            {
                IsBusy = false;
                if (r == null || !(bool)(r["ok"] ?? false))
                {
                    Fail("step : " + r?["error"]);
                    return;
                }
                LastObservation = (JObject)r["observation"];
                LastTruth = (JObject)r["truth"];      // NE PAS AFFICHER au joueur
                Done = (bool)r["done"];
                Outcome = (string)r["outcome"];
                OnObservation?.Invoke(LastObservation);
                onDone?.Invoke(r);
                if (Done) OnGameOver?.Invoke(Outcome);
            });
        });
    }

    // Raccourcis correspondant aux 5 familles d'actions de JEU.md
    public void Wait(Action<JObject> cb = null)
        => Step(new JObject { ["type"] = "wait" }, cb);

    public void Irradiate(float doseGy, Action<JObject> cb = null)
        => Step(new JObject { ["type"] = "irradiate", ["dose_gy"] = doseGy }, cb);

    public void SetMedium(float glucoseMM, float oxygenMM, Action<JObject> cb = null)
        => Step(new JObject
        {
            ["type"] = "set_medium",
            ["glucose_mM"] = glucoseMM,
            ["oxygen_mM"] = oxygenMM
        }, cb);

    public void Biopsy(Action<JObject> cb = null)
        => Step(new JObject { ["type"] = "biopsy" }, cb);

    public void Po2Imaging(Action<JObject> cb = null)
        => Step(new JObject { ["type"] = "po2_imaging" }, cb);

    public void AntiAngiogenic(float intensity = 1.0f, Action<JObject> cb = null)
        => Step(new JObject { ["type"] = "antiangiogenic", ["intensity"] = intensity }, cb);

    public void RequestAiAction(Action<JObject> cb = null)
    {
        if (!IsReady || IsBusy || Done) return;
        IsBusy = true;
        Task.Run(() =>
        {
            var r = Request(new JObject { ["cmd"] = "ai_action" });
            _mainThread.Enqueue(() =>
            {
                IsBusy = false;
                if (r != null && (bool)(r["ok"] ?? false))
                {
                    var action = (JObject)r["action"];
                    OnAiRecommendation?.Invoke(action);
                    cb?.Invoke(action);
                }
            });
        });
    }

    public void PlayAiTurn(Action<JObject> cb = null)
    {
        if (!IsReady || IsBusy || Done) return;
        IsBusy = true;
        Task.Run(() =>
        {
            var recommendation = Request(new JObject { ["cmd"] = "ai_action" });
            if (recommendation == null || !(bool)(recommendation["ok"] ?? false))
            {
                _mainThread.Enqueue(() => { IsBusy = false; Fail("ai_action indisponible"); });
                return;
            }
            var action = (JObject)recommendation["action"];
            var result = Request(new JObject { ["cmd"] = "step", ["action"] = action });
            _mainThread.Enqueue(() =>
            {
                IsBusy = false;
                if (result == null || !(bool)(result["ok"] ?? false))
                {
                    Fail("step IA : " + result?["error"]);
                    return;
                }
                LastObservation = (JObject)result["observation"];
                LastTruth = (JObject)result["truth"];
                Done = (bool)result["done"];
                Outcome = (string)result["outcome"];
                OnAiRecommendation?.Invoke(action);
                OnObservation?.Invoke(LastObservation);
                cb?.Invoke(result);
                if (Done) OnGameOver?.Invoke(Outcome);
            });
        });
    }

    /// Profils radiaux (pO2, glucose, lactate, pH) pour l'affichage.
    public void GetProfile(Action<JObject> cb)
    {
        Task.Run(() =>
        {
            var r = Request(new JObject { ["cmd"] = "profile" });
            _mainThread.Enqueue(() =>
            {
                if (r != null && (bool)(r["ok"] ?? false)) cb?.Invoke((JObject)r["profile"]);
            });
        });
    }

    /// Journalise le tour pour l'apprentissage par imitation.
    public void LogForImitation(string filePath, JObject obsBefore, JObject action,
                                JObject result)
    {
        var rec = new JObject
        {
            ["t_wall"] = DateTime.UtcNow.ToString("o"),
            ["observation"] = obsBefore,
            ["action"] = action,
            ["next_observation"] = result["observation"],
            ["reward"] = result["reward"],
            ["done"] = result["done"]
        };
        File.AppendAllText(filePath, JsonConvert.SerializeObject(rec) + "\n");
    }
}
