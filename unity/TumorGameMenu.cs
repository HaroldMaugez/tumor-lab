// TumorGameMenu.cs
// Drop this component beside TumorEngine. It is intentionally code-driven so
// the first playable build needs no scene wiring beyond the existing bridge.
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

[RequireComponent(typeof(TumorEngine))]
public class TumorGameMenu : MonoBehaviour
{
    TumorEngine _engine;
    readonly string[] _modes = { "tutoriel", "vascular_human", "standard", "in_vitro_strict" };
    int _selected = 1;
    bool _menuOpen = true;
    bool _aiOpponent;
    string _selectedMode = "vascular_human";
    string _playerName = "Player";
    string _status = "Choisissez un mode";
    LeaderboardStore _scores;
    List<LeaderboardEntry> _entries = new List<LeaderboardEntry>();
    Vector2 _scroll;

    public bool IsMenuOpen { get { return _menuOpen; } }

    void Awake()
    {
        _engine = GetComponent<TumorEngine>();
        _scores = new LeaderboardStore();
        _entries = _scores.Load();
        _engine.OnGameOver += OnGameOver;
        _engine.OnError += e => _status = e;
        _engine.OnAiRecommendation += action =>
            _status = "IA: " + (action["reason"]?.ToString() ?? action["type"]?.ToString());
    }

    void OnGUI()
    {
        if (_menuOpen)
        {
            DrawMenu();
            return;
        }
        DrawInGameBar();
    }

    void DrawMenu()
    {
        float width = Mathf.Min(560f, Screen.width - 32f);
        float height = Mathf.Min(620f, Screen.height - 32f);
        GUILayout.BeginArea(new Rect((Screen.width - width) / 2f,
                                     (Screen.height - height) / 2f, width, height), GUI.skin.box);
        GUILayout.Label("TUMOR LAB", HeaderStyle(26));
        GUILayout.Label("Un modèle vivant, un patient à préserver, une stratégie à inventer.");
        GUILayout.Space(12);
        GUILayout.Label("MODE DE JEU", HeaderStyle(15));
        for (int i = 0; i < _modes.Length; i++)
        {
            string label = ModeLabel(_modes[i]);
            if (GUILayout.Toggle(_selected == i, label, GUI.skin.button)) _selected = i;
        }
        GUILayout.Space(8);
        _aiOpponent = GUILayout.Toggle(_aiOpponent, "Adversaire IA explicable");
        GUILayout.Label("Nom");
        _playerName = GUILayout.TextField(_playerName, 24);
        GUILayout.Space(8);
        if (GUILayout.Button("LANCER LA PARTIE", GUILayout.Height(42)))
        {
            _selectedMode = _modes[_selected];
            _engine.ResetGame(_selectedMode, _aiOpponent);
            _menuOpen = false;
            _status = "La partie commence";
        }
        GUILayout.Space(10);
        GUILayout.Label("CLASSEMENT LOCAL", HeaderStyle(15));
        _scroll = GUILayout.BeginScrollView(_scroll, GUILayout.Height(180));
        for (int i = 0; i < _entries.Count; i++)
        {
            var e = _entries[i];
            GUILayout.Label(string.Format("{0,2}. {1}  {2,7:0}  {3}",
                i + 1, e.playerName, e.score, ModeLabel(e.mode)));
        }
        GUILayout.EndScrollView();
        GUILayout.Label("Les scores locaux sont persistants. Un classement mondial exige une validation serveur.",
                        SmallStyle());
        GUILayout.EndArea();
    }

    void DrawInGameBar()
    {
        GUILayout.BeginArea(new Rect(Screen.width - 392, Screen.height - 122, 380, 110), GUI.skin.box);
        GUILayout.Label(_status);
        if (_aiOpponent && GUILayout.Button("TOUR DE L'IA EXPLICABLE"))
            _engine.PlayAiTurn();
        if (GUILayout.Button("MENU PRINCIPAL")) _menuOpen = true;
        GUILayout.EndArea();
    }

    void OnGameOver(string outcome)
    {
        JObject obs = _engine.LastObservation;
        float score = obs != null && obs["score"] != null ? (float)obs["score"] : 0f;
        _scores.Add(new LeaderboardEntry {
            playerName = string.IsNullOrWhiteSpace(_playerName) ? "Player" : _playerName,
            mode = string.IsNullOrEmpty(_selectedMode) ? _modes[_selected] : _selectedMode,
            modelRegime = obs?["model_regime"]?.ToString(),
            score = score, outcome = outcome, seed = _engine.seed,
            createdUtc = System.DateTime.UtcNow.ToString("o")
        });
        _entries = _scores.Load();
        _status = string.Format("Partie terminée: {0} | Score {1:0}", outcome, score);
    }

    static string ModeLabel(string mode)
    {
        switch (mode)
        {
            case "tutoriel": return "Tutoriel: comprendre les signaux";
            case "vascular_human": return "Humain fictif: angiogenèse";
            case "standard": return "Stratégie adaptative";
            case "in_vitro_strict": return "Laboratoire: avasculaire validé";
            default: return mode;
        }
    }

    static GUIStyle HeaderStyle(int size)
    {
        return new GUIStyle(GUI.skin.label) { fontSize = size, fontStyle = FontStyle.Bold };
    }

    static GUIStyle SmallStyle()
    {
        return new GUIStyle(GUI.skin.label) { fontSize = 11, wordWrap = true };
    }
}
