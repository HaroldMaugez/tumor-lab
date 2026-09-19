// SimpleTumorUI.cs — interface minimale "formes simples".
//
// OBJECTIF : jouer une partie complète sans importer le moindre asset.
// Tout est dessiné par code : la tumeur est un disque (Sprite généré à la
// volée), le reste est de l'IMGUI (OnGUI). Aucun Canvas à configurer, aucune
// texture à importer, aucun prefab.
//
// MISE EN PLACE (30 secondes)
//   1. Scène vide -> GameObject vide nommé "Game"
//   2. Ajouter le composant TumorEngine
//   3. Ajouter le composant SimpleTumorUI
//   4. Play
//
// Quand vous passerez à une vraie UI, jetez ce fichier : il ne sert qu'à
// prouver que la boucle fonctionne de bout en bout.

using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

[RequireComponent(typeof(TumorEngine))]
public class SimpleTumorUI : MonoBehaviour
{
    TumorEngine _engine;
    TumorGameMenu _menu;
    SpriteRenderer _tumor;      // disque = tumeur
    SpriteRenderer _necrotic;   // disque sombre = coeur nécrotique
    readonly List<float> _diameterHistory = new List<float>();

    float _displayDiameter = 1f;   // µm, lissé pour l'affichage
    float _targetDiameter = 1f;
    string _message = "Initialisation du moteur…";
    Vector2 _leftPanelScroll;

    // --- mise en place de la scène, entièrement par code -----------------
    void Start()
    {
        _engine = GetComponent<TumorEngine>();
        _menu = GetComponent<TumorGameMenu>();
        _engine.OnObservation += OnObservation;
        _engine.OnGameReset += OnGameReset;
        _engine.OnGameOver += o => _message = "PARTIE TERMINÉE : " + Pretty(o);
        _engine.OnError += e => _message = "ERREUR : " + e;

        var cam = Camera.main;
        if (cam == null)
        {
            var go = new GameObject("Main Camera");
            cam = go.AddComponent<Camera>();
            go.tag = "MainCamera";
        }
        cam.orthographic = true;
        cam.orthographicSize = 6f;
        cam.backgroundColor = new Color(0.09f, 0.10f, 0.13f);
        cam.transform.position = new Vector3(0, 0, -10);

        _tumor = MakeDisc("Tumeur", new Color(0.85f, 0.35f, 0.35f), 0);
        _necrotic = MakeDisc("Necrose", new Color(0.25f, 0.18f, 0.20f), 1);
    }

    SpriteRenderer MakeDisc(string name, Color color, int order)
    {
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var sr = go.AddComponent<SpriteRenderer>();
        sr.sprite = DiscSprite();
        sr.color = color;
        sr.sortingOrder = order;
        return sr;
    }

    // Génère une texture de disque anticrénelé : évite d'importer un sprite.
    static Sprite _cachedDisc;
    static Sprite DiscSprite()
    {
        if (_cachedDisc != null) return _cachedDisc;
        const int S = 256;
        var tex = new Texture2D(S, S, TextureFormat.RGBA32, false);
        float r = S / 2f - 1f;
        var c = new Vector2(S / 2f, S / 2f);
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
            {
                float d = Vector2.Distance(new Vector2(x + .5f, y + .5f), c);
                float a = Mathf.Clamp01(r - d);          // bord adouci sur 1 px
                tex.SetPixel(x, y, new Color(1, 1, 1, a));
            }
        tex.Apply();
        tex.filterMode = FilterMode.Bilinear;
        _cachedDisc = Sprite.Create(tex, new Rect(0, 0, S, S), new Vector2(.5f, .5f), S);
        return _cachedDisc;
    }

    // --- réception des états ---------------------------------------------
    void OnGameReset(JObject obs)
    {
        _diameterHistory.Clear();
        _leftPanelScroll = Vector2.zero;
        _displayDiameter = 1f;
        _targetDiameter = obs != null && obs["diameter_um_observed"] != null
            ? (float)obs["diameter_um_observed"] : 1f;
        _message = "Nouvelle partie";
    }

    void OnObservation(JObject obs)
    {
        _targetDiameter = (float)obs["diameter_um_observed"];
        _diameterHistory.Add(_targetDiameter);
        if (_diameterHistory.Count > 120) _diameterHistory.RemoveAt(0);
        _message = "";
    }

    void Update()
    {
        // interpolation douce : sans cela le disque saute d'un tour à l'autre
        _displayDiameter = Mathf.Lerp(_displayDiameter, _targetDiameter,
                                      Time.deltaTime * 4f);
        // 2000 µm (diamètre de défaite) -> 8 unités de monde
        float scale = _displayDiameter / 2000f * 8f;
        _tumor.transform.localScale = Vector3.one * Mathf.Max(scale, 0.01f);

        // le coeur nécrotique est une simple fraction du rayon (visuel)
        _necrotic.transform.localScale = Vector3.one * Mathf.Max(scale * 0.55f, 0.005f);
        _necrotic.gameObject.SetActive(_displayDiameter > 500f);
    }

    // --- interface immédiate ---------------------------------------------
    void OnGUI()
    {
        // Le menu principal possède sa propre surface; ne pas dessiner le HUD
        // derrière lui, ce qui évite les zones cliquables superposées.
        if (_menu != null && _menu.IsMenuOpen) return;
        var obs = _engine.LastObservation;
        GUI.skin.label.fontSize = 14;
        GUI.skin.button.fontSize = 14;

        // Le contenu dépasse volontairement la hauteur sur les petits écrans :
        // le ScrollView évite que les examens et actions soient coupés.
        GUILayout.BeginArea(new Rect(12, 12, 350, 500), GUI.skin.box);
        _leftPanelScroll = GUILayout.BeginScrollView(
            _leftPanelScroll, false, true, GUILayout.Width(342), GUILayout.Height(492));
        GUILayout.Label("<b>SIMULATEUR DE RADIOTHÉRAPIE</b>",
                        new GUIStyle(GUI.skin.label) { richText = true, fontSize = 16 });
        GUILayout.Space(6);

        if (obs == null)
        {
            GUILayout.Label(_message);
            GUILayout.EndScrollView();
            GUILayout.EndArea();
            return;
        }

        float tox = (float)obs["toxicity_score"];
        GUILayout.Label($"Jour            : {(float)obs["day"]:F0} / {(float)obs["days_remaining"] + (float)obs["day"]:F0}");
        GUILayout.Label($"Diamètre observé: {(float)obs["diameter_um_observed"]:F0} µm  (± 5 %)");
        GUILayout.Label($"Marqueur sérique: {(float)obs["serum_marker"]:E2}  (± 20 %)");
        GUILayout.Label($"Régime           : {obs["model_regime"]}");
        GUILayout.Label($"Dose délivrée   : {(float)obs["dose_delivered_gy"]:F1} Gy");
        GUILayout.Label($"Budget restant  : {(float)obs["dose_budget_remaining_gy"]:F1} Gy");

        // la toxicité est LA jauge à surveiller : défaite à 50
        var toxStyle = new GUIStyle(GUI.skin.label) { richText = true };
        string toxColor = tox < 5 ? "#7fdc7f" : tox < 25 ? "#e8d47a" : "#e87a7a";
        GUILayout.Label($"Toxicité (NTCP) : <color={toxColor}><b>{tox:F2} / 50</b></color>", toxStyle);
        if (obs["vascular_activity"] != null)
            GUILayout.Label($"Activité vasculaire: {(float)obs["vascular_activity"]:P0}");
        GUILayout.Space(4);
        DrawBar(tox / 50f, tox < 5 ? Color.green : tox < 25 ? Color.yellow : Color.red);
        GUILayout.Space(10);

        bool busy = _engine.IsBusy || !_engine.IsReady || _engine.Done;
        GUI.enabled = !busy;

        GUILayout.Label("<b>Actions</b>", new GUIStyle(GUI.skin.label) { richText = true });
        if (GUILayout.Button("Attendre 1 jour")) _engine.Wait();
        if (obs["vascular_activity"] != null && (float)obs["vascular_activity"] > 0f
            && GUILayout.Button("Anti-angiogénique"))
            _engine.AntiAngiogenic(0.65f);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("1.8 Gy")) _engine.Irradiate(1.8f);
        if (GUILayout.Button("2 Gy")) _engine.Irradiate(2.0f);
        if (GUILayout.Button("3 Gy")) _engine.Irradiate(3.0f);
        if (GUILayout.Button("5 Gy")) _engine.Irradiate(5.0f);
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Biopsie (1 j)")) _engine.Biopsy();
        if (GUILayout.Button("Imagerie pO₂ (0.5 j)")) _engine.Po2Imaging();
        GUILayout.EndHorizontal();

        GUILayout.Label("Microenvironnement");
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("Réoxygéner")) _engine.SetMedium(16.5f, 0.28f);
        if (GUILayout.Button("Milieu pauvre")) _engine.SetMedium(0.8f, 0.07f);
        GUILayout.EndHorizontal();

        GUI.enabled = true;
        GUILayout.Space(8);

        if (_engine.IsBusy)
            GUILayout.Label("<i>Le temps passe…</i>",
                            new GUIStyle(GUI.skin.label) { richText = true });
        if (!string.IsNullOrEmpty(_message))
            GUILayout.Label($"<b>{_message}</b>",
                            new GUIStyle(GUI.skin.label) { richText = true, fontSize = 15 });

        var biopsy = obs["last_biopsy"];
        if (biopsy != null && biopsy.Type != JTokenType.Null)
            GUILayout.Label($"Biopsie j{(float)biopsy["day"]:F0} : résistance estimée "
                            + $"{(float)biopsy["resistance_pct_estimated"]:F1} %");

        GUILayout.EndScrollView();
        GUILayout.EndArea();

        DrawHistory();
    }

    void DrawBar(float fill, Color c)
    {
        var r = GUILayoutUtility.GetRect(300, 10);
        GUI.color = new Color(1, 1, 1, 0.15f);
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = c;
        GUI.DrawTexture(new Rect(r.x, r.y, r.width * Mathf.Clamp01(fill), r.height),
                        Texture2D.whiteTexture);
        GUI.color = Color.white;
    }

    // Courbe de suivi, dessinée avec des rectangles de 2 px : pas de dépendance.
    void DrawHistory()
    {
        if (_diameterHistory.Count < 2) return;
        var area = new Rect(Screen.width - 320, 12, 300, 140);
        GUI.Box(area, "Diamètre observé (µm)");
        float max = 100f;
        foreach (var v in _diameterHistory) max = Mathf.Max(max, v);
        int n = _diameterHistory.Count;
        for (int i = 0; i < n; i++)
        {
            float x = area.x + 8 + (area.width - 16) * i / Mathf.Max(n - 1, 1);
            float h = (area.height - 30) * (_diameterHistory[i] / max);
            GUI.color = new Color(0.85f, 0.45f, 0.45f);
            GUI.DrawTexture(new Rect(x, area.yMax - 8 - h, 2, h), Texture2D.whiteTexture);
        }
        GUI.color = Color.white;
    }

    static string Pretty(string outcome)
    {
        switch (outcome)
        {
            case "victoire_totale": return "VICTOIRE — tumeur éradiquée";
            case "controle_durable": return "CONTRÔLE DURABLE";
            case "toxicite_letale": return "ÉCHEC — toxicité létale pour le patient";
            case "progression": return "ÉCHEC — progression tumorale";
            case "echec_controle": return "ÉCHEC — contrôle non obtenu";
            default: return outcome;
        }
    }
}
