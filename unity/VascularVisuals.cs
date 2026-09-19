// VascularVisuals.cs
// Procedural placeholder for the vascular mode. Replace this component with
// authored sprites/materials using ART_ASSETS.md when final art is available.
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

[RequireComponent(typeof(TumorEngine))]
public class VascularVisuals : MonoBehaviour
{
    public int vesselCount = 12;
    public float baseLength = 3.2f;
    public float maxWidth = 0.045f;
    public Color vesselColor = new Color(0.85f, 0.18f, 0.25f, 0.72f);

    TumorEngine _engine;
    readonly List<LineRenderer> _vessels = new List<LineRenderer>();
    float _activity;
    float _diameter = 900f;

    void Awake()
    {
        _engine = GetComponent<TumorEngine>();
        _engine.OnObservation += OnObservation;
        _engine.OnGameReset += OnGameReset;
        BuildVessels();
    }

    void OnDestroy()
    {
        if (_engine != null)
        {
            _engine.OnObservation -= OnObservation;
            _engine.OnGameReset -= OnGameReset;
        }
    }

    void BuildVessels()
    {
        for (int i = 0; i < vesselCount; i++)
        {
            var go = new GameObject("Vessel_" + i);
            go.transform.SetParent(transform, false);
            var line = go.AddComponent<LineRenderer>();
            line.positionCount = 5;
            line.useWorldSpace = false;
            line.material = new Material(Shader.Find("Sprites/Default"));
            line.startColor = vesselColor;
            line.endColor = vesselColor;
            line.sortingOrder = -1;
            _vessels.Add(line);
        }
    }

    void OnGameReset(JObject observation)
    {
        _activity = 0f;
        _diameter = observation != null && observation["diameter_um_observed"] != null
            ? (float)observation["diameter_um_observed"] : 900f;
        UpdateVessels();
    }

    void OnObservation(JObject observation)
    {
        if (observation == null) return;
        if (observation["vascular_activity"] != null)
            _activity = Mathf.Clamp01((float)observation["vascular_activity"]);
        if (observation["diameter_um_observed"] != null)
            _diameter = Mathf.Max(1f, (float)observation["diameter_um_observed"]);
        UpdateVessels();
    }

    void UpdateVessels()
    {
        // Activity controls visibility and length, never the hidden truth state.
        float radius = Mathf.Clamp(_diameter / 2000f * 4f, 0.4f, 4f);
        float visibility = Mathf.Clamp01(_activity * 2.2f);
        for (int i = 0; i < _vessels.Count; i++)
        {
            var line = _vessels[i];
            bool active = visibility > 0.03f && i < Mathf.CeilToInt(vesselCount * visibility);
            line.gameObject.SetActive(active);
            if (!active) continue;

            float angle = 2f * Mathf.PI * i / Mathf.Max(1, vesselCount);
            Vector3 direction = new Vector3(Mathf.Cos(angle), Mathf.Sin(angle), 0f);
            Vector3 tangent = new Vector3(-direction.y, direction.x, 0f);
            float length = baseLength * (0.55f + 0.75f * _activity);
            float inner = radius * (0.78f + 0.12f * Mathf.Sin(i * 2.3f));
            float outer = inner + length * (0.35f + 0.65f * _activity);
            float bend = 0.12f * radius * Mathf.Sin(i * 1.7f);

            line.startWidth = maxWidth * (0.45f + 0.55f * _activity);
            line.endWidth = maxWidth * 0.45f;
            line.SetPosition(0, direction * inner);
            line.SetPosition(1, direction * (inner + (outer - inner) * 0.30f) + tangent * bend);
            line.SetPosition(2, direction * (inner + (outer - inner) * 0.58f) - tangent * bend);
            line.SetPosition(3, direction * (inner + (outer - inner) * 0.82f) + tangent * bend * 0.5f);
            line.SetPosition(4, direction * outer);
        }
    }
}
