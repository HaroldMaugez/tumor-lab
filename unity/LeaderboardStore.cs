// LeaderboardStore.cs
// Offline-first scores for prototypes and classroom builds. A real global
// leaderboard must verify runs on a server; see ILeaderboardBackend below.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using UnityEngine;

[Serializable]
public class LeaderboardEntry
{
    public string playerName;
    public string mode;
    public string modelRegime;
    public float score;
    public string outcome;
    public int seed;
    public string createdUtc;
}

[Serializable]
public class LeaderboardFile
{
    public int schemaVersion = 1;
    public List<LeaderboardEntry> entries = new List<LeaderboardEntry>();
}

public interface ILeaderboardBackend
{
    void Submit(LeaderboardEntry entry, Action<bool, string> callback);
}

public sealed class LeaderboardStore
{
    const int MaxEntries = 100;
    readonly string _path;

    public LeaderboardStore(string fileName = "leaderboard.json")
    {
        _path = Path.Combine(Application.persistentDataPath, fileName);
    }

    public List<LeaderboardEntry> Load()
    {
        try
        {
            if (!File.Exists(_path)) return new List<LeaderboardEntry>();
            var file = JsonConvert.DeserializeObject<LeaderboardFile>(File.ReadAllText(_path));
            return file == null || file.entries == null
                ? new List<LeaderboardEntry>()
                : file.entries.OrderByDescending(e => e.score).Take(MaxEntries).ToList();
        }
        catch (Exception e)
        {
            Debug.LogWarning("Leaderboard illisible: " + e.Message);
            return new List<LeaderboardEntry>();
        }
    }

    public void Add(LeaderboardEntry entry)
    {
        var entries = Load();
        entries.Add(entry);
        entries = entries.OrderByDescending(e => e.score).Take(MaxEntries).ToList();
        var file = new LeaderboardFile { entries = entries };
        Directory.CreateDirectory(Path.GetDirectoryName(_path));
        File.WriteAllText(_path, JsonConvert.SerializeObject(file, Formatting.Indented));
    }
}
