"""
main.py
CLI entry point for the Audio Fingerprinting & Similarity Engine.

Usage:
    python main.py process                    -> extract features for all songs in songs/
    python main.py search <song.mp3>           -> find most similar songs
    python main.py compare <song1> <song2>     -> directly compare two songs
    python main.py report <song.mp3>           -> print a fingerprint report
    python main.py stats                       -> show dataset analytics
"""

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import extract_features
from embedding_generator import build_embedding, USE_PRETRAINED
from similarity_engine import find_similar, cosine_similarity
from explainability import explain_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SONGS_DIR = os.path.join(BASE_DIR, "songs")
DATA_PATH = os.path.join(BASE_DIR, "data", "features.json")


def load_db():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    return {}


def save_db(db):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(db, f, indent=2)


def cmd_process():
    songs = [f for f in os.listdir(SONGS_DIR) if f.lower().endswith((".mp3", ".wav"))]
    if not songs:
        print(f"No audio files found in {SONGS_DIR}")
        return

    print(f"Processing {len(songs)} songs...\n")
    db = load_db()

    for song in songs:
        path = os.path.join(SONGS_DIR, song)
        try:
            if USE_PRETRAINED:
                features, y, sr = extract_features(path, return_audio=True)
                features["embedding"] = build_embedding(features, y=y, sr=sr)
            else:
                features = extract_features(path)
                features["embedding"] = build_embedding(features)
            db[song] = features
            print(f"✓ {song}")
        except Exception as e:
            print(f"✗ {song} (error: {e})")

    save_db(db)
    print("\nFeatures saved.")


def cmd_search(query_song, top_n=3):
    db = load_db()
    if query_song not in db:
        print(f"'{query_song}' not found. Run 'python main.py process' first.")
        return

    results = find_similar(query_song, db, top_n=top_n)

    print("\nMost Similar Songs\n")
    for i, (song, score) in enumerate(results, start=1):
        print(f"{i}. {song}")
        print(f"   Similarity: {score:.2f}\n")

        reasons = explain_similarity(db[query_song], db[song])
        print(f"   Reason:")
        for r in reasons:
            print(f"   {r}")
        print()


def cmd_stats():
    db = load_db()
    if not db:
        print("No data yet. Run 'python main.py process' first.")
        return

    total = len(db)
    avg_bpm = sum(s["bpm"] for s in db.values()) / total
    highest_energy = max(db.items(), key=lambda x: x[1]["energy"])
    lowest_energy = min(db.items(), key=lambda x: x[1]["energy"])

    print(f"Total Songs: {total}")
    print(f"Average BPM: {avg_bpm:.0f}\n")
    print(f"Highest Energy:\n    {highest_energy[0]}\n")
    print(f"Lowest Energy:\n    {lowest_energy[0]}")


def _energy_label(energy, db):
    """Bucket a raw energy value into Low/Medium/High relative to the dataset."""
    if not db:
        return "Unknown"
    values = sorted(s["energy"] for s in db.values())
    n = len(values)
    low_cut = values[n // 3] if n >= 3 else values[0]
    high_cut = values[(2 * n) // 3] if n >= 3 else values[-1]
    if energy <= low_cut:
        return "Low"
    elif energy >= high_cut:
        return "High"
    return "Medium"


def cmd_compare(song_a, song_b):
    db = load_db()
    for s in (song_a, song_b):
        if s not in db:
            print(f"'{s}' not found. Run 'python main.py process' first.")
            return

    feat_a, feat_b = db[song_a], db[song_b]
    score = cosine_similarity(feat_a["embedding"], feat_b["embedding"])
    reasons = explain_similarity(feat_a, feat_b)

    chroma_a = feat_a.get("chroma", [])
    chroma_b = feat_b.get("chroma", [])
    harmonic = "Unknown"
    if chroma_a and chroma_b:
        diff = sum(abs(x - y) for x, y in zip(chroma_a, chroma_b)) / len(chroma_a)
        harmonic = "High" if diff < 0.08 else ("Medium" if diff < 0.15 else "Low")

    print(f"\nSimilarity Score: {score:.2f}\n")
    print(f"Tempo:\n{feat_a['bpm']} vs {feat_b['bpm']} BPM\n")
    print(f"Energy:\n{feat_a['energy']} vs {feat_b['energy']}\n")
    print(f"Harmonic Similarity:\n{harmonic}\n")
    print("Reason:")
    for r in reasons:
        print(f"   {r}")


def cmd_report(song, top_n=3):
    db = load_db()
    if song not in db:
        print(f"'{song}' not found. Run 'python main.py process' first.")
        return

    feat = db[song]
    matches = find_similar(song, db, top_n=top_n)

    print("\nAudio Fingerprint Report\n")
    print(f"Tempo: {feat['bpm']} BPM")
    print(f"Energy: {_energy_label(feat['energy'], db)}")
    print(f"Brightness: {_energy_label(feat['brightness'], {k: {'energy': v['brightness']} for k, v in db.items()})}\n")
    print("Closest Matches:")
    for i, (match_song, _score) in enumerate(matches, start=1):
        print(f"{i}. {match_song}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "process":
        cmd_process()
    elif command == "search":
        if len(sys.argv) < 3:
            print("Usage: python main.py search <song.mp3>")
            return
        cmd_search(sys.argv[2])
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: python main.py compare <song1.mp3> <song2.mp3>")
            return
        cmd_compare(sys.argv[2], sys.argv[3])
    elif command == "report":
        if len(sys.argv) < 3:
            print("Usage: python main.py report <song.mp3>")
            return
        cmd_report(sys.argv[2])
    elif command == "stats":
        cmd_stats()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
