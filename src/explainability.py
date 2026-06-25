"""
explainability.py
Generates human-readable reasons for why two songs were considered similar.
"""

def explain_similarity(query_features, candidate_features, threshold_ratio=0.15):
    reasons = []

    # --- Tempo ---
    bpm_q, bpm_c = query_features["bpm"], candidate_features["bpm"]
    if abs(bpm_q - bpm_c) <= max(bpm_q, bpm_c) * threshold_ratio:
        reasons.append("✓ Similar tempo")

    # --- Energy ---
    e_q, e_c = query_features["energy"], candidate_features["energy"]
    if abs(e_q - e_c) <= max(e_q, e_c, 0.001) * threshold_ratio * 2:
        reasons.append("✓ Similar energy profile")

    # --- Harmonic structure (chroma) ---
    chroma_q = query_features.get("chroma", [])
    chroma_c = candidate_features.get("chroma", [])
    if chroma_q and chroma_c:
        diff = sum(abs(a - b) for a, b in zip(chroma_q, chroma_c)) / len(chroma_q)
        if diff < 0.08:
            reasons.append("✓ Similar harmonic structure")

    # --- Brightness / timbre ---
    b_q, b_c = query_features.get("brightness", 0), candidate_features.get("brightness", 0)
    if abs(b_q - b_c) <= max(b_q, b_c, 1) * threshold_ratio * 2:
        reasons.append("✓ Similar timbre / brightness")

    if not reasons:
        reasons.append("• Matched primarily on overall embedding similarity")

    return reasons
