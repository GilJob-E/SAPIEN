"""Claude-controller ratings for Gate #1.

Reads gate1_responses_raw.json, applies manual ratings, writes final
gate1_korean.json using the scripts._smoke_helpers aggregator.

Each rating is 1-5 on (fluency, relevance, register). Rationales are in
smoke_results.md for operator verification.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts._smoke_helpers import Rating, aggregate_ratings, format_pass_fail

REPO_ROOT = Path(__file__).parent.parent
RAW = REPO_ROOT / "scripts" / "_results" / "gate1_responses_raw.json"
OUT = REPO_ROOT / "scripts" / "_results" / "gate1_korean.json"

MIN_MEAN = 3.5
MIN_INDIVIDUAL = 2.0

# Ratings by prompt id. Format: (fluency, relevance, register)
# Scored by Claude controller based on responses in gate1_responses_raw.json.
# Key calibration notes:
#  - Korean fluency: natural → 5, placeholders/broken → 2-3
#  - Relevance: correctly-roled, substantive → 5, role-confused/shallow → 3-4,
#               template-only or off-topic → 2
#  - Register: 존댓말 consistent & formal → 5; slight casual drift → 4
RATINGS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # prompt_id: (gemma_rating, gemini_rating)
    "p01": ((4, 3, 5), (5, 5, 5)),
    "p02": ((3, 3, 5), (5, 5, 5)),
    "p03": ((3, 2, 5), (5, 4, 5)),   # Gemma: 4x [placeholder]s; Gemini: AI persona pivot
    "p04": ((2, 3, 4), (5, 5, 5)),   # Gemma: echoes prompt + "면접관 답변:" header
    "p05": ((5, 5, 5), (5, 3, 5)),   # Gemini: AI persona ("사용자분께") breaks character
    "p06": ((5, 5, 5), (5, 5, 5)),   # Both solid on technical question
    "p07": ((5, 5, 5), (5, 5, 5)),
    "p08": ((5, 5, 4), (5, 5, 5)),   # Gemma: "주시겠어요?" slightly less formal
    "p09": ((5, 5, 5), (4, 4, 5)),   # Gemini: archaic "~것이오니" + meta-comment
    "p10": ((5, 3, 5), (5, 4, 5)),   # Gemma: role-mirrors (asks company vision back)
}


def _to_ratings(trio: tuple[int, int, int]) -> Rating:
    f, rel, reg = trio
    return Rating(fluency=f, relevance=rel, register=reg)


def main() -> int:
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    per_prompt = []
    gemma_ratings: list[Rating] = []
    gemini_ratings: list[Rating] = []

    for entry in raw["responses"]:
        pid = entry["id"]
        if pid not in RATINGS:
            print(f"[ERROR] missing ratings for {pid}")
            return 1
        g_trio, gg_trio = RATINGS[pid]
        r_gemma = _to_ratings(g_trio)
        r_gemini = _to_ratings(gg_trio)
        gemma_ratings.append(r_gemma)
        gemini_ratings.append(r_gemini)
        per_prompt.append({
            "id": pid,
            "category": entry["category"],
            "prompt_text": entry["prompt_text"],
            "gemma_response": entry["gemma_response"],
            "gemma_latency_s": entry["gemma_latency_s"],
            "gemma_rating": r_gemma.__dict__,
            "gemini_response": entry["gemini_response"],
            "gemini_latency_s": entry["gemini_latency_s"],
            "gemini_rating": r_gemini.__dict__,
            "gemini_model": entry.get("gemini_model", "gemini-2.5-flash"),
        })

    agg_gemma = aggregate_ratings(gemma_ratings)
    agg_gemini = aggregate_ratings(gemini_ratings)
    passed = agg_gemma["mean"] >= MIN_MEAN and agg_gemma["min"] > MIN_INDIVIDUAL

    result = {
        "gate": "1_korean_dialogue_quality",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"min_mean": MIN_MEAN, "min_individual_floor": MIN_INDIVIDUAL},
        "rater": "claude-controller (automated, to be verified by human operator)",
        "gemma": agg_gemma,
        "gemini_baseline": agg_gemini,
        "passed": passed,
        "per_prompt": per_prompt,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"Gate #1 result: {format_pass_fail(passed)}")
    print(f"  Gemma mean = {agg_gemma['mean']:.2f} (threshold ≥ {MIN_MEAN})")
    print(f"  Gemma min  = {agg_gemma['min']} (threshold > {MIN_INDIVIDUAL})")
    print(f"  Gemini baseline mean = {agg_gemini['mean']:.2f}")
    print(f"  Gemini baseline min  = {agg_gemini['min']}")
    print(f"  Results: {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
