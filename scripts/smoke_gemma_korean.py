"""
smoke_gemma_korean.py — Gate #1 from design spec §8.1.

Loads 10 Korean interview prompts. For each:
  - Runs Gemma 4 E2B locally (LiteRT-LM)
  - Runs Gemini 2.0 Flash (cloud) as baseline
  - Shows both responses side-by-side with blind labels (A / B)
  - Prompts operator for ratings (fluency/relevance/register, 1-5)
Aggregates and writes results JSON + prints pass/fail verdict.

PASS criteria (spec §8.1 Gate #1):
  - mean rating across all axes and prompts >= 3.5 / 5
  - no individual rating <= 2.0

Note: the LiteRT-LM C++ extension may SIGSEGV at interpreter teardown after
main() returns. Result JSON is always written BEFORE the success marker so the
file survives the crash.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt

from scripts._smoke_helpers import Rating, aggregate_ratings, format_pass_fail

REPO_ROOT = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "gemma-4-e2b" / "gemma-4-E2B-it.litertlm"
FIXTURE = REPO_ROOT / "scripts" / "fixtures" / "korean_prompts.json"
RESULTS_DIR = REPO_ROOT / "scripts" / "_results"

# Pass/fail thresholds (spec §8.1 Gate #1)
MIN_MEAN = 3.5
MIN_INDIVIDUAL = 2.0  # any rating <= this is a fail

# System message appended to each prompt (mirrors production registry).
SYSTEM_MSG = (
    "당신은 한국어 면접관입니다. "
    "정중한 존댓말로, 2-3문장 이내로 간결하게 응답하세요. "
    "면접 주제에서 벗어나지 마세요."
)


def _extract_text(response: dict) -> str:
    """Unwrap LiteRT-LM response dict to plain text."""
    try:
        return response["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return str(response)


def _load_gemma() -> Callable[[str], str]:
    from litert_lm import Engine, Backend  # type: ignore

    engine = Engine(model_path=str(MODEL_PATH), backend=Backend.CPU)

    def gen(user_msg: str) -> str:
        # Fresh conversation per prompt so each turn starts with the system message.
        conv = engine.create_conversation()
        full_prompt = f"{SYSTEM_MSG}\n\n사용자: {user_msg}\n면접관:"
        resp = conv.send_message(full_prompt)
        return _extract_text(resp).strip()

    return gen


def _load_gemini() -> Callable[[str], str]:
    # Reuse the project's existing Gemini wrapper for fair comparison.
    sys.path.insert(0, str(REPO_ROOT))
    from start_app.dialogue_manager.meeting import gemini_chat_call  # type: ignore

    def gen(user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ]
        return gemini_chat_call(messages, max_tokens=200, temperature=0.7)

    return gen


def _collect_rating(console: Console, prompt_id: str) -> Rating:
    console.print(f"[bold]Rate prompt [{prompt_id}][/bold] on three axes (1..5):")
    fluency = IntPrompt.ask("  유창성 (Korean fluency)", choices=["1", "2", "3", "4", "5"])
    relevance = IntPrompt.ask("  관련성 (on-topic)", choices=["1", "2", "3", "4", "5"])
    register = IntPrompt.ask("  격식 (formal register)", choices=["1", "2", "3", "4", "5"])
    return Rating(fluency=fluency, relevance=relevance, register=register)


def main() -> int:
    console = Console()
    RESULTS_DIR.mkdir(exist_ok=True)

    console.print(Panel.fit("[bold]Gate #1 — Gemma 4 E2B vs Gemini 2.0 Flash (Korean)[/bold]"))
    prompts = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]

    console.print("Loading models...")
    gemma = _load_gemma()
    gemini = _load_gemini()
    console.print("[green]Both models ready.[/green]\n")

    per_prompt: list[dict] = []
    ratings_gemma: list[Rating] = []
    ratings_gemini: list[Rating] = []

    for p in prompts:
        console.rule(f"[bold]{p['id']}[/bold] — {p['category']}")
        console.print(f"[italic]Prompt:[/italic] {p['text']}\n")

        # Blind assignment — randomise which model is shown as A vs B per prompt.
        show_gemma_first = random.random() < 0.5
        resp_gemma = gemma(p["text"])
        resp_gemini = gemini(p["text"])

        label_a, label_b = ("gemma", "gemini") if show_gemma_first else ("gemini", "gemma")
        resp_a = resp_gemma if show_gemma_first else resp_gemini
        resp_b = resp_gemini if show_gemma_first else resp_gemma

        console.print(Panel(resp_a, title="[bold]Response A[/bold]", expand=False))
        console.print(Panel(resp_b, title="[bold]Response B[/bold]", expand=False))

        console.print("\n[cyan]Rate response A:[/cyan]")
        r_a = _collect_rating(console, f"{p['id']}-A")
        console.print("\n[cyan]Rate response B:[/cyan]")
        r_b = _collect_rating(console, f"{p['id']}-B")

        # Map blind ratings back to the actual model.
        r_gemma = r_a if label_a == "gemma" else r_b
        r_gemini = r_b if label_b == "gemini" else r_a
        ratings_gemma.append(r_gemma)
        ratings_gemini.append(r_gemini)

        per_prompt.append({
            "id": p["id"],
            "category": p["category"],
            "prompt_text": p["text"],
            "gemma_response": resp_gemma,
            "gemini_response": resp_gemini,
            "gemma_rating": r_gemma.__dict__,
            "gemini_rating": r_gemini.__dict__,
        })

    agg_gemma = aggregate_ratings(ratings_gemma)
    agg_gemini = aggregate_ratings(ratings_gemini)

    passed = agg_gemma["mean"] >= MIN_MEAN and agg_gemma["min"] > MIN_INDIVIDUAL

    result = {
        "gate": "1_korean_dialogue_quality",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "thresholds": {"min_mean": MIN_MEAN, "min_individual_floor": MIN_INDIVIDUAL},
        "gemma": agg_gemma,
        "gemini_baseline": agg_gemini,
        "passed": passed,
        "per_prompt": per_prompt,
    }
    out_path = RESULTS_DIR / "gate1_korean.json"
    # Write BEFORE the final print so SIGSEGV at teardown doesn't lose results.
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print("\n" + "=" * 60)
    console.print(f"[bold]Gate #1 result:[/bold] {format_pass_fail(passed)}")
    console.print(f"  Gemma 4 E2B mean = {agg_gemma['mean']:.2f} (threshold ≥ {MIN_MEAN})")
    console.print(f"  Gemma 4 E2B min  = {agg_gemma['min']} (threshold > {MIN_INDIVIDUAL})")
    console.print(f"  Gemini baseline  mean = {agg_gemini['mean']:.2f}")
    console.print(f"  Results written to: {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    random.seed(42)  # reproducible blind assignment
    sys.exit(main())
