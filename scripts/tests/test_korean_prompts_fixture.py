"""Structural validation of the Korean prompts fixture used by Gate #1."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures" / "korean_prompts.json"


def test_fixture_exists():
    assert FIXTURE.exists(), f"Missing fixture: {FIXTURE}"


def test_fixture_is_valid_json():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_fixture_has_exactly_ten_prompts():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert "prompts" in data
    assert len(data["prompts"]) == 10


def test_every_prompt_has_required_fields():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for i, p in enumerate(data["prompts"]):
        assert "id" in p, f"prompt[{i}] missing 'id'"
        assert "category" in p, f"prompt[{i}] missing 'category'"
        assert "text" in p, f"prompt[{i}] missing 'text'"
        assert len(p["text"]) >= 10, f"prompt[{i}] too short"


def test_prompt_ids_are_unique():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["prompts"]]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"


def test_categories_cover_interview_breadth():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    categories = {p["category"] for p in data["prompts"]}
    # At minimum, we want these five to exist.
    required = {"self_intro", "motivation", "experience", "weakness", "technical"}
    assert required.issubset(categories), f"missing: {required - categories}"
