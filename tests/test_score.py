"""Parsing and the scorer wrapper, checked against hand-computed values."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import official  # noqa: E402
from src.score import aggregate, parse_records  # noqa: E402

pytestmark = pytest.mark.skipif(
    not official.REFERENCE_SRC.exists(), reason="reference implementation not cloned"
)


def rec(idx: int, response: str, cat: str, brand: str = "acme", used: bool = False) -> dict:
    return {
        "split": "val",
        "idx": idx,
        "response": response,
        "gt_category": cat,
        "gt_brand": brand,
        "gt_is_secondhand": used,
    }


def answer(cat: str, brand: str = "acme", used: bool = False) -> str:
    return json.dumps({"category": cat, "brand": brand, "is_secondhand": used})


def test_exact_and_partial_match() -> None:
    records = [
        rec(0, answer("A > B > C"), "A > B > C"),  # 3/3
        rec(1, answer("A > B > X"), "A > B > C"),  # 2/3 (prefix match stops at X)
        rec(2, answer("Z > B > C"), "A > B > C"),  # 0/3 (first level wrong)
        rec(3, answer("A > B"), "A > B > C"),  # 2 of pred 2, true 3
    ]
    rows = parse_records(records)
    assert [r["h_intersection"] for r in rows] == [3, 2, 0, 2]
    m = aggregate(rows)
    hp, hr = 7 / 11, 7 / 12
    assert m["category_hierarchical_f1"] == pytest.approx(2 * hp * hr / (hp + hr))
    assert m["category_exact_match"] == 0.25
    assert m["json_parse_rate"] == 1.0


def test_unparseable_output_is_a_complete_mismatch() -> None:
    bad = [
        "",
        "not json",
        "```json\n" + answer("A > B") + "\n```",  # fenced JSON is rejected, as in the reference
        json.dumps({"category": "A > B", "brand": "acme"}),  # missing field
        json.dumps({"category": "A > B", "brand": "acme", "is_secondhand": False, "x": 1}),
    ]
    rows = parse_records([rec(i, r, "A > B") for i, r in enumerate(bad)])
    assert not any(r["parsed"] for r in rows)
    assert all(r["h_intersection"] == 0 and r["h_pred_len"] == 2 for r in rows)
    m = aggregate(rows)
    assert m["category_hierarchical_f1"] == 0.0
    assert m["json_parse_rate"] == 0.0
    assert m["brand_score"] == 0.0


def test_brand_is_fuzzy_and_case_insensitive() -> None:
    rows = parse_records(
        [
            rec(0, answer("A", brand="  ACME "), "A", brand="acme"),
            rec(1, answer("A", brand="other"), "A", brand="acme"),
        ]
    )
    assert aggregate(rows)["brand_score"] == 0.5


def test_parsing_is_deterministic() -> None:
    records = [rec(i, "garbage", "A > B", used=bool(i % 2)) for i in range(20)]
    assert parse_records(records) == parse_records(records)
