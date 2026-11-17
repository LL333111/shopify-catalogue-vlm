"""The request we send is the reference's, built from the original image bytes."""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import official  # noqa: E402
from src.data import build_request, load_split  # noqa: E402

pytestmark = pytest.mark.skipif(
    not official.REFERENCE_SRC.exists(), reason="reference implementation not cloned"
)


def make_sample() -> dict:
    buffer = BytesIO()
    Image.new("RGB", (40, 30), "red").save(buffer, format="PNG")
    buffer.seek(0)
    return {
        "product_title": "Red Mug",
        "product_description": "<p>A red mug</p>",
        "product_image": Image.open(buffer),
        "potential_product_categories": ["Home > Kitchen > Mugs", "Home > Decor"],
    }


def test_request_matches_reference_structure() -> None:
    request = build_request(make_sample())
    system, user = request["messages"]
    assert request["response_format"] is None  # guided decoding off, as in the reference
    assert system["role"] == "system" and '"is_secondhand"' in system["content"]
    text, image = user["content"]
    assert "Red Mug" in text["text"] and "Home > Kitchen > Mugs" in text["text"]
    url = image["image_url"]["url"]
    assert url.startswith("data:image/PNG;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.size == (40, 30)  # not resized


def test_test_split_is_guarded() -> None:
    with pytest.raises(PermissionError):
        load_split("test", {"dataset": {}})
