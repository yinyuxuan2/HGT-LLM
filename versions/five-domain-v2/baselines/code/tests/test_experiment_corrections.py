import numpy as np
import unittest

from ours_min.protocol import sample_negatives
from scripts.prepare_uniform_prompt_nodes import uniform_text


def test_negative_sampling_uniform_and_reproducible():
    pairs = np.zeros((3000, 2), dtype=np.int64)
    observed = {0: {0, 1, 2}}
    values = sample_negatives(pairs, 1, 1000, observed, 100, 42)
    assert np.array_equal(values, sample_negatives(pairs, 1, 1000, observed, 100, 42))
    assert not np.isin(values[:, 1], [0, 1, 2]).any()
    # The old sort-and-truncate implementation centers near 270, not 501.
    assert 490 < values[:, 1].mean() < 510
    assert all(len(np.unique(row)) == 100 for row in values[:, 1].reshape(-1, 100))


def test_no_available_negative_raises_instead_of_hanging():
    with unittest.TestCase().assertRaises(ValueError):
        sample_negatives([[0, 0]], 1, 1, {0: {0}}, 1, 42)


def test_single_negative_rng_is_unchanged():
    pairs = np.column_stack([np.zeros(100, dtype=int), np.zeros(100, dtype=int)])
    expected = np.random.default_rng(42).integers(1000, size=100, dtype=np.int64)
    np.testing.assert_array_equal(sample_negatives(pairs, 1, 1000, {0: set()}, 1, 42)[:, 1], expected)


def test_uniform_prompt_preserves_category_semantics():
    row = {"id": "category:Accessories", "type": "category", "text":
           "Category: Accessories. Parent category: Electronics. Contains products like: Cable; Charger; Hub."}
    assert uniform_text(row) == "Text: Accessories; Electronics; Cable; Charger; Hub."
    assert uniform_text({"id": "item:1", "type": "item", "text":
           "Product: Cable. Description: USB cable."}) == "Text: Cable; USB cable."


def test_uniform_prompt_rejects_users():
    with unittest.TestCase().assertRaises(ValueError):
        uniform_text({"type": "user", "text": "anything"})


def test_fp16_can_change_item_ranking_but_fp32_roundtrip_does_not():
    import io
    original = np.array([1.0001, 1.0002], dtype=np.float32)
    assert original[1] > original[0]
    assert original.astype(np.float16)[1] == original.astype(np.float16)[0]
    buffer = io.BytesIO()
    np.save(buffer, original)
    buffer.seek(0)
    restored = np.load(buffer)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, original)


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("PASS", name, flush=True)
