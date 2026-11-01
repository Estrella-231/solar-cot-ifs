from __future__ import annotations

import unittest

from droppath0_validation_manifest import select_validation_rows


def row(split: str, seq_id: str, frames: int = 24) -> dict[str, str]:
    return {"split": split, "seq_id": seq_id,
            "data_relpaths": "|".join(f"frame{i}" for i in range(frames))}


class ValidationManifestTest(unittest.TestCase):
    def test_preserves_base_bank_order_and_manifest_positions(self):
        manifest = [row("train", "t0"), row("val", "v0"), row("val", "v1"), row("val", "v2")]
        base = [{"split": "train", "seq_id": "t0"},
                {"split": "val", "seq_id": "v2"}, {"split": "val", "seq_id": "v0"}]
        selected, ids, indices = select_validation_rows(manifest, base)
        self.assertEqual(ids, ["v2", "v0"])
        self.assertEqual([x["seq_id"] for x in selected], ids)
        self.assertEqual(indices, [2, 0])

    def test_rejects_missing_or_malformed_candidate_keys(self):
        with self.assertRaisesRegex(ValueError, "lacks 1"):
            select_validation_rows([row("val", "v0")], [{"split": "val", "seq_id": "v1"}])
        with self.assertRaisesRegex(ValueError, "24 source"):
            select_validation_rows([row("val", "v0", 23)], [{"split": "val", "seq_id": "v0"}])


if __name__ == "__main__":
    unittest.main()
