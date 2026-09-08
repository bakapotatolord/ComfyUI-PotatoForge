from pathlib import Path
import unittest

from potatoforge_nodes.quant_patches.manifest import QuantPatchManifest
from potatoforge_nodes.quant_patches.merge import overlay_quant_patches
from potatoforge_nodes.quant_patches.stack import QuantPatchRef


def patch_ref(name: str, *families: str) -> QuantPatchRef:
    return QuantPatchRef(
        name=name,
        path=Path(name),
        manifest=QuantPatchManifest(name, families),
    )


def family(prefix: str, value: object) -> dict[str, object]:
    return {
        f"{prefix}.weight": value,
        f"{prefix}.weight_scale": value,
        f"{prefix}.comfy_quant": value,
    }


class QuantPatchMergeTests(unittest.TestCase):
    def test_replaces_only_declared_family_and_preserves_other_references(self):
        original_b_weight = object()
        original_bias = object()
        baseline = {
            **family("A", object()),
            "A.bias": original_bias,
            **family("B", original_b_weight),
        }
        patched_a_weight = object()

        report = overlay_quant_patches(
            baseline,
            [(patch_ref("patch-a", "A"), family("A", patched_a_weight))],
        )

        self.assertEqual(report.replacement_count, 1)
        self.assertIs(baseline["A.weight"], patched_a_weight)
        self.assertIs(baseline["A.bias"], original_bias)
        self.assertIs(baseline["B.weight"], original_b_weight)

    def test_later_patch_replaces_the_entire_overlapping_family(self):
        baseline = family("A", object())
        first = family("A", object())
        second_value = object()
        second = family("A", second_value)

        report = overlay_quant_patches(
            baseline,
            [(patch_ref("first", "A"), first), (patch_ref("second", "A"), second)],
        )

        self.assertEqual(len(report.conflicts), 1)
        self.assertEqual(report.conflicts[0].previous_patch_id, "first")
        self.assertIs(baseline["A.weight"], second_value)

    def test_rejects_a_patch_target_missing_from_the_baseline(self):
        baseline = family("A", object())

        with self.assertRaisesRegex(ValueError, "B.weight"):
            overlay_quant_patches(
                baseline,
                [(patch_ref("patch-b", "B"), family("B", object()))],
            )


if __name__ == "__main__":
    unittest.main()
