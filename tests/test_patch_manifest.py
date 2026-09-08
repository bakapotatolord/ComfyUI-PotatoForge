import unittest

from potatoforge_nodes.quant_patches.manifest import (
    QuantPatchValidationError,
    parse_quant_patch_metadata,
    validate_patch_tensor_keys,
)


class QuantPatchManifestTests(unittest.TestCase):
    def metadata(self):
        return {
            "potatoforge_file_type": "quant_patch",
            "potatoforge_patch_format": "1",
            "potatoforge_patch_id": "tail-int6cr",
            "potatoforge_patch_replaces": '["blocks.27.mlp.down"]',
        }

    def test_parses_valid_metadata(self):
        manifest = parse_quant_patch_metadata(self.metadata(), "tail.patch.safetensors")

        self.assertEqual(manifest.patch_id, "tail-int6cr")
        self.assertEqual(manifest.replaces, ("blocks.27.mlp.down",))

    def test_rejects_invalid_required_metadata(self):
        invalid_metadata = [
            {},
            {**self.metadata(), "potatoforge_file_type": "model"},
            {**self.metadata(), "potatoforge_patch_format": "2"},
            {**self.metadata(), "potatoforge_patch_id": ""},
            {**self.metadata(), "potatoforge_patch_replaces": "not json"},
            {**self.metadata(), "potatoforge_patch_replaces": "[]"},
            {**self.metadata(), "potatoforge_patch_replaces": '["A", "A"]'},
            {**self.metadata(), "potatoforge_patch_replaces": '["A", 1]'},
        ]

        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(QuantPatchValidationError):
                parse_quant_patch_metadata(metadata, "bad.patch.safetensors")

    def test_rejects_missing_or_unexpected_family_tensors(self):
        manifest = parse_quant_patch_metadata(self.metadata(), "tail.patch.safetensors")

        with self.assertRaisesRegex(QuantPatchValidationError, "missing"):
            validate_patch_tensor_keys(manifest, ["blocks.27.mlp.down.weight"], "tail")
        with self.assertRaisesRegex(QuantPatchValidationError, "unexpected"):
            validate_patch_tensor_keys(
                manifest,
                [
                    "blocks.27.mlp.down.weight",
                    "blocks.27.mlp.down.weight_scale",
                    "blocks.27.mlp.down.comfy_quant",
                    "other.weight",
                ],
                "tail",
            )


if __name__ == "__main__":
    unittest.main()
