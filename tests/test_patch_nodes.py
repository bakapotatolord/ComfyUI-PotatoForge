from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from potatoforge_nodes.quant_patches import nodes
from potatoforge_nodes.quant_patches.manifest import QuantPatchManifest
from potatoforge_nodes.quant_patches.stack import QuantPatchRef, QuantPatchStack


class FakeFolderPaths:
    def __init__(self, root: Path):
        self.models_dir = str(root / "models")
        self.patch_path = root / "patch.safetensors"
        self.added: tuple[str, str] | None = None

    def add_model_folder_path(self, name, path):
        self.added = (name, path)

    def get_filename_list(self, name):
        self.assertEqual(name, "potatoforge_patches")
        return ["nested/valid.safetensors", "wrong.ckpt", "valid.sft"]

    def get_full_path_or_raise(self, name, filename):
        self.assertEqual(name, "potatoforge_patches")
        self.assertEqual(filename, "valid.safetensors")
        return str(self.patch_path)

    def assertEqual(self, left, right):
        if left != right:
            raise AssertionError(f"{left!r} != {right!r}")


class QuantPatchNodeTests(unittest.TestCase):
    def test_registers_dedicated_folder_and_builds_an_immutable_stack(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            folders = FakeFolderPaths(Path(temporary_directory))
            patch_ref = QuantPatchRef(
                "valid.safetensors",
                folders.patch_path,
                QuantPatchManifest("patch", ("A",)),
            )
            with patch.object(nodes, "_folder_paths", return_value=folders), patch.object(
                nodes, "inspect_quant_patch", return_value=patch_ref
            ):
                nodes.register_patch_folder()
                stack = nodes.PotatoForgeAddQuantPatch().add_patch("valid.safetensors", True)[0]
                names = nodes.available_patch_names()

            self.assertEqual(folders.added[0], "potatoforge_patches")
            self.assertTrue(Path(folders.added[1]).is_dir())
            self.assertEqual(stack.patches, (patch_ref,))
            self.assertEqual(names, ["nested/valid.safetensors"])

    def test_disabled_patch_preserves_the_input_stack(self):
        incoming = QuantPatchStack()

        result = nodes.PotatoForgeAddQuantPatch().add_patch("ignored.safetensors", False, incoming)

        self.assertIs(result[0], incoming)


if __name__ == "__main__":
    unittest.main()
