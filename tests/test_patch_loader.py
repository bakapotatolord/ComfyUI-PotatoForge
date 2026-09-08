import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from potatoforge_nodes.quant_patches import loader
from potatoforge_nodes.quant_patches.stack import QuantPatchStack, inspect_quant_patch


def write_patch_header(path: Path, metadata: dict[str, str], keys: list[str]) -> None:
    header = {"__metadata__": metadata}
    header.update({key: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]} for key in keys})
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"x")


def patch_metadata(patch_id: str = "patch-a") -> dict[str, str]:
    return {
        "potatoforge_file_type": "quant_patch",
        "potatoforge_patch_format": "1",
        "potatoforge_patch_id": patch_id,
        "potatoforge_patch_replaces": '["A"]',
    }


class FakePatcher:
    pass


class QuantPatchLoaderTests(unittest.TestCase):
    def test_loads_baseline_then_overlay_and_preserves_reload_factory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            baseline_path = temporary_path / "baseline.safetensors"
            patch_path = temporary_path / "patch.safetensors"
            baseline_path.touch()
            patch_keys = ["A.weight", "A.weight_scale", "A.comfy_quant"]
            write_patch_header(patch_path, patch_metadata(), patch_keys)
            patch_stack = QuantPatchStack().append(inspect_quant_patch(patch_path.name, patch_path))

            baseline_weight = object()
            patch_weight = object()
            baseline = {"A.weight": baseline_weight, "B.weight": object()}
            patch_payload = {
                "A.weight": patch_weight,
                "A.weight_scale": object(),
                "A.comfy_quant": object(),
            }
            calls: list[str] = []
            captured: dict[str, object] = {}

            class FakeUtils:
                @staticmethod
                def load_torch_file(path, safe_load=False, return_metadata=False):
                    calls.append(Path(path).name)
                    if Path(path) == baseline_path:
                        return baseline, {"baseline": "metadata"}
                    return patch_payload, patch_metadata()

            class FakeSD:
                @staticmethod
                def load_diffusion_model_state_dict(state_dict, model_options, metadata, disable_dynamic):
                    captured.update(
                        state_dict=state_dict,
                        model_options=model_options,
                        metadata=metadata,
                        disable_dynamic=disable_dynamic,
                    )
                    return FakePatcher()

                @staticmethod
                def load_diffusion_model(*args, **kwargs):
                    raise AssertionError("patched stacks must use the state-dict loader")

            with patch.object(loader, "_comfy_modules", return_value=(FakeSD, FakeUtils)):
                patcher = loader.load_patched_diffusion_model(
                    str(baseline_path), patch_stack, {"option": "value"}
                )

                reloaded = patcher.cached_patcher_init[0](
                    *patcher.cached_patcher_init[1], disable_dynamic=True
                )

            self.assertEqual(calls, ["baseline.safetensors", "patch.safetensors", "baseline.safetensors", "patch.safetensors"])
            self.assertIs(captured["state_dict"], baseline)
            self.assertIs(baseline["A.weight"], patch_weight)
            self.assertEqual(captured["metadata"], {"baseline": "metadata"})
            self.assertTrue(captured["disable_dynamic"])
            self.assertIsInstance(reloaded, FakePatcher)

    def test_uses_the_stock_loader_when_no_patches_are_connected(self):
        class FakeSD:
            @staticmethod
            def load_diffusion_model(path, model_options, disable_dynamic):
                return (path, model_options, disable_dynamic)

        class FakeUtils:
            pass

        with patch.object(loader, "_comfy_modules", return_value=(FakeSD, FakeUtils)):
            result = loader.load_patched_diffusion_model("baseline", None, {"dtype": "x"})

        self.assertEqual(result, ("baseline", {"dtype": "x"}, False))


if __name__ == "__main__":
    unittest.main()
