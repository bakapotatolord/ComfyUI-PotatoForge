from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - the ComfyUI runtime provides Torch.
    torch = None

try:
    from safetensors.torch import load_file
except ImportError:  # pragma: no cover - the ComfyUI runtime provides Safetensors.
    load_file = None

from potatoforge_nodes import activation_calibration as calibration


_TorchModule = torch.nn.Module if torch is not None else object


@unittest.skipUnless(torch is not None, "Torch is required for activation calibration tests")
class ActivationCalibrationTests(unittest.TestCase):
    def test_accumulates_single_and_higher_dimensional_activations(self):
        stats = calibration.LayerActivationStats("block.weight", 3)

        stats.add(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertTrue(torch.equal(stats.finalize_to_cpu(), torch.tensor([17.0, 29.0, 45.0])))
        self.assertEqual(stats.sample_count, 2)
        self.assertEqual(stats.invocation_count, 1)

        higher_dimensional = calibration.LayerActivationStats("block.weight", 3)
        higher_dimensional.add(
            torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ]
            )
        )

        self.assertTrue(
            torch.equal(
                higher_dimensional.finalize_to_cpu(),
                torch.tensor([166.0, 214.0, 270.0]),
            )
        )
        self.assertEqual(higher_dimensional.sample_count, 4)
        self.assertEqual(higher_dimensional.invocation_count, 1)

    def test_multiple_invocations_add_raw_sums_and_counts(self):
        stats = calibration.LayerActivationStats("block.weight", 2)

        stats.add(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        stats.add(torch.tensor([[5.0, 6.0]]))

        self.assertTrue(torch.equal(stats.finalize_to_cpu(), torch.tensor([35.0, 56.0])))
        self.assertEqual(stats.sample_count, 3)
        self.assertEqual(stats.invocation_count, 2)

    def test_rejects_feature_mismatch(self):
        stats = calibration.LayerActivationStats("blocks.12.attn.wq.weight", 3)

        with self.assertRaisesRegex(ValueError, r"expected 3.*\(2, 4\)"):
            stats.add(torch.zeros(2, 4))

    def test_session_hooks_filter_and_cleanup(self):
        model = _TinyDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "test run",
                "int8_convrot",
                include_regex=r"block\.weight$|excluded\.weight$",
                exclude_regex=r"excluded\.weight$",
                output_directory=temporary_directory,
            )
            session.attach(model)

            self.assertEqual(tuple(session.layer_stats), ("block.weight",))
            model(torch.ones(2, 3))
            before_cleanup = session.layer_stats["block.weight"].finalize_to_cpu().clone()
            self.assertEqual(session.layer_stats["block.weight"].sample_count, 2)

            session.cleanup()
            session.cleanup()
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

            model(torch.ones(2, 3))
            self.assertTrue(
                torch.equal(
                    session.layer_stats["block.weight"].finalize_to_cpu(),
                    before_cleanup,
                )
            )

    def test_accepts_structural_linear_with_quantized_weight_and_shape_fallback(self):
        model = _StructuralDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "structural linear",
                "int8_convrot",
                output_directory=temporary_directory,
            )
            with self.assertLogs(calibration.LOGGER, level="INFO") as captured:
                session.attach(model)

            self.assertEqual(tuple(session.layer_stats), ("block.weight",))
            self.assertTrue(
                any(
                    "discovered linear layers=1 hooked layers=1 skipped layers=1" in message
                    for message in captured.output
                )
            )
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                r"block\.weight.*module class.*Linear.*weight shape \(2, 3\).*input shape \(2, 4\)",
            ):
                model(torch.ones(2, 4))
            session.cleanup()

    @unittest.skipUnless(load_file is not None, "Safetensors is required for serialization tests")
    def test_finalize_writes_safetensors_and_metadata_atomically(self):
        model = _TinyDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "calibration run",
                "bf16",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
            )
            session.attach(model)
            model(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

            stats_path, metadata_path = session.finalize()
            self.assertEqual((stats_path, metadata_path), session.finalize())
            self.assertFalse(session.active)
            self.assertTrue(stats_path.is_file())
            self.assertTrue(metadata_path.is_file())

            tensors = load_file(str(stats_path), device="cpu")
            self.assertEqual(tuple(tensors), ("block.weight.sum_x2",))
            self.assertEqual(tensors["block.weight.sum_x2"].dtype, torch.float32)
            self.assertTrue(
                torch.equal(tensors["block.weight.sum_x2"], torch.tensor([17.0, 29.0, 45.0]))
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["format"], "potatoforge_activation_calibration")
            self.assertEqual(metadata["layer_count"], 1)
            self.assertEqual(metadata["baseline_label"], "bf16")
            self.assertEqual(metadata["layers"]["block.weight"]["input_features"], 3)
            self.assertEqual(metadata["layers"]["block.weight"]["sample_count"], 2)
            self.assertEqual(metadata["layers"]["block.weight"]["invocation_count"], 1)

    def test_stale_session_is_cleaned_before_new_session(self):
        model = _TinyDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            old = calibration.ActivationCalibrationSession(
                "old",
                "bf16",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
            )
            old.attach(model)

            new = calibration.start_activation_calibration(
                SimpleNamespace(model=SimpleNamespace(diffusion_model=model)),
                session_name="new",
                baseline_label="int8_convrot",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
            )
            self.assertIsNotNone(new)
            self.assertFalse(old.active)

            model(torch.ones(1, 3))
            self.assertEqual(old.layer_stats["block.weight"].sample_count, 0)
            self.assertEqual(new.layer_stats["block.weight"].sample_count, 1)
            new.cleanup()

    def test_invalid_filters_and_missing_diffusion_model_fail_before_hooks(self):
        with self.assertRaisesRegex(ValueError, "include_regex"):
            calibration.start_activation_calibration(
                object(),
                include_regex="[",
                output_directory=tempfile.gettempdir(),
            )
        with self.assertRaisesRegex(RuntimeError, "could not locate"):
            calibration.start_activation_calibration(
                object(),
                output_directory=tempfile.gettempdir(),
            )

    def test_duplicate_canonical_keys_abort_attachment(self):
        model = _DuplicateNamedModules()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "duplicate",
                "bf16",
                output_directory=temporary_directory,
            )
            with self.assertRaisesRegex(ValueError, "duplicate canonical tensor key"):
                session.attach(model)
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

    def test_failed_save_detaches_hooks_and_does_not_leave_session_active(self):
        model = _TinyDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "failed save",
                "bf16",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
            )
            session.attach(model)
            with patch.object(
                calibration,
                "_save_calibration",
                side_effect=OSError("disk full"),
            ), self.assertRaisesRegex(OSError, "disk full"):
                session.finalize()
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

    def test_nodes_pass_model_and_latent_through(self):
        model = _TinyDiffusion()
        latent = {"samples": torch.zeros(1, 3)}
        with tempfile.TemporaryDirectory() as temporary_directory:
            folders = SimpleNamespace(get_output_directory=lambda: temporary_directory)
            with patch.object(calibration, "_folder_paths", return_value=folders):
                returned_model, session = calibration.PotatoForgeActivationCalibration().calibrate(
                    SimpleNamespace(model=SimpleNamespace(diffusion_model=model)),
                    "node test",
                    "int8_convrot",
                    r"block\.weight$",
                    "",
                    True,
                )

            self.assertIs(returned_model.model.diffusion_model, model)
            model(torch.ones(1, 3))
            returned_latent, stats_path, metadata_path = (
                calibration.PotatoForgeFinalizeActivationCalibration().finalize(latent, session)
            )
            self.assertIs(returned_latent, latent)
            self.assertTrue(Path(stats_path).is_file())
            self.assertTrue(Path(metadata_path).is_file())
            self.assertTrue(torch.isnan(torch.tensor(calibration.PotatoForgeActivationCalibration.IS_CHANGED())))
            self.assertEqual(
                calibration.PotatoForgeFinalizeActivationCalibration().finalize(latent, None),
                (latent, "", ""),
            )


class _TinyDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(3, 2, bias=False)
        self.excluded = torch.nn.Linear(3, 2, bias=False)

    def forward(self, activation):
        return self.block(activation) + self.excluded(activation)


class QuantizedTensor:
    def __init__(self, shape):
        self.shape = shape


class Linear(_TorchModule):
    def __init__(self):
        super().__init__()
        self.weight = QuantizedTensor((2, 3))

    def forward(self, activation):
        return activation


class _RankTwoLike(_TorchModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))


class _StructuralDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = Linear()
        self.arbitrary = _RankTwoLike()

    def forward(self, activation):
        return self.block(activation)


class _DuplicateNamedModules(_TorchModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)

    def named_modules(self):
        yield "", self
        yield "linear", self.linear
        yield "linear", self.linear


if __name__ == "__main__":
    unittest.main()
