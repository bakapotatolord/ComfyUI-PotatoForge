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
    def test_v1_reduction_helpers_cover_higher_dimensional_inputs(self):
        activation = torch.tensor(
            [
                [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                [[-7.0, 8.0, 9.0], [10.0, -11.0, 12.0]],
            ]
        )

        self.assertTrue(
            torch.equal(
                calibration._per_feature_sum(activation),
                torch.tensor([8.0, 0.0, 18.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                calibration._per_feature_sum_of_squares(activation),
                torch.tensor([166.0, 214.0, 270.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                calibration._per_feature_max_abs(activation),
                torch.tensor([10.0, 11.0, 12.0]),
            )
        )
        self.assertEqual(calibration._sample_position_count(activation), 4)

    def test_multiple_evaluations_keep_order_and_aggregate(self):
        model = _KnownDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "multiple evaluations",
                "bf16",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
                sample_rows_per_evaluation=1,
            )
            session.attach(model)
            model(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            model(torch.tensor([[5.0, 6.0]]))

            stats = session.layer_stats["block.weight"]
            self.assertEqual(list(stats.completed_evaluations), [0, 1])
            self.assertTrue(
                torch.equal(
                    stats.completed_evaluations[0].sum_x2,
                    torch.tensor([10.0, 20.0]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stats.completed_evaluations[1].sum_x2,
                    torch.tensor([25.0, 36.0]),
                )
            )
            self.assertTrue(
                torch.equal(
                    stats.finalize_to_cpu(),
                    torch.tensor([35.0, 56.0]),
                )
            )
            self.assertEqual(len(session._evaluation_records), 2)
            session.cleanup()

    def test_repeated_linear_invocations_share_one_evaluation(self):
        model = _RepeatedDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "repeated",
                "bf16",
                include_regex=r"block\.weight$",
                output_directory=temporary_directory,
                sample_rows_per_evaluation=2,
            )
            session.attach(model)
            model(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

            completed = session.layer_stats["block.weight"].completed_evaluations[0]
            self.assertEqual(completed.sample_count, 4)
            self.assertEqual(completed.invocation_count, 2)
            self.assertTrue(torch.equal(completed.sum_x, torch.tensor([10.0, 14.0])))
            self.assertTrue(torch.equal(completed.sum_x2, torch.tensor([30.0, 54.0])))
            self.assertTrue(torch.equal(completed.max_abs_x, torch.tensor([4.0, 5.0])))
            self.assertTrue(torch.equal(completed.sum_y, torch.tensor([10.0, 14.0])))
            self.assertTrue(torch.equal(completed.sum_y2, torch.tensor([30.0, 54.0])))
            self.assertTrue(
                torch.equal(
                    completed.sample_x,
                    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                )
            )
            session.cleanup()

    def test_layer_absent_from_evaluation_is_zero_filled(self):
        model = _ConditionalDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "conditional",
                "bf16",
                output_directory=temporary_directory,
                sample_rows_per_evaluation=0,
            )
            session.attach(model)
            model(torch.ones(1, 2), use_b=True)
            model(torch.full((1, 2), 2.0), use_b=False)

            self.assertEqual(list(session.layer_stats["b.weight"].completed_evaluations), [0])
            tensors = session._materialize_tensors()
            self.assertTrue(
                torch.equal(
                    tensors["b.weight.eval_sum_x2"],
                    torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["b.weight.eval_invocation_count"],
                    torch.tensor([1, 0], dtype=torch.int64),
                )
            )
            session.cleanup()

    def test_output_shape_mismatch_aborts_and_detaches(self):
        model = _OutputMismatchDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "bad output",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                r"block\.weight.*expected output feature count 2.*output shape \(1, 1\)",
            ):
                model(torch.ones(1, 3))
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

    def test_sentinel_rows_are_deterministic_and_do_not_flatten(self):
        contiguous = torch.arange(12.0).reshape(2, 3, 2)
        noncontiguous = contiguous.transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        selected = calibration._select_sentinel_rows(noncontiguous, 3)
        self.assertTrue(
            torch.equal(
                selected,
                torch.tensor([[6.0, 7.0], [8.0, 9.0], [4.0, 5.0]]),
            )
        )
        self.assertEqual(tuple(calibration._select_sentinel_rows(contiguous, 0).shape), (0, 2))
        self.assertEqual(tuple(calibration._select_sentinel_rows(contiguous[:0], 2).shape), (0, 2))

    def test_sentinel_sampling_does_not_consume_torch_rng(self):
        activation = torch.arange(12.0).reshape(2, 3, 2)
        torch.manual_seed(1234)
        expected = torch.rand(4)
        torch.manual_seed(1234)
        calibration._select_sentinel_rows(activation, 2)
        actual = torch.rand(4)
        self.assertTrue(torch.equal(actual, expected))

    def test_timestep_metadata_is_best_effort(self):
        model = _TimeDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "time",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            model(torch.ones(1, 2), timestep=torch.tensor(0.95))
            record = session._evaluation_records[0]
            self.assertEqual(record.time_parameter_name, "timestep")
            self.assertAlmostEqual(record.time_value, 0.95, places=6)
            session.cleanup()

            no_time = _NoTimeDiffusion()
            no_time_session = calibration.ActivationCalibrationSession(
                "no time",
                "bf16",
                output_directory=temporary_directory,
            )
            no_time_session.attach(no_time)
            no_time(torch.ones(1, 2), arbitrary=torch.tensor(0.5))
            no_time_record = no_time_session._evaluation_records[0]
            self.assertIsNone(no_time_record.time_parameter_name)
            self.assertIsNone(no_time_record.time_value)
            no_time_session.cleanup()

    def test_nonfinite_statistics_abort_and_cleanup(self):
        model = _KnownDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "nan input",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                r"nonfinite statistic 'sum_x'",
            ):
                model(torch.tensor([[float("nan"), 1.0]]))
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

        model = _KnownDiffusion()
        with torch.no_grad():
            model.block.weight.fill_(float("nan"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "nan output",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                r"nonfinite statistic 'sum_y'",
            ):
                model(torch.ones(1, 2))
            self.assertFalse(session.active)

    def test_reentrant_root_forward_is_rejected(self):
        model = _ReentrantDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "reentrant",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                "reentrant",
            ):
                model(torch.ones(1, 2))
            self.assertFalse(session.active)
            self.assertEqual(session.hook_handles, ())

    def test_final_consistency_validation_rejects_tampered_evaluation(self):
        model = _KnownDiffusion()
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = calibration.ActivationCalibrationSession(
                "tampered",
                "bf16",
                output_directory=temporary_directory,
            )
            session.attach(model)
            model(torch.ones(1, 2))
            session.layer_stats["block.weight"].completed_evaluations[0].sum_x2.add_(1.0)
            with self.assertRaisesRegex(
                calibration.ActivationCalibrationError,
                "aggregate sum_x2 did not match",
            ):
                session.finalize()
            self.assertFalse(session.active)

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
            root_output = model(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

            stats_path, metadata_path = session.finalize()
            self.assertEqual((stats_path, metadata_path), session.finalize())
            self.assertFalse(session.active)
            self.assertTrue(stats_path.is_file())
            self.assertTrue(metadata_path.is_file())

            tensors = load_file(str(stats_path), device="cpu")
            self.assertEqual(
                set(tensors),
                {
                    "__pf__.root_input_sum_x2",
                    "__pf__.root_input_valid",
                    "__pf__.root_output_sum_y2",
                    "__pf__.root_output_valid",
                    "block.weight.sum_x2",
                    "block.weight.eval_sum_x",
                    "block.weight.eval_sum_x2",
                    "block.weight.eval_max_abs_x",
                    "block.weight.eval_sum_y",
                    "block.weight.eval_sum_y2",
                    "block.weight.eval_sample_count",
                    "block.weight.eval_invocation_count",
                    "block.weight.sample_x",
                    "block.weight.sample_x_valid",
                },
            )
            self.assertEqual(tensors["block.weight.sum_x2"].dtype, torch.float32)
            self.assertTrue(
                torch.equal(tensors["block.weight.sum_x2"], torch.tensor([17.0, 29.0, 45.0]))
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.eval_sum_x"],
                    torch.tensor([[5.0, 7.0, 9.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.eval_sum_x2"],
                    torch.tensor([[17.0, 29.0, 45.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.eval_max_abs_x"],
                    torch.tensor([[4.0, 5.0, 6.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.eval_sample_count"],
                    torch.tensor([2], dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.eval_invocation_count"],
                    torch.tensor([1], dtype=torch.int64),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["block.weight.sample_x"],
                    torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["__pf__.root_input_sum_x2"],
                    torch.tensor([91.0]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["__pf__.root_input_valid"],
                    torch.tensor([True]),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["__pf__.root_output_sum_y2"],
                    root_output.square().sum().reshape(1),
                )
            )
            self.assertTrue(
                torch.equal(
                    tensors["__pf__.root_output_valid"],
                    torch.tensor([True]),
                )
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["format"], "potatoforge_activation_calibration")
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["layer_count"], 1)
            self.assertEqual(metadata["evaluation_count"], 1)
            self.assertEqual(metadata["evaluation_basis"], "diffusion_model_forward")
            self.assertEqual(metadata["evaluations"][0]["evaluation_index"], 0)
            self.assertEqual(metadata["baseline_label"], "bf16")
            self.assertEqual(metadata["layers"]["block.weight"]["input_features"], 3)
            self.assertEqual(metadata["layers"]["block.weight"]["output_features"], 2)
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
        self.assertEqual(
            calibration.PotatoForgeActivationCalibration.INPUT_TYPES()["required"][
                "sample_rows_per_evaluation"
            ],
            ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
        )
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


class _KnownDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.block.weight.copy_(torch.eye(2))

    def forward(self, activation, **kwargs):
        return self.block(activation)


class _RepeatedDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.block.weight.copy_(torch.eye(2))

    def forward(self, activation):
        return self.block(activation) + self.block(activation + 1.0)


class _ConditionalDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 2, bias=False)
        self.b = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.a.weight.copy_(torch.eye(2))
            self.b.weight.copy_(2.0 * torch.eye(2))

    def forward(self, activation, use_b=True):
        result = self.a(activation)
        if use_b:
            result = result + self.b(activation)
        return result


class _OutputMismatchDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = _BadOutputLinear(3, 2, bias=False)

    def forward(self, activation):
        return self.block(activation)


class _BadOutputLinear(torch.nn.Linear):
    def forward(self, activation):
        return super().forward(activation)[..., :1]


class _TimeDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2, bias=False)

    def forward(self, activation, timestep, context=None):
        return self.block(activation)


class _NoTimeDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2, bias=False)

    def forward(self, activation, arbitrary=None):
        return self.block(activation)


class _ReentrantDiffusion(_TorchModule):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(2, 2, bias=False)
        self._entered = False

    def forward(self, activation):
        if not self._entered:
            self._entered = True
            self(activation)
        return self.block(activation)


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
        return activation[..., :2]


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
