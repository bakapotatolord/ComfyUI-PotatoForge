from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
import logging
import math
from numbers import Number
import os
from pathlib import Path
import re
import uuid
import weakref
from typing import Any


LOGGER = logging.getLogger("potatoforge")

CALIBRATION_SESSION_TYPE = "POTATOFORGE_CALIBRATION_SESSION"
CALIBRATION_FORMAT = "potatoforge_activation_calibration"
CALIBRATION_VERSION = 1
CALIBRATION_DIRECTORY = "potatoforge_calibration"
_ROOT_INPUT_SUM_X2_KEY = "__pf__.root_input_sum_x2"
_ROOT_INPUT_VALID_KEY = "__pf__.root_input_valid"
_ROOT_OUTPUT_SUM_Y2_KEY = "__pf__.root_output_sum_y2"
_ROOT_OUTPUT_VALID_KEY = "__pf__.root_output_valid"


def _torch() -> Any:
    import torch

    return torch


def _folder_paths() -> Any:
    import folder_paths

    return folder_paths


class ActivationCalibrationError(ValueError):
    """Raised when a calibration input or artifact violates the V1 contract."""


def _qualified_class_name(value: object) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _compile_filters(
    include_regex: str,
    exclude_regex: str,
) -> tuple[str, str, Any, Any | None]:
    if not isinstance(include_regex, str) or not isinstance(exclude_regex, str):
        raise TypeError("Activation calibration regex filters must be strings.")

    include_source = include_regex or ".*"
    exclude_source = exclude_regex or ""
    try:
        include_pattern = re.compile(include_source)
    except re.error as error:
        raise ValueError(f"Invalid activation calibration include_regex: {error}.") from error

    exclude_pattern = None
    if exclude_source:
        try:
            exclude_pattern = re.compile(exclude_source)
        except re.error as error:
            raise ValueError(f"Invalid activation calibration exclude_regex: {error}.") from error

    return include_source, exclude_source, include_pattern, exclude_pattern


def canonical_weight_name(module_name: str) -> str:
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("A named module is required for canonical tensor naming.")
    return f"{module_name}.weight"


def _weight_shape(weight: Any) -> tuple[Any, ...] | None:
    if weight is None:
        return None
    shape = getattr(weight, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except (TypeError, ValueError):
        return None


def _require_tensor_with_feature_axis(activation: Any) -> Any:
    torch = _torch()
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Activation calibration expected a torch.Tensor input.")
    if activation.ndim < 1:
        raise ActivationCalibrationError(
            f"Activation calibration expected at least one dimension, got shape {tuple(activation.shape)}."
        )
    return activation.detach()


def _sample_position_count(activation: Any) -> int:
    detached = _require_tensor_with_feature_axis(activation)
    return math.prod(int(dimension) for dimension in detached.shape[:-1])


def _per_feature_sum(activation: Any) -> Any:
    torch = _torch()
    detached = _require_tensor_with_feature_axis(activation)
    math_activation = detached.to(dtype=torch.float32)
    reduction_dims = tuple(range(detached.ndim - 1))
    if not reduction_dims:
        return math_activation.contiguous()
    return math_activation.sum(dim=reduction_dims, dtype=torch.float32).contiguous()


def _per_feature_sum_of_squares(activation: Any) -> Any:
    torch = _torch()
    detached = _require_tensor_with_feature_axis(activation)
    math_activation = detached.to(dtype=torch.float32)
    reduction_dims = tuple(range(detached.ndim - 1))
    if not reduction_dims:
        return math_activation.square().contiguous()
    return math_activation.square().sum(dim=reduction_dims, dtype=torch.float32).contiguous()


def _per_feature_max_abs(activation: Any) -> Any:
    torch = _torch()
    detached = _require_tensor_with_feature_axis(activation)
    if _sample_position_count(detached) == 0:
        return torch.zeros(
            int(detached.shape[-1]),
            dtype=torch.float32,
            device=detached.device,
        )
    math_activation = detached.to(dtype=torch.float32)
    reduction_dims = tuple(range(detached.ndim - 1))
    if not reduction_dims:
        return math_activation.abs().contiguous()
    return math_activation.abs().amax(dim=reduction_dims).contiguous()


def _select_sentinel_rows(activation: Any, needed_count: int) -> Any:
    """Select deterministic rows without flattening the complete activation."""
    torch = _torch()
    detached = _require_tensor_with_feature_axis(activation)
    if type(needed_count) is not int or needed_count < 0:
        raise ValueError("Activation calibration needed_count must be a non-negative integer.")

    feature_count = int(detached.shape[-1])
    position_count = _sample_position_count(detached)
    take_count = min(needed_count, position_count)
    if take_count == 0:
        return torch.zeros((0, feature_count), dtype=torch.float32)

    leading_shape = tuple(int(dimension) for dimension in detached.shape[:-1])
    rows: list[Any] = []
    for row_number in range(take_count):
        flat_position = min(
            position_count - 1,
            ((row_number + 1) * position_count) // (take_count + 1),
        )
        remainder = flat_position
        coordinates = [0] * len(leading_shape)
        for axis in range(len(leading_shape) - 1, -1, -1):
            dimension = leading_shape[axis]
            coordinates[axis] = remainder % dimension
            remainder //= dimension
        row = detached[tuple(coordinates) + (slice(None),)]
        rows.append(row.to(device="cpu", dtype=torch.float32).contiguous())
    return torch.stack(rows, dim=0).contiguous()


def per_feature_sum_of_squares(activation: Any) -> Any:
    """Return FP32 per-last-dimension sums without retaining the activation."""
    return _per_feature_sum_of_squares(activation)


def _validate_finite_statistic(
    value: Any,
    *,
    session_id: str,
    evaluation_index: int,
    tensor_name: str,
    statistic_name: str,
) -> None:
    torch = _torch()
    if not bool(torch.isfinite(value).all().item()):
        raise ActivationCalibrationError(
            f"Activation calibration session '{session_id}' evaluation {evaluation_index} "
            f"layer '{tensor_name}' produced nonfinite statistic '{statistic_name}' "
            f"with shape {tuple(value.shape)} and dtype {value.dtype}."
        )


@dataclass(frozen=True)
class CompletedLayerEvaluation:
    sum_x: Any
    sum_x2: Any
    max_abs_x: Any
    sum_y: Any
    sum_y2: Any
    sample_count: int
    invocation_count: int
    sample_x: Any


@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_index: int
    time_parameter_name: str | None
    time_value: Any
    time_value_truncated: bool
    root_input_sum_x2: float
    root_input_valid: bool
    root_output_sum_y2: float
    root_output_valid: bool


@dataclass
class EvaluationContext:
    evaluation_index: int
    time_parameter_name: str | None
    time_value: Any
    time_value_truncated: bool
    root_input_sum_x2: float
    root_input_valid: bool
    layer_accumulators: dict[str, LayerEvaluationAccumulator] = field(default_factory=dict)


class LayerEvaluationAccumulator:
    """Collect one layer's current evaluation and release it to CPU on close."""

    def __init__(
        self,
        tensor_name: str,
        input_features: int,
        output_features: int,
        sample_rows_per_evaluation: int,
    ) -> None:
        self.tensor_name = tensor_name
        self.input_features = input_features
        self.output_features = output_features
        self.sample_rows_per_evaluation = sample_rows_per_evaluation
        self.sum_x_by_device: dict[str, Any] = {}
        self.sum_x2_by_device: dict[str, Any] = {}
        self.max_abs_x_by_device: dict[str, Any] = {}
        self.sum_y_by_device: dict[str, Any] = {}
        self.sum_y2_by_device: dict[str, Any] = {}
        self.sample_count = 0
        self.invocation_count = 0
        self.output_invocation_count = 0
        self._pending_sample_counts: list[int] = []
        self.sample_rows_cpu: list[Any] = []

    @staticmethod
    def _accumulate(
        destination: dict[str, Any],
        value: Any,
        *,
        maximum: bool = False,
    ) -> None:
        torch = _torch()
        value = value.detach().to(dtype=torch.float32)
        device_key = str(value.device)
        existing = destination.get(device_key)
        if existing is None:
            destination[device_key] = value.contiguous().clone()
        elif maximum:
            torch.maximum(existing, value, out=existing)
        else:
            existing.add_(value)

    @staticmethod
    def _to_cpu(
        source: dict[str, Any],
        feature_count: int,
        *,
        maximum: bool = False,
    ) -> Any:
        torch = _torch()
        result = torch.zeros(feature_count, dtype=torch.float32)
        for partial in source.values():
            partial_cpu = partial.detach().to(device="cpu", dtype=torch.float32)
            if maximum:
                torch.maximum(result, partial_cpu, out=result)
            else:
                result.add_(partial_cpu)
        return result.contiguous()

    def add_input(self, activation: Any) -> tuple[Any, Any, Any, int]:
        sum_x = _per_feature_sum(activation)
        sum_x2 = _per_feature_sum_of_squares(activation)
        max_abs_x = _per_feature_max_abs(activation)
        sample_count = _sample_position_count(activation)
        self._accumulate(self.sum_x_by_device, sum_x)
        self._accumulate(self.sum_x2_by_device, sum_x2)
        self._accumulate(self.max_abs_x_by_device, max_abs_x, maximum=True)
        self.sample_count += sample_count
        self.invocation_count += 1
        self._pending_sample_counts.append(sample_count)

        if len(self.sample_rows_cpu) < self.sample_rows_per_evaluation:
            selected = _select_sentinel_rows(
                activation,
                self.sample_rows_per_evaluation - len(self.sample_rows_cpu),
            )
            self.sample_rows_cpu.extend(selected)
        return sum_x, sum_x2, max_abs_x, sample_count

    def add_output(self, output: Any) -> tuple[Any, Any]:
        if not self._pending_sample_counts:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{self.tensor_name}' received an output "
                "without a pending input invocation."
            )
        expected_sample_count = self._pending_sample_counts[-1]
        output_sample_count = _sample_position_count(output)
        if output_sample_count != expected_sample_count:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{self.tensor_name}' output position count "
                f"{output_sample_count} did not match input position count {expected_sample_count}."
            )
        sum_y = _per_feature_sum(output)
        sum_y2 = _per_feature_sum_of_squares(output)
        self._accumulate(self.sum_y_by_device, sum_y)
        self._accumulate(self.sum_y2_by_device, sum_y2)
        self._pending_sample_counts.pop()
        self.output_invocation_count += 1
        return sum_y, sum_y2

    def finalize_to_cpu(self) -> CompletedLayerEvaluation:
        torch = _torch()
        if self._pending_sample_counts:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{self.tensor_name}' has "
                f"{len(self._pending_sample_counts)} pending Linear invocations."
            )
        if self.output_invocation_count != self.invocation_count:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{self.tensor_name}' recorded "
                f"{self.invocation_count} inputs but {self.output_invocation_count} outputs."
            )
        if self.sample_rows_cpu:
            sample_x = torch.stack(self.sample_rows_cpu, dim=0).contiguous()
        else:
            sample_x = torch.zeros((0, self.input_features), dtype=torch.float32)
        completed = CompletedLayerEvaluation(
            sum_x=self._to_cpu(self.sum_x_by_device, self.input_features),
            sum_x2=self._to_cpu(self.sum_x2_by_device, self.input_features),
            max_abs_x=self._to_cpu(
                self.max_abs_x_by_device,
                self.input_features,
                maximum=True,
            ),
            sum_y=self._to_cpu(self.sum_y_by_device, self.output_features),
            sum_y2=self._to_cpu(self.sum_y2_by_device, self.output_features),
            sample_count=self.sample_count,
            invocation_count=self.invocation_count,
            sample_x=sample_x,
        )
        self.sum_x_by_device = {}
        self.sum_x2_by_device = {}
        self.max_abs_x_by_device = {}
        self.sum_y_by_device = {}
        self.sum_y2_by_device = {}
        self._pending_sample_counts = []
        self.sample_rows_cpu = []
        return completed


class LayerActivationStats:
    """Raw aggregate and sparse per-evaluation statistics for one logical weight."""

    def __init__(
        self,
        tensor_name: str,
        input_features: int,
        output_features: int | None = None,
    ) -> None:
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Layer activation statistics require a tensor name.")
        if type(input_features) is not int or input_features < 1:
            raise ValueError("Layer activation statistics require positive input features.")
        if output_features is None:
            output_features = input_features
        if type(output_features) is not int or output_features < 1:
            raise ValueError("Layer activation statistics require positive output features.")

        self.tensor_name = tensor_name
        self.input_features = input_features
        self.output_features = output_features
        self.sample_count = 0
        self.invocation_count = 0
        self.sum_x2_by_device: dict[str, Any] = {}
        self.completed_evaluations: dict[int, CompletedLayerEvaluation] = {}

    def add(self, activation: Any) -> None:
        torch = _torch()
        if not isinstance(activation, torch.Tensor):
            raise TypeError(
                f"Layer '{self.tensor_name}' expected a torch.Tensor activation."
            )
        if activation.ndim < 1:
            raise ActivationCalibrationError(
                f"Layer '{self.tensor_name}' expected an activation with a last dimension; "
                f"observed shape {tuple(activation.shape)}."
            )
        if activation.shape[-1] != self.input_features:
            raise ActivationCalibrationError(
                f"Layer '{self.tensor_name}' expected {self.input_features} input features, "
                f"observed shape {tuple(activation.shape)}."
            )

        contribution = per_feature_sum_of_squares(activation)
        sample_count = _sample_position_count(activation)
        self.add_contribution(contribution, sample_count)

    def add_contribution(
        self,
        contribution: Any,
        sample_count: int,
        invocation_count: int = 1,
    ) -> None:
        torch = _torch()
        if not isinstance(contribution, torch.Tensor):
            raise TypeError("Activation calibration contributions must be torch.Tensor values.")
        if contribution.ndim != 1 or contribution.shape[0] != self.input_features:
            raise ActivationCalibrationError(
                f"Layer '{self.tensor_name}' expected a contribution shaped "
                f"[{self.input_features}], observed {tuple(contribution.shape)}."
            )
        if type(sample_count) is not int or sample_count < 0:
            raise ValueError("Activation calibration sample_count must be a non-negative integer.")
        if type(invocation_count) is not int or invocation_count < 0:
            raise ValueError(
                "Activation calibration invocation_count must be a non-negative integer."
            )

        contribution = contribution.detach()
        if contribution.dtype != torch.float32:
            contribution = contribution.to(dtype=torch.float32)
        if not contribution.is_contiguous():
            contribution = contribution.contiguous()
        if not bool(torch.isfinite(contribution).all().item()):
            raise ActivationCalibrationError(
                f"Layer '{self.tensor_name}' produced nonfinite statistic 'sum_x2' "
                f"with shape {tuple(contribution.shape)} and dtype {contribution.dtype}."
            )

        device_key = str(contribution.device)
        existing = self.sum_x2_by_device.get(device_key)
        if existing is None:
            self.sum_x2_by_device[device_key] = contribution.clone()
        else:
            existing.add_(contribution)
        self.sample_count += sample_count
        self.invocation_count += invocation_count

    def finalize_to_cpu(self) -> Any:
        torch = _torch()
        total = torch.zeros(self.input_features, dtype=torch.float32)
        for partial in self.sum_x2_by_device.values():
            total.add_(partial.detach().to(device="cpu", dtype=torch.float32))
        total = total.contiguous()
        self.sum_x2_by_device = {"cpu": total}
        return total


def _discover_linear_layers(
    diffusion_model: Any,
    include_pattern: Any,
    exclude_pattern: Any | None,
) -> tuple[list[tuple[str, Any, int, tuple[Any, ...]]], int, int]:
    torch = _torch()
    if not isinstance(diffusion_model, torch.nn.Module):
        raise TypeError("Activation calibration requires a torch diffusion model module.")

    selected: list[tuple[str, Any, int, tuple[Any, ...]]] = []
    seen_weight_names: set[str] = set()
    discovered_count = 0
    skipped_count = 0
    for module_name, module in diffusion_model.named_modules():
        if not module_name:
            continue
        weight_shape = _weight_shape(getattr(module, "weight", None))
        if weight_shape is None or len(weight_shape) != 2:
            continue
        if not (
            isinstance(module, torch.nn.Linear)
            or module.__class__.__name__ == "Linear"
        ):
            skipped_count += 1
            continue

        discovered_count += 1

        tensor_name = canonical_weight_name(module_name)
        if tensor_name in seen_weight_names:
            raise ActivationCalibrationError(
                f"Activation calibration discovered duplicate canonical tensor key '{tensor_name}'."
            )
        seen_weight_names.add(tensor_name)

        input_features = getattr(module, "in_features", None)
        if input_features is None:
            input_features = weight_shape[1]
        if type(input_features) is not int or input_features < 1:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{tensor_name}' has invalid input feature count "
                f"{input_features!r} from weight shape {weight_shape!r}."
            )
        if not include_pattern.search(tensor_name):
            skipped_count += 1
            continue
        if exclude_pattern is not None and exclude_pattern.search(tensor_name):
            skipped_count += 1
            continue
        selected.append((tensor_name, module, input_features, weight_shape))

    return selected, discovered_count, skipped_count


def _extract_first_tensor_input(args: object) -> Any:
    if not isinstance(args, (tuple, list)) or not args:
        raise ActivationCalibrationError(
            "Activation calibration hook received no positional tensor input."
        )
    return args[0]


_TIME_PARAMETER_NAMES = ("timestep", "timesteps", "sigma", "sigmas", "t")


def _serialize_time_value(value: Any) -> tuple[Any, bool]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None, False
        detached = value.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(detached).all().item()):
            return None, False
        flat = detached.reshape(-1)
        if bool(torch.all(flat == flat[0]).item()):
            return float(flat[0].item()), False
        if flat.numel() <= 16:
            return [float(item) for item in flat.to(device="cpu").tolist()], False
        return None, True
    if isinstance(value, Number):
        numeric = value.item() if hasattr(value, "item") else value
        if isinstance(numeric, float) and not math.isfinite(numeric):
            return None, False
        return numeric, False
    if isinstance(value, (tuple, list)) and len(value) <= 16:
        serialized: list[Any] = []
        for item in value:
            if not isinstance(item, Number):
                return None, False
            numeric = item.item() if hasattr(item, "item") else item
            if isinstance(numeric, float) and not math.isfinite(numeric):
                return None, False
            serialized.append(numeric)
        return serialized, False
    if isinstance(value, (tuple, list)):
        return None, True
    return None, False


def _extract_evaluation_time_value(
    module: Any,
    args: object,
    kwargs: object,
) -> tuple[str | None, Any, bool]:
    normalized_kwargs = kwargs if isinstance(kwargs, dict) else {}
    candidates: dict[str, Any] = dict(normalized_kwargs)
    try:
        signature = inspect.signature(module.forward)
        bound = signature.bind_partial(*tuple(args), **normalized_kwargs)
        candidates.update(bound.arguments)
    except (TypeError, ValueError):
        pass

    for name in _TIME_PARAMETER_NAMES:
        if name in candidates:
            time_value, truncated = _serialize_time_value(candidates[name])
            return name, time_value, truncated
    return None, None, False


def _optional_tensor_energy(value: Any) -> tuple[float, bool]:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        return 0.0, False
    try:
        math_value = value.detach().to(dtype=torch.float32)
        energy = math_value.square().sum(dtype=torch.float32)
        if not bool(torch.isfinite(energy).item()):
            return 0.0, False
        return float(energy.item()), True
    except Exception:
        return 0.0, False


def _root_output_tensor(output: Any) -> Any:
    torch = _torch()
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) == 1:
        candidate = output[0]
        if isinstance(candidate, torch.Tensor):
            return candidate
    return None


def _optional_root_input_energy(args: object) -> tuple[float, bool]:
    if not isinstance(args, (tuple, list)) or not args:
        return 0.0, False
    return _optional_tensor_energy(args[0])


def _optional_root_output_energy(output: Any) -> tuple[float, bool]:
    return _optional_tensor_energy(_root_output_tensor(output))


def sanitize_session_name(session_name: str) -> str:
    if not isinstance(session_name, str):
        raise TypeError("Activation calibration session_name must be a string.")
    requested_name = session_name.strip()
    if not requested_name:
        raise ValueError("Activation calibration session_name must not be empty.")
    if (
        Path(requested_name).is_absolute()
        or "/" in requested_name
        or "\\" in requested_name
        or ".." in requested_name
    ):
        raise ValueError(
            "Activation calibration session_name must not contain path separators, '..', or an absolute path."
        )

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", requested_name).strip("._-")
    if not safe_name:
        raise ValueError("Activation calibration session_name contains no usable filename characters.")
    return safe_name


def _prepare_output_directory(output_directory: str | Path) -> Path:
    directory = Path(output_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"PotatoForge activation calibration output directory '{directory}' could not be prepared: {error}."
        ) from error
    if not directory.is_dir():
        raise RuntimeError(
            f"PotatoForge activation calibration output path '{directory}' is not a directory."
        )
    return directory


def calibration_output_directory() -> Path:
    folder_paths = _folder_paths()
    try:
        output_directory = folder_paths.get_output_directory()
    except AttributeError as error:
        raise RuntimeError(
            "PotatoForge activation calibration requires ComfyUI folder_paths.get_output_directory()."
        ) from error
    return _prepare_output_directory(Path(output_directory) / CALIBRATION_DIRECTORY)


_ACTIVE_SESSIONS: weakref.WeakSet[Any] = weakref.WeakSet()


def _cleanup_stale_sessions() -> None:
    for session in tuple(_ACTIVE_SESSIONS):
        if not session.active:
            continue
        LOGGER.warning(
            "[PotatoForge] Cleaning stale activation calibration hooks from aborted session=%s",
            session.session_id,
        )
        session.cleanup()


def _make_forward_pre_hook(
    session: ActivationCalibrationSession,
    tensor_name: str,
    input_features: int,
    weight_shape: tuple[Any, ...],
) -> Any:
    def hook(module: Any, args: object) -> None:
        try:
            activation = _extract_first_tensor_input(args)
            session.record_activation(
                tensor_name,
                input_features,
                weight_shape,
                activation,
                module,
            )
        except BaseException:
            session.cleanup()
            raise

    return hook


def _make_forward_hook(
    session: ActivationCalibrationSession,
    tensor_name: str,
    output_features: int,
    weight_shape: tuple[Any, ...],
) -> Any:
    def hook(module: Any, args: object, output: Any) -> Any:
        try:
            session.record_output(
                tensor_name,
                output_features,
                weight_shape,
                output,
                module,
            )
        except BaseException:
            session.cleanup()
            raise
        return output

    return hook


def _make_root_forward_pre_hook(session: ActivationCalibrationSession) -> Any:
    def hook(module: Any, args: object, kwargs: object | None = None) -> None:
        try:
            session.begin_evaluation(module, args, kwargs or {})
        except BaseException:
            session.cleanup()
            raise

    return hook


def _make_root_forward_hook(session: ActivationCalibrationSession) -> Any:
    def hook(
        module: Any,
        args: object,
        kwargs: object,
        output: Any,
    ) -> Any:
        try:
            session.finish_evaluation(module, args, kwargs, output)
        except BaseException:
            session.cleanup()
            raise
        return output

    return hook


def _make_root_forward_hook_without_kwargs(session: ActivationCalibrationSession) -> Any:
    def hook(module: Any, args: object, output: Any) -> Any:
        try:
            session.finish_evaluation(module, args, {}, output)
        except BaseException:
            session.cleanup()
            raise
        return output

    return hook


def _register_root_pre_hook(module: Any, hook: Any) -> Any:
    try:
        return module.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(
            lambda root_module, args: hook(root_module, args, {})
        )


def _register_root_forward_hook(module: Any, hook: Any, fallback_hook: Any) -> Any:
    try:
        return module.register_forward_hook(hook, with_kwargs=True, always_call=True)
    except TypeError:
        try:
            return module.register_forward_hook(fallback_hook, always_call=True)
        except TypeError:
            return module.register_forward_hook(fallback_hook)


class ActivationCalibrationSession:
    """Owns one calibration run's statistics and forward-hook lifecycle."""

    def __init__(
        self,
        session_name: str,
        baseline_label: str,
        include_regex: str = ".*",
        exclude_regex: str = "",
        output_directory: str | Path = ".",
        diffusion_model_class: str = "unknown",
        session_id: str | None = None,
        sample_rows_per_evaluation: int = 2,
    ) -> None:
        include_source, exclude_source, include_pattern, exclude_pattern = _compile_filters(
            include_regex,
            exclude_regex,
        )
        self.session_name = session_name.strip() if isinstance(session_name, str) else session_name
        self.baseline_label = baseline_label.strip() if isinstance(baseline_label, str) else baseline_label
        if not isinstance(self.session_name, str) or not self.session_name:
            raise ValueError("Activation calibration session_name must not be empty.")
        if not isinstance(self.baseline_label, str):
            raise TypeError("Activation calibration baseline_label must be a string.")
        if (
            type(sample_rows_per_evaluation) is not int
            or not 0 <= sample_rows_per_evaluation <= 8
        ):
            raise ValueError(
                "Activation calibration sample_rows_per_evaluation must be an integer from 0 to 8."
            )
        sanitize_session_name(self.session_name)
        self.include_regex = include_source
        self.exclude_regex = exclude_source
        self._include_pattern = include_pattern
        self._exclude_pattern = exclude_pattern
        self.output_directory = _prepare_output_directory(output_directory)
        self.diffusion_model_class = diffusion_model_class
        self.session_id = session_id or uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.sample_rows_per_evaluation = sample_rows_per_evaluation
        self.layer_stats: dict[str, LayerActivationStats] = {}
        self.stats = self.layer_stats
        self._hook_handles: list[Any] = []
        self._state = "active"
        self._saved_paths: tuple[Path, Path] | None = None
        self._active_evaluation: EvaluationContext | None = None
        self._next_evaluation_index = 0
        self._evaluation_records: list[EvaluationRecord] = []
        _ACTIVE_SESSIONS.add(self)

    @property
    def active(self) -> bool:
        return self._state == "active"

    @property
    def closed(self) -> bool:
        return self._state in {"finalized", "aborted"}

    @property
    def hook_handles(self) -> tuple[Any, ...]:
        return tuple(self._hook_handles)

    def attach(self, diffusion_model: Any) -> None:
        if not self.active:
            raise RuntimeError(
                f"Activation calibration session '{self.session_id}' is no longer active."
            )
        if self._hook_handles:
            raise RuntimeError(
                f"Activation calibration session '{self.session_id}' is already attached."
            )

        try:
            selected, discovered_count, skipped_count = _discover_linear_layers(
                diffusion_model,
                self._include_pattern,
                self._exclude_pattern,
            )
        except BaseException:
            self.cleanup()
            raise
        if not selected:
            self.cleanup()
            LOGGER.warning(
                "[PotatoForge] Activation calibration found no matching diffusion Linear layers"
            )
            raise ValueError(
                "PotatoForge activation calibration found no supported diffusion Linear layers "
                "matching the configured filters."
            )

        try:
            for tensor_name, _module, input_features, _weight_shape in selected:
                self.layer_stats[tensor_name] = LayerActivationStats(
                    tensor_name,
                    input_features,
                    int(_weight_shape[0]),
                )
            self._hook_handles.append(
                _register_root_pre_hook(
                    diffusion_model,
                    _make_root_forward_pre_hook(self),
                )
            )
            self._hook_handles.append(
                _register_root_forward_hook(
                    diffusion_model,
                    _make_root_forward_hook(self),
                    _make_root_forward_hook_without_kwargs(self),
                )
            )
            for tensor_name, module, input_features, weight_shape in selected:
                self._hook_handles.append(
                    module.register_forward_pre_hook(
                        _make_forward_pre_hook(
                            self,
                            tensor_name,
                            input_features,
                            weight_shape,
                        )
                    )
                )
                self._hook_handles.append(
                    module.register_forward_hook(
                        _make_forward_hook(
                            self,
                            tensor_name,
                            int(weight_shape[0]),
                            weight_shape,
                        )
                    )
                )
        except BaseException:
            self.cleanup()
            raise

        LOGGER.info(
            "[PotatoForge] Activation calibration started session=%s baseline=%s "
            "discovered linear layers=%d hooked layers=%d skipped layers=%d "
            "sample_rows_per_evaluation=%d format_version=%d",
            self.session_id,
            self.baseline_label,
            discovered_count,
            len(self.layer_stats),
            skipped_count,
            self.sample_rows_per_evaluation,
            CALIBRATION_VERSION,
        )

    def begin_evaluation(self, module: Any, args: object, kwargs: object) -> None:
        if not self.active:
            return
        if self._active_evaluation is not None:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' rejected reentrant "
                f"diffusion-model evaluation {self._active_evaluation.evaluation_index}."
            )
        try:
            time_parameter_name, time_value, time_value_truncated = _extract_evaluation_time_value(
                module,
                args,
                kwargs,
            )
        except Exception:
            time_parameter_name, time_value, time_value_truncated = None, None, False
        root_input_sum_x2, root_input_valid = _optional_root_input_energy(args)
        self._active_evaluation = EvaluationContext(
            evaluation_index=self._next_evaluation_index,
            time_parameter_name=time_parameter_name,
            time_value=time_value,
            time_value_truncated=time_value_truncated,
            root_input_sum_x2=root_input_sum_x2,
            root_input_valid=root_input_valid,
        )

    def _current_accumulator(self, tensor_name: str) -> LayerEvaluationAccumulator:
        context = self._active_evaluation
        if context is None:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' observed layer "
                f"'{tensor_name}' outside an active diffusion-model evaluation."
            )
        accumulator = context.layer_accumulators.get(tensor_name)
        if accumulator is None:
            stats = self.layer_stats[tensor_name]
            accumulator = LayerEvaluationAccumulator(
                tensor_name,
                stats.input_features,
                stats.output_features,
                self.sample_rows_per_evaluation,
            )
            context.layer_accumulators[tensor_name] = accumulator
        return accumulator

    def record_activation(
        self,
        tensor_name: str,
        input_features: int,
        weight_shape: tuple[Any, ...],
        activation: Any,
        module: Any,
    ) -> None:
        if not self.active:
            return
        context = self._active_evaluation
        if context is None:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' observed layer "
                f"'{tensor_name}' outside an active diffusion-model evaluation."
            )
        torch = _torch()
        observed_shape = (
            tuple(activation.shape)
            if isinstance(activation, torch.Tensor)
            else type(activation).__name__
        )
        if (
            not isinstance(activation, torch.Tensor)
            or activation.ndim < 1
            or activation.shape[-1] != input_features
        ):
            error = ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' evaluation "
                f"{context.evaluation_index} layer '{tensor_name}' received module class "
                f"'{_qualified_class_name(module)}'; weight shape {weight_shape!r}; "
                f"expected input feature count {input_features}, observed input shape "
                f"{observed_shape}."
            )
            LOGGER.error("[PotatoForge] %s", error)
            raise error

        stats = self.layer_stats[tensor_name]
        try:
            accumulator = self._current_accumulator(tensor_name)
            sum_x, sum_x2, max_abs_x, sample_count = accumulator.add_input(activation)
            _validate_finite_statistic(
                sum_x,
                session_id=self.session_id,
                evaluation_index=context.evaluation_index,
                tensor_name=tensor_name,
                statistic_name="sum_x",
            )
            _validate_finite_statistic(
                sum_x2,
                session_id=self.session_id,
                evaluation_index=context.evaluation_index,
                tensor_name=tensor_name,
                statistic_name="sum_x2",
            )
            _validate_finite_statistic(
                max_abs_x,
                session_id=self.session_id,
                evaluation_index=context.evaluation_index,
                tensor_name=tensor_name,
                statistic_name="max_abs_x",
            )
            stats.add_contribution(sum_x2, sample_count)
        except ActivationCalibrationError:
            raise
        except Exception as error:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' evaluation "
                f"{context.evaluation_index} layer '{tensor_name}' input capture failed "
                f"for shape {observed_shape}: {error}"
            ) from error

    def record_output(
        self,
        tensor_name: str,
        output_features: int,
        weight_shape: tuple[Any, ...],
        output: Any,
        module: Any,
    ) -> None:
        if not self.active:
            return
        context = self._active_evaluation
        if context is None:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' observed output "
                f"for layer '{tensor_name}' outside an active diffusion-model evaluation."
            )
        torch = _torch()
        observed_shape = (
            tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
        )
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim < 1
            or output.shape[-1] != output_features
        ):
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' evaluation "
                f"{context.evaluation_index} layer '{tensor_name}' received module class "
                f"'{_qualified_class_name(module)}'; weight shape {weight_shape!r}; "
                f"expected output feature count {output_features}, observed output shape "
                f"{observed_shape}."
            )

        try:
            accumulator = self._current_accumulator(tensor_name)
            sum_y, sum_y2 = accumulator.add_output(output)
            _validate_finite_statistic(
                sum_y,
                session_id=self.session_id,
                evaluation_index=context.evaluation_index,
                tensor_name=tensor_name,
                statistic_name="sum_y",
            )
            _validate_finite_statistic(
                sum_y2,
                session_id=self.session_id,
                evaluation_index=context.evaluation_index,
                tensor_name=tensor_name,
                statistic_name="sum_y2",
            )
        except ActivationCalibrationError as error:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' evaluation "
                f"{context.evaluation_index} layer '{tensor_name}' output capture failed "
                f"for shape {observed_shape}: {error}"
            ) from error
        except Exception as error:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' evaluation "
                f"{context.evaluation_index} layer '{tensor_name}' output capture failed "
                f"for shape {observed_shape}: {error}"
            ) from error

    def finish_evaluation(
        self,
        module: Any,
        args: object,
        kwargs: object,
        output: Any,
    ) -> Any:
        if not self.active:
            return output
        context = self._active_evaluation
        if context is None:
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' received a root "
                "forward completion without an active evaluation."
            )
        if output is None:
            self.cleanup()
            return output

        root_output_sum_y2, root_output_valid = _optional_root_output_energy(output)
        for tensor_name, accumulator in sorted(context.layer_accumulators.items()):
            if accumulator.output_invocation_count != accumulator.invocation_count:
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' evaluation "
                    f"{context.evaluation_index} layer '{tensor_name}' recorded "
                    f"{accumulator.invocation_count} inputs but "
                    f"{accumulator.output_invocation_count} outputs."
                )
            completed = accumulator.finalize_to_cpu()
            for statistic_name in (
                "sum_x",
                "sum_x2",
                "max_abs_x",
                "sum_y",
                "sum_y2",
                "sample_x",
            ):
                statistic = getattr(completed, statistic_name)
                _validate_finite_statistic(
                    statistic,
                    session_id=self.session_id,
                    evaluation_index=context.evaluation_index,
                    tensor_name=tensor_name,
                    statistic_name=statistic_name,
                )
            stats = self.layer_stats[tensor_name]
            if context.evaluation_index in stats.completed_evaluations:
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' received a "
                    f"duplicate evaluation index {context.evaluation_index} for layer "
                    f"'{tensor_name}'."
                )
            stats.completed_evaluations[context.evaluation_index] = completed

        self._evaluation_records.append(
            EvaluationRecord(
                evaluation_index=context.evaluation_index,
                time_parameter_name=context.time_parameter_name,
                time_value=context.time_value,
                time_value_truncated=context.time_value_truncated,
                root_input_sum_x2=context.root_input_sum_x2,
                root_input_valid=context.root_input_valid,
                root_output_sum_y2=root_output_sum_y2,
                root_output_valid=root_output_valid,
            )
        )
        self._active_evaluation = None
        self._next_evaluation_index += 1
        return output

    def _detach_hooks(self) -> None:
        handles = self._hook_handles
        self._hook_handles = []
        for handle in reversed(handles):
            try:
                handle.remove()
            except Exception:
                LOGGER.exception(
                    "[PotatoForge] Failed to remove activation calibration hook for session=%s",
                    self.session_id,
                )

    def cleanup(self) -> None:
        self._detach_hooks()
        self._active_evaluation = None
        if self._state not in {"finalized", "aborted"}:
            self._state = "aborted"
        _ACTIVE_SESSIONS.discard(self)

    def _metadata(self) -> dict[str, Any]:
        layers: dict[str, dict[str, Any]] = {}
        for tensor_name, stats in sorted(self.layer_stats.items()):
            layers[tensor_name] = {
                "input_features": stats.input_features,
                "output_features": stats.output_features,
                "sample_count": stats.sample_count,
                "invocation_count": stats.invocation_count,
                "stats_key": f"{tensor_name}.sum_x2",
                "eval_sum_x_key": f"{tensor_name}.eval_sum_x",
                "eval_sum_x2_key": f"{tensor_name}.eval_sum_x2",
                "eval_max_abs_x_key": f"{tensor_name}.eval_max_abs_x",
                "eval_sum_y_key": f"{tensor_name}.eval_sum_y",
                "eval_sum_y2_key": f"{tensor_name}.eval_sum_y2",
                "eval_sample_count_key": f"{tensor_name}.eval_sample_count",
                "eval_invocation_count_key": f"{tensor_name}.eval_invocation_count",
                "sample_x_key": (
                    f"{tensor_name}.sample_x"
                    if self.sample_rows_per_evaluation > 0
                    else None
                ),
                "sample_x_valid_key": (
                    f"{tensor_name}.sample_x_valid"
                    if self.sample_rows_per_evaluation > 0
                    else None
                ),
            }
        evaluations = [
            {
                "evaluation_index": record.evaluation_index,
                "time_parameter_name": record.time_parameter_name,
                "time_value": record.time_value,
                "time_value_truncated": record.time_value_truncated,
            }
            for record in self._evaluation_records
        ]
        return {
            "format": CALIBRATION_FORMAT,
            "version": CALIBRATION_VERSION,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "baseline_label": self.baseline_label,
            "activation_basis": "logical_linear_input",
            "activation_axis": "last_dimension",
            "include_regex": self.include_regex,
            "exclude_regex": self.exclude_regex,
            "diffusion_model_class": self.diffusion_model_class,
            "started_at": self.started_at,
            "layer_count": len(layers),
            "evaluation_count": len(evaluations),
            "sample_rows_per_evaluation": self.sample_rows_per_evaluation,
            "evaluation_basis": "diffusion_model_forward",
            "evaluations": evaluations,
            "layers": layers,
        }

    def _materialize_tensors(self) -> dict[str, Any]:
        torch = _torch()
        evaluation_count = len(self._evaluation_records)
        for expected_index, record in enumerate(self._evaluation_records):
            if record.evaluation_index != expected_index:
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' has non-contiguous "
                    f"evaluation index {record.evaluation_index}; expected {expected_index}."
                )
        tensors: dict[str, Any] = {
            _ROOT_INPUT_SUM_X2_KEY: torch.tensor(
                [
                    record.root_input_sum_x2 if record.root_input_valid else 0.0
                    for record in self._evaluation_records
                ],
                dtype=torch.float32,
            ).contiguous(),
            _ROOT_INPUT_VALID_KEY: torch.tensor(
                [record.root_input_valid for record in self._evaluation_records],
                dtype=torch.bool,
            ).contiguous(),
            _ROOT_OUTPUT_SUM_Y2_KEY: torch.tensor(
                [
                    record.root_output_sum_y2 if record.root_output_valid else 0.0
                    for record in self._evaluation_records
                ],
                dtype=torch.float32,
            ).contiguous(),
            _ROOT_OUTPUT_VALID_KEY: torch.tensor(
                [record.root_output_valid for record in self._evaluation_records],
                dtype=torch.bool,
            ).contiguous(),
        }

        for tensor_name, stats in sorted(self.layer_stats.items()):
            eval_sum_x = torch.zeros(
                (evaluation_count, stats.input_features),
                dtype=torch.float32,
            )
            eval_sum_x2 = torch.zeros_like(eval_sum_x)
            eval_max_abs_x = torch.zeros_like(eval_sum_x)
            eval_sum_y = torch.zeros(
                (evaluation_count, stats.output_features),
                dtype=torch.float32,
            )
            eval_sum_y2 = torch.zeros_like(eval_sum_y)
            eval_sample_count = torch.zeros(evaluation_count, dtype=torch.int64)
            eval_invocation_count = torch.zeros(evaluation_count, dtype=torch.int64)
            sample_x = (
                torch.zeros(
                    (evaluation_count, self.sample_rows_per_evaluation, stats.input_features),
                    dtype=torch.float32,
                )
                if self.sample_rows_per_evaluation > 0
                else None
            )
            sample_x_valid = (
                torch.zeros(evaluation_count, dtype=torch.int64)
                if self.sample_rows_per_evaluation > 0
                else None
            )
            target_tensors = {
                "sum_x": eval_sum_x,
                "sum_x2": eval_sum_x2,
                "max_abs_x": eval_max_abs_x,
                "sum_y": eval_sum_y,
                "sum_y2": eval_sum_y2,
            }
            expected_shapes = {
                "sum_x": (stats.input_features,),
                "sum_x2": (stats.input_features,),
                "max_abs_x": (stats.input_features,),
                "sum_y": (stats.output_features,),
                "sum_y2": (stats.output_features,),
            }

            for evaluation_index, completed in stats.completed_evaluations.items():
                if not 0 <= evaluation_index < evaluation_count:
                    raise ActivationCalibrationError(
                        f"Activation calibration session '{self.session_id}' layer "
                        f"'{tensor_name}' contains invalid evaluation index {evaluation_index}."
                    )
                for statistic_name, expected_shape in expected_shapes.items():
                    statistic = getattr(completed, statistic_name)
                    if statistic.dtype != torch.float32 or tuple(statistic.shape) != expected_shape:
                        raise ActivationCalibrationError(
                            f"Activation calibration session '{self.session_id}' evaluation "
                            f"{evaluation_index} layer '{tensor_name}' produced invalid "
                            f"{statistic_name} dtype={statistic.dtype} shape={tuple(statistic.shape)}; "
                            f"expected dtype=torch.float32 shape={expected_shape}."
                        )
                    _validate_finite_statistic(
                        statistic,
                        session_id=self.session_id,
                        evaluation_index=evaluation_index,
                        tensor_name=tensor_name,
                        statistic_name=statistic_name,
                    )
                    target_tensors[statistic_name][evaluation_index].copy_(statistic)

                if completed.sample_count < 0 or completed.invocation_count < 0:
                    raise ActivationCalibrationError(
                        f"Activation calibration session '{self.session_id}' evaluation "
                        f"{evaluation_index} layer '{tensor_name}' contains negative counts."
                    )
                eval_sample_count[evaluation_index] = completed.sample_count
                eval_invocation_count[evaluation_index] = completed.invocation_count
                if sample_x is not None and sample_x_valid is not None:
                    valid_count = int(completed.sample_x.shape[0])
                    if (
                        completed.sample_x.dtype != torch.float32
                        or completed.sample_x.ndim != 2
                        or tuple(completed.sample_x.shape[1:]) != (stats.input_features,)
                        or valid_count > self.sample_rows_per_evaluation
                    ):
                        raise ActivationCalibrationError(
                            f"Activation calibration session '{self.session_id}' evaluation "
                            f"{evaluation_index} layer '{tensor_name}' produced invalid sample_x "
                            f"dtype={completed.sample_x.dtype} shape={tuple(completed.sample_x.shape)}."
                        )
                    _validate_finite_statistic(
                        completed.sample_x,
                        session_id=self.session_id,
                        evaluation_index=evaluation_index,
                        tensor_name=tensor_name,
                        statistic_name="sample_x",
                    )
                    if valid_count:
                        sample_x[evaluation_index, :valid_count].copy_(completed.sample_x)
                    sample_x_valid[evaluation_index] = valid_count

            aggregate_sum_x2 = stats.finalize_to_cpu()
            if aggregate_sum_x2.dtype != torch.float32 or tuple(aggregate_sum_x2.shape) != (
                stats.input_features,
            ):
                raise ActivationCalibrationError(
                    f"Activation calibration layer '{tensor_name}' produced invalid "
                    f"CPU statistics dtype={aggregate_sum_x2.dtype} shape={tuple(aggregate_sum_x2.shape)}."
                )
            _validate_finite_statistic(
                aggregate_sum_x2,
                session_id=self.session_id,
                evaluation_index=-1,
                tensor_name=tensor_name,
                statistic_name="sum_x2",
            )
            if not torch.allclose(
                eval_sum_x2.sum(dim=0),
                aggregate_sum_x2,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' layer '{tensor_name}' "
                    "aggregate sum_x2 did not match per-evaluation sum_x2."
                )
            if int(eval_sample_count.sum().item()) != stats.sample_count:
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' layer '{tensor_name}' "
                    "aggregate sample_count did not match per-evaluation counts."
                )
            if int(eval_invocation_count.sum().item()) != stats.invocation_count:
                raise ActivationCalibrationError(
                    f"Activation calibration session '{self.session_id}' layer '{tensor_name}' "
                    "aggregate invocation_count did not match per-evaluation counts."
                )

            tensors[f"{tensor_name}.sum_x2"] = aggregate_sum_x2.contiguous()
            tensors[f"{tensor_name}.eval_sum_x"] = eval_sum_x.contiguous()
            tensors[f"{tensor_name}.eval_sum_x2"] = eval_sum_x2.contiguous()
            tensors[f"{tensor_name}.eval_max_abs_x"] = eval_max_abs_x.contiguous()
            tensors[f"{tensor_name}.eval_sum_y"] = eval_sum_y.contiguous()
            tensors[f"{tensor_name}.eval_sum_y2"] = eval_sum_y2.contiguous()
            tensors[f"{tensor_name}.eval_sample_count"] = eval_sample_count.contiguous()
            tensors[f"{tensor_name}.eval_invocation_count"] = eval_invocation_count.contiguous()
            if sample_x is not None and sample_x_valid is not None:
                tensors[f"{tensor_name}.sample_x"] = sample_x.contiguous()
                tensors[f"{tensor_name}.sample_x_valid"] = sample_x_valid.contiguous()

        return tensors

    def finalize(self) -> tuple[Path, Path]:
        if self._saved_paths is not None:
            return self._saved_paths
        if not self.active:
            raise RuntimeError(
                f"Activation calibration session '{self.session_id}' is already {self._state}."
            )
        if self._active_evaluation is not None:
            evaluation_index = self._active_evaluation.evaluation_index
            self.cleanup()
            raise ActivationCalibrationError(
                f"Activation calibration session '{self.session_id}' cannot finalize "
                f"while evaluation {evaluation_index} is active."
            )

        self._state = "finalizing"
        try:
            tensors = self._materialize_tensors()
            for tensor_name, stats in sorted(self.layer_stats.items()):
                if stats.invocation_count == 0:
                    LOGGER.warning(
                        "[PotatoForge] Activation calibration layer had zero invocations "
                        "session=%s layer=%s",
                        self.session_id,
                        tensor_name,
                    )
            if self._evaluation_records and not any(
                record.time_parameter_name for record in self._evaluation_records
            ):
                LOGGER.warning(
                    "[PotatoForge] Activation calibration could not identify a timestep/sigma "
                    "parameter for any evaluation session=%s",
                    self.session_id,
                )
            if self._evaluation_records and not all(
                record.root_input_valid for record in self._evaluation_records
            ):
                LOGGER.warning(
                    "[PotatoForge] Activation calibration root input energy was unavailable "
                    "for one or more evaluations session=%s",
                    self.session_id,
                )
            if self._evaluation_records and not all(
                record.root_output_valid for record in self._evaluation_records
            ):
                LOGGER.warning(
                    "[PotatoForge] Activation calibration root output energy was unavailable "
                    "for one or more evaluations session=%s",
                    self.session_id,
                )
            stats_path, metadata_path = _save_calibration(
                self.output_directory,
                self.session_name,
                self.session_id,
                tensors,
                self._metadata(),
            )
            self._saved_paths = (stats_path, metadata_path)
            self._state = "finalized"
            aggregate_samples = sum(stats.sample_count for stats in self.layer_stats.values())
            LOGGER.info(
                "[PotatoForge] Activation calibration finalized layers=%d evaluations=%d "
                "samples=%d stats=%s metadata=%s",
                len(self.layer_stats),
                len(self._evaluation_records),
                aggregate_samples,
                stats_path,
                metadata_path,
            )
            return self._saved_paths
        finally:
            self._detach_hooks()
            _ACTIVE_SESSIONS.discard(self)
            if self._state == "finalizing":
                self._state = "aborted"


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        LOGGER.warning("[PotatoForge] Could not remove temporary calibration file '%s'", path)


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _save_calibration(
    output_directory: Path,
    session_name: str,
    session_id: str,
    tensors: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    safe_name = sanitize_session_name(session_name)
    basename = f"{safe_name}_{session_id[:8]}"
    stats_path = output_directory / f"{basename}.safetensors"
    metadata_path = output_directory / f"{basename}.json"
    if stats_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"PotatoForge activation calibration output already exists for session '{session_id}'."
        )

    temporary_stats_path = _temporary_path(stats_path)
    temporary_metadata_path = _temporary_path(metadata_path)
    moved_paths: list[Path] = []
    try:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError(
                "PotatoForge activation calibration requires ComfyUI's safetensors support."
            ) from error

        save_file(tensors, str(temporary_stats_path))
        _fsync_file(temporary_stats_path)
        with temporary_metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if stats_path.exists() or metadata_path.exists():
            raise FileExistsError(
                f"PotatoForge activation calibration output already exists for session '{session_id}'."
            )
        temporary_stats_path.rename(stats_path)
        moved_paths.append(stats_path)
        temporary_metadata_path.rename(metadata_path)
        moved_paths.append(metadata_path)
    except BaseException:
        LOGGER.exception(
            "[PotatoForge] Activation calibration save failed session=%s",
            session_id,
        )
        for path in moved_paths:
            _remove_if_present(path)
        _remove_if_present(temporary_stats_path)
        _remove_if_present(temporary_metadata_path)
        raise

    return stats_path, metadata_path


def resolve_diffusion_model(model: Any) -> Any:
    torch = _torch()
    candidates = (
        getattr(getattr(model, "model", None), "diffusion_model", None),
        getattr(model, "diffusion_model", None),
    )
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            return candidate
    raise RuntimeError(
        "PotatoForge activation calibration could not locate a diffusion model; "
        "expected a ComfyUI MODEL with model.model.diffusion_model."
    )


def start_activation_calibration(
    model: Any,
    session_name: str = "calibration",
    baseline_label: str = "unknown",
    include_regex: str = ".*",
    exclude_regex: str = "",
    enabled: bool = True,
    output_directory: str | Path | None = None,
    sample_rows_per_evaluation: int = 2,
) -> ActivationCalibrationSession | None:
    if not enabled:
        return None

    include_source, exclude_source, _include_pattern, _exclude_pattern = _compile_filters(
        include_regex,
        exclude_regex,
    )
    diffusion_model = resolve_diffusion_model(model)
    output_path = (
        calibration_output_directory()
        if output_directory is None
        else _prepare_output_directory(output_directory)
    )
    _cleanup_stale_sessions()
    session = ActivationCalibrationSession(
        session_name=session_name,
        baseline_label=baseline_label,
        include_regex=include_source,
        exclude_regex=exclude_source,
        output_directory=output_path,
        diffusion_model_class=_qualified_class_name(diffusion_model),
        sample_rows_per_evaluation=sample_rows_per_evaluation,
    )
    try:
        session.attach(diffusion_model)
    except BaseException:
        session.cleanup()
        raise
    return session


class PotatoForgeActivationCalibration:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "model": ("MODEL",),
                "session_name": ("STRING", {"default": "calibration"}),
                "baseline_label": ("STRING", {"default": "unknown"}),
                "include_regex": ("STRING", {"default": ".*"}),
                "exclude_regex": ("STRING", {"default": ""}),
                "enabled": ("BOOLEAN", {"default": True}),
                "sample_rows_per_evaluation": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", CALIBRATION_SESSION_TYPE)
    RETURN_NAMES = ("model", "session")
    FUNCTION = "calibrate"
    CATEGORY = "PotatoForge/Calibration"

    @classmethod
    def IS_CHANGED(cls, *args: Any, **kwargs: Any) -> float:
        return float("nan")

    def calibrate(
        self,
        model: Any,
        session_name: str,
        baseline_label: str,
        include_regex: str,
        exclude_regex: str,
        enabled: bool,
        sample_rows_per_evaluation: int = 2,
    ) -> tuple[Any, ActivationCalibrationSession | None]:
        return (
            model,
            start_activation_calibration(
                model,
                session_name=session_name,
                baseline_label=baseline_label,
                include_regex=include_regex,
                exclude_regex=exclude_regex,
                enabled=enabled,
                sample_rows_per_evaluation=sample_rows_per_evaluation,
            ),
        )


class PotatoForgeFinalizeActivationCalibration:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "latent": ("LATENT",),
                "session": (CALIBRATION_SESSION_TYPE,),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "stats_path", "metadata_path")
    FUNCTION = "finalize"
    CATEGORY = "PotatoForge/Calibration"

    @classmethod
    def IS_CHANGED(cls, *args: Any, **kwargs: Any) -> float:
        return float("nan")

    def finalize(
        self,
        latent: Any,
        session: ActivationCalibrationSession | None,
    ) -> tuple[Any, str, str]:
        if session is None:
            return latent, "", ""
        if not isinstance(session, ActivationCalibrationSession):
            raise TypeError(
                "PotatoForge finalization expected a POTATOFORGE_CALIBRATION_SESSION value."
            )
        stats_path, metadata_path = session.finalize()
        return latent, str(stats_path), str(metadata_path)


__all__ = [
    "ActivationCalibrationError",
    "ActivationCalibrationSession",
    "CALIBRATION_SESSION_TYPE",
    "LayerActivationStats",
    "PotatoForgeActivationCalibration",
    "PotatoForgeFinalizeActivationCalibration",
    "canonical_weight_name",
    "calibration_output_directory",
    "per_feature_sum_of_squares",
    "resolve_diffusion_model",
    "sanitize_session_name",
    "start_activation_calibration",
]
