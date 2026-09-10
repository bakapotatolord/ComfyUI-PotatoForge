from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
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


def per_feature_sum_of_squares(activation: Any) -> Any:
    """Return FP32 per-last-dimension sums without retaining the activation."""
    torch = _torch()
    if not isinstance(activation, torch.Tensor):
        raise TypeError("Activation calibration expected a torch.Tensor input.")
    if activation.ndim < 1:
        raise ActivationCalibrationError(
            f"Activation calibration expected at least one dimension, got shape {tuple(activation.shape)}."
        )

    detached = activation.detach()
    squared = detached.square()
    reduction_dims = tuple(range(detached.ndim - 1))
    if not reduction_dims:
        return squared.to(dtype=torch.float32)
    return squared.sum(dim=reduction_dims, dtype=torch.float32)


class LayerActivationStats:
    """Raw per-input-feature activation energy for one logical weight."""

    def __init__(self, tensor_name: str, input_features: int) -> None:
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("Layer activation statistics require a tensor name.")
        if type(input_features) is not int or input_features < 1:
            raise ValueError("Layer activation statistics require positive input features.")

        self.tensor_name = tensor_name
        self.input_features = input_features
        self.sample_count = 0
        self.invocation_count = 0
        self.sum_x2_by_device: dict[str, Any] = {}

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
        sample_count = math.prod(int(dimension) for dimension in activation.shape[:-1])
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

        device_key = str(contribution.device)
        existing = self.sum_x2_by_device.get(device_key)
        if existing is None:
            self.sum_x2_by_device[device_key] = contribution
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
        sanitize_session_name(self.session_name)
        self.include_regex = include_source
        self.exclude_regex = exclude_source
        self._include_pattern = include_pattern
        self._exclude_pattern = exclude_pattern
        self.output_directory = _prepare_output_directory(output_directory)
        self.diffusion_model_class = diffusion_model_class
        self.session_id = session_id or uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.layer_stats: dict[str, LayerActivationStats] = {}
        self.stats = self.layer_stats
        self._hook_handles: list[Any] = []
        self._state = "active"
        self._saved_paths: tuple[Path, Path] | None = None
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
        except BaseException:
            self.cleanup()
            raise

        LOGGER.info(
            "[PotatoForge] Activation calibration started session=%s baseline=%s "
            "discovered linear layers=%d hooked layers=%d skipped layers=%d",
            self.session_id,
            self.baseline_label,
            discovered_count,
            len(self.layer_stats),
            skipped_count,
        )

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
        torch = _torch()
        observed_shape = getattr(activation, "shape", None)
        if isinstance(activation, torch.Tensor):
            observed_shape = tuple(activation.shape)
        else:
            observed_shape = type(activation).__name__
        if (
            not isinstance(activation, torch.Tensor)
            or activation.ndim < 1
            or activation.shape[-1] != input_features
        ):
            error = ActivationCalibrationError(
                f"Activation calibration layer '{tensor_name}' received module class "
                f"'{_qualified_class_name(module)}'; weight shape {weight_shape!r}; "
                f"expected input feature count {input_features}, observed input shape "
                f"{observed_shape}."
            )
            LOGGER.error("[PotatoForge] %s", error)
            raise error

        stats = self.layer_stats[tensor_name]
        try:
            stats.add(activation)
        except Exception as error:
            raise ActivationCalibrationError(
                f"Activation calibration layer '{tensor_name}' received module class "
                f"'{_qualified_class_name(module)}'; weight shape {weight_shape!r}; "
                f"expected input feature count {input_features}, observed input shape "
                f"{observed_shape}: {error}"
            ) from error

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
        if self._state not in {"finalized", "aborted"}:
            self._state = "aborted"
        _ACTIVE_SESSIONS.discard(self)

    def _metadata(self) -> dict[str, Any]:
        layers: dict[str, dict[str, Any]] = {}
        for tensor_name, stats in sorted(self.layer_stats.items()):
            layers[tensor_name] = {
                "input_features": stats.input_features,
                "sample_count": stats.sample_count,
                "invocation_count": stats.invocation_count,
                "stats_key": f"{tensor_name}.sum_x2",
            }
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
            "layers": layers,
        }

    def finalize(self) -> tuple[Path, Path]:
        if self._saved_paths is not None:
            return self._saved_paths
        if not self.active:
            raise RuntimeError(
                f"Activation calibration session '{self.session_id}' is already {self._state}."
            )

        self._state = "finalizing"
        try:
            tensors: dict[str, Any] = {}
            for tensor_name, stats in sorted(self.layer_stats.items()):
                if stats.invocation_count == 0:
                    LOGGER.warning(
                        "[PotatoForge] Activation calibration layer had zero invocations "
                        "session=%s layer=%s",
                        self.session_id,
                        tensor_name,
                    )
                sum_x2 = stats.finalize_to_cpu()
                torch = _torch()
                if sum_x2.dtype != torch.float32 or tuple(sum_x2.shape) != (
                    stats.input_features,
                ):
                    raise ActivationCalibrationError(
                        f"Activation calibration layer '{tensor_name}' produced invalid "
                        f"CPU statistics dtype={sum_x2.dtype} shape={tuple(sum_x2.shape)}."
                    )
                tensors[f"{tensor_name}.sum_x2"] = sum_x2

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
                "[PotatoForge] Activation calibration finalized layers=%d samples=%d stats=%s metadata=%s",
                len(self.layer_stats),
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
