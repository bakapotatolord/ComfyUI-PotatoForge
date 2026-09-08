from __future__ import annotations

from dataclasses import replace
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import QuantPatchValidationError, parse_quant_patch_metadata, validate_patch_tensor_keys
from .merge import overlay_quant_patches
from .stack import QuantPatchRef, QuantPatchStack, as_patch_stack, inspect_quant_patch


LOGGER = logging.getLogger("potatoforge")


def _comfy_modules() -> tuple[Any, Any]:
    import comfy.sd
    import comfy.utils

    return comfy.sd, comfy.utils


def _load_patch_payload(
    patch: QuantPatchRef,
    comfy_utils: Any,
) -> tuple[QuantPatchRef, Mapping[str, Any]]:
    state_dict, metadata = comfy_utils.load_torch_file(
        str(patch.path), safe_load=True, return_metadata=True
    )
    if not isinstance(state_dict, Mapping):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch.name}' did not load as a tensor state dictionary."
        )
    manifest = parse_quant_patch_metadata(metadata, patch.name)
    validate_patch_tensor_keys(manifest, state_dict.keys(), patch.name)
    return replace(patch, manifest=manifest), state_dict


def load_patched_diffusion_model(
    model_path: str,
    patch_stack: QuantPatchStack | None = None,
    model_options: Mapping[str, Any] | None = None,
    disable_dynamic: bool = False,
) -> Any:
    """Load a baseline through ComfyUI after applying complete quant-patch families."""
    comfy_sd, comfy_utils = _comfy_modules()
    options = dict(model_options or {})
    stack = as_patch_stack(patch_stack)
    if not stack.patches:
        return comfy_sd.load_diffusion_model(
            model_path, model_options=options, disable_dynamic=disable_dynamic
        )

    patches = tuple(inspect_quant_patch(patch.name, patch.path) for patch in stack.patches)
    seen_paths: set[Path] = set()
    for patch in patches:
        if patch.path in seen_paths:
            LOGGER.warning("[PotatoForge] Reapplying duplicate quant patch %s", patch.name)
        seen_paths.add(patch.path)

    LOGGER.info(
        "[PotatoForge] Loading %s baseline with %d quant patches",
        Path(model_path).name,
        len(patches),
    )
    baseline_state_dict, baseline_metadata = comfy_utils.load_torch_file(
        model_path, return_metadata=True
    )
    loaded_patches = tuple(_load_patch_payload(patch, comfy_utils) for patch in patches)
    report = overlay_quant_patches(baseline_state_dict, loaded_patches)

    for conflict in report.conflicts:
        LOGGER.warning(
            "[PotatoForge] Quant patch conflict: %s; previous: %s; replacement: %s; "
            "using later patch",
            conflict.family,
            conflict.previous_patch_id,
            conflict.replacement_patch_id,
        )
    LOGGER.info("[PotatoForge] Applied %d logical layer replacements", report.replacement_count)

    model = comfy_sd.load_diffusion_model_state_dict(
        baseline_state_dict,
        model_options=options,
        metadata=baseline_metadata,
        disable_dynamic=disable_dynamic,
    )
    if model is None:
        raise RuntimeError(
            f"PotatoForge could not detect the diffusion model after applying "
            f"{len(patches)} quant patches to '{Path(model_path).name}'."
        )

    model.cached_patcher_init = (
        load_patched_diffusion_model,
        (model_path, QuantPatchStack(tuple(patch for patch, _ in loaded_patches)), options),
    )
    return model
