from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from .manifest import QUANT_FAMILY_SUFFIXES, validate_patch_tensor_keys
from .stack import QuantPatchRef


@dataclass(frozen=True)
class PatchConflict:
    family: str
    previous_patch_id: str
    replacement_patch_id: str


@dataclass(frozen=True)
class OverlayReport:
    replacement_count: int
    conflicts: tuple[PatchConflict, ...]


def overlay_quant_patches(
    state_dict: MutableMapping[str, Any],
    patches: Iterable[tuple[QuantPatchRef, Mapping[str, Any]]],
) -> OverlayReport:
    """Overlay validated complete quantized families onto one baseline state dict."""
    owners: dict[str, str] = {}
    conflicts: list[PatchConflict] = []
    replacement_count = 0

    for patch_ref, patch_state_dict in patches:
        validate_patch_tensor_keys(patch_ref.manifest, patch_state_dict.keys(), patch_ref.name)
        missing_baseline_families = [
            family
            for family in patch_ref.manifest.replaces
            if family not in owners and f"{family}.weight" not in state_dict
        ]
        if missing_baseline_families:
            raise ValueError(
                f"PotatoForge patch '{patch_ref.name}' cannot replace "
                f"{', '.join(missing_baseline_families)}: baseline checkpoint does not contain "
                f"{missing_baseline_families[0]}.weight."
            )

        for family in patch_ref.manifest.replaces:
            previous_owner = owners.get(family)
            if previous_owner is not None:
                conflicts.append(
                    PatchConflict(family, previous_owner, patch_ref.manifest.patch_id)
                )
            for suffix in QUANT_FAMILY_SUFFIXES:
                state_dict.pop(f"{family}{suffix}", None)
            owners[family] = patch_ref.manifest.patch_id

        state_dict.update(patch_state_dict)
        replacement_count += len(patch_ref.manifest.replaces)

    return OverlayReport(replacement_count, tuple(conflicts))
