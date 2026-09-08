from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .manifest import (
    QuantPatchManifest,
    read_safetensors_header,
    parse_quant_patch_metadata,
    validate_patch_tensor_keys,
)


@dataclass(frozen=True)
class QuantPatchRef:
    name: str
    path: Path
    manifest: QuantPatchManifest


@dataclass(frozen=True)
class QuantPatchStack:
    patches: tuple[QuantPatchRef, ...] = ()

    def append(self, patch: QuantPatchRef) -> QuantPatchStack:
        return QuantPatchStack(self.patches + (patch,))


def inspect_quant_patch(name: str, path: str | Path) -> QuantPatchRef:
    resolved_path = Path(path).resolve()
    metadata, tensor_keys = read_safetensors_header(resolved_path)
    manifest = parse_quant_patch_metadata(metadata, name)
    validate_patch_tensor_keys(manifest, tensor_keys, name)
    return QuantPatchRef(
        name=name,
        path=resolved_path,
        manifest=manifest,
    )


def as_patch_stack(value: QuantPatchStack | None) -> QuantPatchStack:
    if value is None:
        return QuantPatchStack()
    if not isinstance(value, QuantPatchStack):
        raise TypeError("Expected a POTATOFORGE_QUANT_PATCH_STACK value.")
    return value
