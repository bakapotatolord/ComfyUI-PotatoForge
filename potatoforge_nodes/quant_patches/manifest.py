from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Mapping


QUANT_FAMILY_SUFFIXES = (".weight", ".weight_scale", ".comfy_quant")


class QuantPatchValidationError(ValueError):
    """Raised when a PotatoForge quant patch violates its V1 contract."""


@dataclass(frozen=True)
class QuantPatchManifest:
    patch_id: str
    replaces: tuple[str, ...]


def parse_quant_patch_metadata(
    metadata: Mapping[str, object] | None,
    patch_label: str,
) -> QuantPatchManifest:
    if not isinstance(metadata, Mapping):
        raise QuantPatchValidationError(f"PotatoForge patch '{patch_label}' has no metadata.")

    required = (
        "potatoforge_file_type",
        "potatoforge_patch_format",
        "potatoforge_patch_id",
        "potatoforge_patch_replaces",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' is missing metadata: {', '.join(missing)}."
        )

    if metadata["potatoforge_file_type"] != "quant_patch":
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' has unsupported file type "
            f"{metadata['potatoforge_file_type']!r}; expected 'quant_patch'."
        )
    if metadata["potatoforge_patch_format"] != "1":
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' has unsupported patch format "
            f"{metadata['potatoforge_patch_format']!r}; expected '1'."
        )

    patch_id = metadata["potatoforge_patch_id"]
    if not isinstance(patch_id, str) or not patch_id.strip():
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' has an empty or invalid potatoforge_patch_id."
        )

    replaces_json = metadata["potatoforge_patch_replaces"]
    if not isinstance(replaces_json, str):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' has a non-string potatoforge_patch_replaces value."
        )
    try:
        replaces = json.loads(replaces_json)
    except json.JSONDecodeError as error:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' has invalid JSON in potatoforge_patch_replaces: {error.msg}."
        ) from error
    if not isinstance(replaces, list) or not replaces:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' must declare a non-empty list of replaced families."
        )
    if any(not isinstance(family, str) or not family.strip() for family in replaces):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' declares an empty or non-string replacement family."
        )
    if len(set(replaces)) != len(replaces):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{patch_label}' declares duplicate replacement families."
        )

    return QuantPatchManifest(
        patch_id=patch_id,
        replaces=tuple(replaces),
    )


def validate_patch_tensor_keys(
    manifest: QuantPatchManifest,
    tensor_keys: object,
    patch_label: str,
) -> None:
    actual = set(tensor_keys)
    expected = {f"{family}{suffix}" for family in manifest.replaces for suffix in QUANT_FAMILY_SUFFIXES}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return

    details: list[str] = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected: {', '.join(unexpected)}")
    raise QuantPatchValidationError(
        f"PotatoForge patch '{patch_label}' tensor family validation failed ({'; '.join(details)})."
    )


def read_safetensors_header(path: Path) -> tuple[Mapping[str, object], tuple[str, ...]]:
    """Read only Safetensors metadata and names, never the tensor payloads."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as patch_file:
            length_bytes = patch_file.read(8)
            if len(length_bytes) != 8:
                raise QuantPatchValidationError(
                    f"PotatoForge patch '{path.name}' is not a complete Safetensors file."
                )
            header_size = struct.unpack("<Q", length_bytes)[0]
            if header_size > file_size - 8:
                raise QuantPatchValidationError(
                    f"PotatoForge patch '{path.name}' declares an invalid Safetensors header length."
                )
            header_bytes = patch_file.read(header_size)
    except OSError as error:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{path}' could not be read: {error}."
        ) from error

    if len(header_bytes) != header_size:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{path.name}' has a truncated Safetensors header."
        )
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuantPatchValidationError(
            f"PotatoForge patch '{path.name}' has invalid Safetensors header JSON."
        ) from error
    if not isinstance(header, dict):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{path.name}' has a non-object Safetensors header."
        )

    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise QuantPatchValidationError(
            f"PotatoForge patch '{path.name}' has no Safetensors __metadata__ object."
        )
    tensor_keys = tuple(key for key in header if key != "__metadata__")
    return metadata, tensor_keys
