from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import load_patched_diffusion_model
from .stack import QuantPatchStack, as_patch_stack, inspect_quant_patch


PATCH_FOLDER_NAME = "potatoforge_patches"
PATCH_STACK_TYPE = "POTATOFORGE_QUANT_PATCH_STACK"
PATCH_EXTENSIONS = {".safetensors"}


def _folder_paths() -> Any:
    import folder_paths

    return folder_paths


def _torch() -> Any:
    import torch

    return torch


def register_patch_folder() -> Path:
    folder_paths = _folder_paths()
    patch_directory = Path(folder_paths.models_dir) / PATCH_FOLDER_NAME
    patch_directory.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(PATCH_FOLDER_NAME, str(patch_directory))
    return patch_directory


def available_patch_names() -> list[str]:
    folder_paths = _folder_paths()
    return [
        name
        for name in folder_paths.get_filename_list(PATCH_FOLDER_NAME)
        if Path(name).suffix.lower() in PATCH_EXTENSIONS
    ]


def build_model_options(weight_dtype: str) -> dict[str, Any]:
    if weight_dtype == "default":
        return {}

    torch = _torch()
    if weight_dtype == "fp8_e4m3fn":
        return {"dtype": torch.float8_e4m3fn}
    if weight_dtype == "fp8_e4m3fn_fast":
        return {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}
    if weight_dtype == "fp8_e5m2":
        return {"dtype": torch.float8_e5m2}
    raise ValueError(f"Unsupported diffusion-model weight dtype: {weight_dtype!r}.")


class PotatoForgeAddQuantPatch:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "patch_name": (available_patch_names(),),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {"patch_stack": (PATCH_STACK_TYPE,)},
        }

    RETURN_TYPES = (PATCH_STACK_TYPE,)
    FUNCTION = "add_patch"
    CATEGORY = "PotatoForge/Quant Patches"

    @classmethod
    def IS_CHANGED(
        cls,
        patch_name: str,
        enabled: bool = True,
        patch_stack: QuantPatchStack | None = None,
    ) -> str:
        if not enabled:
            return ""
        path = Path(_folder_paths().get_full_path_or_raise(PATCH_FOLDER_NAME, patch_name))
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"

    def add_patch(
        self,
        patch_name: str,
        enabled: bool,
        patch_stack: QuantPatchStack | None = None,
    ) -> tuple[QuantPatchStack | None]:
        if not enabled:
            return (patch_stack,)
        path = _folder_paths().get_full_path_or_raise(PATCH_FOLDER_NAME, patch_name)
        return (as_patch_stack(patch_stack).append(inspect_quant_patch(patch_name, path)),)


class PotatoForgePatchedDiffusionModelLoader:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        folder_paths = _folder_paths()
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                    {"advanced": True},
                ),
            },
            "optional": {"patch_stack": (PATCH_STACK_TYPE,)},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "PotatoForge/Quant Patches"

    def load_model(
        self,
        model_name: str,
        weight_dtype: str,
        patch_stack: QuantPatchStack | None = None,
    ) -> tuple[Any]:
        model_path = _folder_paths().get_full_path_or_raise("diffusion_models", model_name)
        return (
            load_patched_diffusion_model(
                model_path,
                patch_stack,
                build_model_options(weight_dtype),
            ),
        )
