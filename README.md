# ComfyUI-PotatoForge

ComfyUI runtime nodes for PotatoForge quantization patches. A patch is a
Safetensors artifact that replaces complete serialized quantized layer families
while ComfyUI loads a diffusion model; it is not a LoRA or a model patch.

## Install

Place this repository in `ComfyUI/custom_nodes/ComfyUI-PotatoForge` and restart
ComfyUI. No additional Python package is required beyond ComfyUI's existing
Torch and Safetensors support.

The extension creates this folder on startup:

```text
ComfyUI/models/potatoforge_patches/
```

Subfolders are supported. Only `.safetensors` files appear in the patch
dropdown.

## Workflow

```text
PotatoForge Add Quant Patch
          ↓
PotatoForge Add Quant Patch
          ↓
PotatoForge Load Diffusion Model + Patches
          ↓
        MODEL
```

`PotatoForge Add Quant Patch` outputs `POTATOFORGE_QUANT_PATCH_STACK`; chain as
many nodes as needed. Disabled nodes pass their incoming stack through unchanged.

Patches apply in stack order. If two patches replace the same logical layer, the
later patch replaces the earlier patch's whole `.weight`, `.weight_scale`, and
`.comfy_quant` family. Unrelated tensors, including a layer's `.bias`, are left
unchanged. Conflicts are logged.

The loader validates metadata and complete tensor families before it loads the
baseline checkpoint. It overlays tensors in memory and then calls ComfyUI's
normal diffusion-model state-dict loader; no merged checkpoint is written.

INT6 and INT6 ConvRot patches require the separate
`ComfyUI-PotatoForge-INT6` runtime. This repository intentionally does not
include INT6 kernels, layouts, or a dependency on the PotatoForge quantization
CLI.
