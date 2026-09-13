# ComfyUI-PotatoForge

ComfyUI runtime nodes for PotatoForge quantization patches. A patch is a
Safetensors artifact that replaces complete serialized quantized layer families
while ComfyUI loads a diffusion model; it is not a LoRA or a model patch.

## Purpose

This repository provides the ComfyUI runtime side of PotatoForge: it loads
diffusion models with validated, ordered quantization patches and can collect
activation statistics for offline analysis.

Patch generation happens in the separate
[potatoforge-quantization](https://github.com/bakapotatolord/potatoforge-quantization)
repository, which creates the Safetensors patch files consumed by these nodes.

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

## Available nodes

- **PotatoForge Add Quant Patch** — Validates a quant patch and appends it to
  the immutable patch stack.
- **PotatoForge Load Diffusion Model + Patches** — Loads a diffusion model and
  applies the connected quant patches in stack order.
- **PotatoForge Activation Calibration** — Attaches to selected Linear layers
  and collects per-input-channel activation energy during the workflow.
- **PotatoForge Finalize Activation Calibration** — Removes calibration hooks,
  saves the collected statistics and metadata, and passes the latent through.

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

## Activation calibration

Use `PotatoForge Activation Calibration` between a diffusion-model loader and
`KSampler`, then connect its `session` output to
`PotatoForge Finalize Activation Calibration` after `KSampler`. The finalizer
passes the latent through unchanged and writes one pair of files under
ComfyUI's output directory:

```text
potatoforge_calibration/<session>_<id>.safetensors
potatoforge_calibration/<session>_<id>.json
```

V1 observes native or Comfy diffusion modules that are semantically named
`Linear` and expose a rank-2 logical weight. It preserves the legacy raw FP32
per-input-channel `<logical_weight_name>.sum_x2` vector and adds aligned
per-evaluation input moments (`sum_x`, `sum_x2`, `max_abs_x`), actual Linear
output moments (`sum_y`, `sum_y2`), exact sample/invocation counts, and optional
root input/output energy diagnostics. An evaluation means one complete
diffusion-model forward; it is not guaranteed to equal one KSampler step.

The node does not store full activations. By default it stores two deterministic
FP32 sentinel input rows per layer/evaluation; set
`sample_rows_per_evaluation=0` to omit those actual-vector samples while keeping
all V1 moments. Timestep/sigma metadata is best-effort. The artifact is intended
for offline analysis, enabling reducers such as P95 or CVaR without claiming
that the custom node calculates them. The V1 artifact scales with selected layers,
evaluations, and feature widths.

`baseline_label` describes the model that actually ran, including an INT8
ConvRot baseline. The captured basis is the logical input to each Linear layer,
so later quantization comparisons must use the same logical basis.

The offline consumer should validate `W` and `Wq` as
`[out_features, in_features]` and each `sum_x2` vector as
`[in_features]` before calculating activation-weighted reconstruction error.

The nodes collect activations only. They do not quantize weights, optimize
scales, run candidate image generations, or retain full activation tensors.

INT6 and INT6 ConvRot patches require the separate
`ComfyUI-PotatoForge-INT6` runtime. This repository intentionally does not
include INT6 kernels, layouts, or a dependency on the PotatoForge quantization
CLI.
