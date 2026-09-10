from .potatoforge_nodes.quant_patches.nodes import (
    PotatoForgeAddQuantPatch,
    PotatoForgePatchedDiffusionModelLoader,
    register_patch_folder,
)
from .potatoforge_nodes.activation_calibration import (
    PotatoForgeActivationCalibration,
    PotatoForgeFinalizeActivationCalibration,
)


register_patch_folder()

NODE_CLASS_MAPPINGS = {
    "PotatoForgeAddQuantPatch": PotatoForgeAddQuantPatch,
    "PotatoForgePatchedDiffusionModelLoader": PotatoForgePatchedDiffusionModelLoader,
    "PotatoForgeActivationCalibration": PotatoForgeActivationCalibration,
    "PotatoForgeFinalizeActivationCalibration": PotatoForgeFinalizeActivationCalibration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PotatoForgeAddQuantPatch": "PotatoForge Add Quant Patch",
    "PotatoForgePatchedDiffusionModelLoader": "PotatoForge Load Diffusion Model + Patches",
    "PotatoForgeActivationCalibration": "PotatoForge Activation Calibration",
    "PotatoForgeFinalizeActivationCalibration": "PotatoForge Finalize Activation Calibration",
}
