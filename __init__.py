from .potatoforge_nodes.quant_patches.nodes import (
    PotatoForgeAddQuantPatch,
    PotatoForgePatchedDiffusionModelLoader,
    register_patch_folder,
)


register_patch_folder()

NODE_CLASS_MAPPINGS = {
    "PotatoForgeAddQuantPatch": PotatoForgeAddQuantPatch,
    "PotatoForgePatchedDiffusionModelLoader": PotatoForgePatchedDiffusionModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PotatoForgeAddQuantPatch": "PotatoForge Add Quant Patch",
    "PotatoForgePatchedDiffusionModelLoader": "PotatoForge Load Diffusion Model + Patches",
}
