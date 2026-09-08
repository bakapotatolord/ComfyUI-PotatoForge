from pathlib import Path
import unittest

from potatoforge_nodes.quant_patches.manifest import QuantPatchManifest
from potatoforge_nodes.quant_patches.stack import QuantPatchRef, QuantPatchStack


def patch_ref(name: str) -> QuantPatchRef:
    return QuantPatchRef(
        name=name,
        path=Path(f"C:/{name}"),
        manifest=QuantPatchManifest(name, (name,)),
    )


class QuantPatchStackTests(unittest.TestCase):
    def test_appending_keeps_parent_stack_unchanged(self):
        base = QuantPatchStack().append(patch_ref("A"))

        branch_b = base.append(patch_ref("B"))
        branch_c = base.append(patch_ref("C"))

        self.assertEqual([ref.name for ref in base.patches], ["A"])
        self.assertEqual([ref.name for ref in branch_b.patches], ["A", "B"])
        self.assertEqual([ref.name for ref in branch_c.patches], ["A", "C"])


if __name__ == "__main__":
    unittest.main()
