"""Agent written: exact tier masks against the archived accepted local120."""
import ast
import json
import os
from pathlib import Path
import unittest

source=Path(__file__).with_name("audit-authoritative-g3-local.py")
tree=ast.parse(source.read_text())
gate=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="validate_cache_mask")
scope={}
exec(compile(ast.Module(body=[gate],type_ignores=[]),str(source),"exec"),scope)
validate=scope["validate_cache_mask"]
reference=Path(os.environ["G3_AUDIT_LOCAL_REFERENCE"]) if os.environ.get("G3_AUDIT_LOCAL_REFERENCE") else None

@unittest.skipUnless(reference is not None and reference.is_dir(),"Set G3_AUDIT_LOCAL_REFERENCE to the restored accepted local120 directory; no synthetic performance data is substituted.")
class Masks(unittest.TestCase):
 def test_all120_actual_masks_exact(self):
  rows=[json.loads(line) for path in reference.glob("context-*/samples.jsonl") for line in path.read_text().splitlines()]
  self.assertEqual(len(rows),120)
  for row in rows:
   validate(row["cache"],"local_ssd" if row["scenario"]=="ssd" else row["scenario"],row["context_tokens"])
 def test_isolated_but_overlong_cache_hit_rejected(self):
  for tier,field in [("gpu","device"),("host","host"),("local_ssd","storage")]:
   cache=dict(device=0,host=0,storage=0,cached_tokens=449)
   cache[field]=449
   with self.assertRaises(AssertionError):validate(cache,tier,512)
 def test_inconsistent_total_rejected(self):
  with self.assertRaises(AssertionError):validate(dict(device=448,host=0,storage=0,cached_tokens=449),"gpu",512)
 def test_wrong_tier_contamination_rejected(self):
  with self.assertRaises(AssertionError):validate(dict(device=448,host=1,storage=0,cached_tokens=449),"gpu",512)

if __name__=="__main__":unittest.main(verbosity=2)
