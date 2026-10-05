"""Agent written: real archived-artifact rejection regressions; no serving calls."""
import importlib.util,json,tempfile,tarfile,shutil,unittest,contextlib,io,os
from pathlib import Path
spec=importlib.util.spec_from_file_location("g3audit",Path(__file__).with_name("audit-authoritative-g3-remote.py"))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
ROOT=Path(os.environ["G3_AUDIT_FIXTURE_ROOT"]) if os.environ.get("G3_AUDIT_FIXTURE_ROOT") else None
RUN="20261005-a10-b7-remote-staged"
@unittest.skipUnless(ROOT is not None and (ROOT/"a10-b7-staged-audited-requester.tar.gz").exists() and (ROOT/"a10-b7-staged-owner-ready.tar.gz").exists(),"Set G3_AUDIT_FIXTURE_ROOT to the saved accepted A10/B7 archive directory; no synthetic performance fixture is substituted.")
class GuardTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(prefix="g3-audit-tests-",dir=os.environ.get("NIXLSHARD_TEST_DIR"));cls.root=Path(cls.tmp.name)
  cls.reference=cls.root/"reference";cls.reference.mkdir()
  with tarfile.open(ROOT/"a10-b7-staged-audited-requester.tar.gz") as t:t.extractall(cls.reference,filter="data")
  owner=cls.root/"owner";owner.mkdir()
  with tarfile.open(ROOT/"a10-b7-staged-owner-ready.tar.gz") as t:t.extractall(owner,filter="data")
  target=cls.root/"owner-artifacts";target.mkdir(parents=True,exist_ok=True)
  cls.previous_owner_path=os.environ.get("AUDIT_OWNER_ARTIFACTS");os.environ["AUDIT_OWNER_ARTIFACTS"]=str(target)
  cls.previous_native=a.NATIVE;a.NATIVE="3310a0cd924619f60f42978247c1817206a1f008"
  for f in ["config.json","runtime-maps.json","domain-witness.json"]:shutil.copyfile(owner/f,target/("owner-"+f))
  with contextlib.redirect_stdout(io.StringIO()):a.audit(cls.reference)
 @classmethod
 def tearDownClass(cls):
  a.NATIVE=cls.previous_native
  if cls.previous_owner_path is None:os.environ.pop("AUDIT_OWNER_ARTIFACTS",None)
  else:os.environ["AUDIT_OWNER_ARTIFACTS"]=cls.previous_owner_path
  cls.tmp.cleanup()
 def setUp(self):
  self.role=self.root/self._testMethodName;shutil.copytree(self.reference,self.role)
 def invoke(self):
  with contextlib.redirect_stdout(io.StringIO()):a.audit(self.role,self.reference)
 def mutate_normal(self,callback):
  p=self.role/"remote-full/request-diagnostics.json";n=json.loads(p.read_text());r=next(x for x in n["samples"] if x["tier"]=="remote_ssd");callback(r);p.write_text(json.dumps(n))
 def test_unchanged_actual_cohort_accepts(self):self.invoke()
 def test_full_generation_counter_tampering_rejected(self):
  p=self.role/"remote-full/0001-512-remote/owner-after-stats.json";x=json.loads(p.read_text());k=next(k for k in x["metrics"] if "component=\"metadata_read\"" in k and "bytes_total" in k);x["metrics"][k]+=4096;p.write_text(json.dumps(x))
  with self.assertRaises(AssertionError):self.invoke()
 def test_native_metadata_byte_forgery_rejected(self):
  def bad(r):
   e=next(e for b in r["native_batches"] for e in b["events"] if e["stage"]=="remote_rpc");e["owner_metadata_bytes"]=0
  self.mutate_normal(bad)
  with self.assertRaises(AssertionError):self.invoke()
 def test_late_dma_interval_rejected(self):
  def bad(r):
   f=next(e["start_ns"] for e in r["trace_events"] if e["stage"]=="first_forward_entry")
   e=next(e for b in r["native_batches"] for e in b["events"] if e["stage"]=="remote_rpc");e["end_ns"]=f+1
  self.mutate_normal(bad)
  with self.assertRaises(AssertionError):self.invoke()
 def test_changed_sampling_cannot_replay(self):
  p=self.role/"remote-full/0001-512-remote/measured-request.json";x=json.loads(p.read_text());x["sampling_params"]["temperature"]=0.1;p.write_text(json.dumps(x))
  with self.assertRaises(AssertionError):self.invoke()
 def test_unapproved_geometry_change_rejected(self):
  p=self.role/"config.json";x=json.loads(p.read_text());x["agent"]["staging_slots"]=8;p.write_text(json.dumps(x))
  with self.assertRaises(AssertionError):self.invoke()
if __name__=="__main__":unittest.main(verbosity=2)
