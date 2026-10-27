"""Small fail-closed lifecycle helpers for T06B, without algorithm execution."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

THREAD_VARIABLES=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_path(root, relative):
    root=Path(root).resolve(strict=True)
    p=Path(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("unsafe relative artifact path")
    current=root
    for part in p.parts:
        current=current/part
        if current.is_symlink() or current.is_junction():
            raise ValueError("linked artifact path")
    if root not in current.resolve().parents:
        raise ValueError("artifact outside root")
    return current

def json_read(path):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out: raise ValueError("duplicate JSON key")
            out[key]=value
        return out
    def bad(value):raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=pairs,parse_constant=bad)

def runtime_fingerprint():
    import io
    import contextlib
    import numpy as np
    import scipy
    import psutil
    text=io.StringIO()
    with contextlib.redirect_stdout(text):np.show_config()
    return dict(python=sys.version,executable=str(Path(sys.executable).resolve()),
        numpy=np.__version__,scipy=scipy.__version__,psutil=psutil.__version__,
        platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),
        cpu_count=os.cpu_count(),blas_config=text.getvalue(),
        thread_environment={k:os.environ.get(k) for k in THREAD_VARIABLES})

def verify_sources(root, entries):
    paths=[r["path"] for r in entries]
    if not paths or len(paths)!=len(set(paths)):raise ValueError("empty/duplicate source inventory")
    for row in entries:
        path=safe_path(root,row["path"])
        if not path.is_file() or path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:
            raise ValueError("source identity mismatch: "+row["path"])

class ResourceLimitError(RuntimeError):pass

class RuntimeGuard:
    """Watchdog records process-tree RSS; callbacks stop computation safely."""
    def __init__(self,output,policy,*,clock=time.perf_counter,rss_reader=None):
        import psutil
        self.output=Path(output);self.policy=policy;self.clock=clock;self.start=clock()
        self.peak_rss=0;self.failure=None;self.closed=False
        self.stop_event=threading.Event()
        process=psutil.Process()
        def real_rss():
            children=process.children(recursive=True)
            if children:raise ResourceLimitError("unregistered child process")
            info=process.memory_info()
            return max(info.rss,getattr(info,"peak_wset",info.rss))
        self.rss_reader=rss_reader or real_rss
        self.thread=threading.Thread(target=self._monitor,daemon=True)
        self.thread.start()
    def _monitor(self):
        while not self.stop_event.wait(self.policy["resource_poll_seconds"]):
            try:self._sample()
            except Exception as error:self.failure=error;return
    def _sample(self):
        rss=int(self.rss_reader());self.peak_rss=max(self.peak_rss,rss)
        if rss>self.policy["rss_cap_bytes"]:raise ResourceLimitError("RSS cap")
        if self.elapsed()>self.policy["wall_cap_seconds"]:raise ResourceLimitError("wall cap")
    def elapsed(self):return self.clock()-self.start
    def size(self):
        if not self.output.exists():return 0
        total=0
        for p in self.output.rglob("*"):
            if p.is_symlink() or p.is_junction():raise ResourceLimitError("linked output")
            if p.is_file():total+=p.stat().st_size
        return total
    def check(self,*,finalizing=False,check_disk=False):
        if self.failure is not None:raise ResourceLimitError(str(self.failure))
        self._sample()
        reserve=0 if finalizing else self.policy["finalization_wall_reserve_seconds"]
        if self.elapsed()>self.policy["wall_cap_seconds"]-reserve:
            raise ResourceLimitError("finalization wall reserve")
        if check_disk:
            reserve=0 if finalizing else self.policy["artifact_metadata_reserve_bytes"]
            if self.size()>self.policy["artifact_cap_bytes"]-reserve:
                raise ResourceLimitError("artifact cap/reserve")
    def close(self):
        self.stop_event.set();self.thread.join(timeout=2);self.closed=True
        if self.thread.is_alive():raise ResourceLimitError("watchdog did not stop")

class RunDirectory:
    """Exclusive one-shot output; existing failed/completed siblings block retry."""
    def __init__(self,output,identity,*,kind="SCIENTIFIC_DEVELOPMENT"):
        self.final=Path(output)
        self.stage=Path(str(output)+".STAGING")
        self.identity=identity;self.kind=kind;self.published=False;self.claimed=False
    def claim(self):
        if self.kind=="SOFTWARE_FIXTURE" and not self.identity.startswith("SOFTWARE_FIXTURE_"):
            raise ValueError("fixture identity required")
        if self.kind not in ("SOFTWARE_FIXTURE","SCIENTIFIC_DEVELOPMENT"):raise ValueError("kind")
        for p in (self.final,self.stage,Path(str(self.final)+".FAILED"),Path(str(self.final)+".INCOMPLETE"),
                  Path(str(self.final)+".independent_validation.json"),Path(str(self.final)+".execution_receipt.json")):
            if p.exists() or p.is_symlink():raise FileExistsError("identity consumed")
        parent=self.final.parent
        parent.mkdir(parents=True,exist_ok=True)
        for p in (parent,*parent.parents):
            if p.is_symlink() or p.is_junction():raise ValueError("linked output ancestry")
        self.stage.mkdir(exist_ok=False);self.claimed=True
        self.write_json("RUNNING.json",dict(identity=self.identity,kind=self.kind))
        return self
    def write_json(self,name,value):
        data=canonical(value)+b"\n"
        return self.write_bytes(name,data)
    def write_bytes(self,name,data):
        if type(data) is not bytes:raise TypeError("bytes required")
        path=safe_path(self.stage,name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        return sha(path)
    def write_npz(self,name,**arrays):
        import numpy as np
        for a in arrays.values():
            if not isinstance(a,np.ndarray) or a.dtype.hasobject or not np.isfinite(a).all():
                raise ValueError("invalid NPZ array")
        path=safe_path(self.stage,name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as handle:
            np.savez(handle,**arrays);handle.flush();os.fsync(handle.fileno())
        return sha(path)
    def append_jsonl(self,name,value):
        path=safe_path(self.stage,name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("ab") as handle:
            handle.write(canonical(value)+b"\n");handle.flush()
    def complete(self,guard):
        guard.check(finalizing=True,check_disk=True)
        files=[]
        for p in sorted(self.stage.rglob("*")):
            if p.is_symlink() or p.is_junction():raise ValueError("linked publication input")
            if p.is_file():files.append(dict(path=p.relative_to(self.stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
        self.write_json("MANIFEST.json",dict(files=files))
        self.write_json("COMPLETE.json",dict(identity=self.identity,kind=self.kind,manifest_sha256=sha(self.stage/"MANIFEST.json")))
        for row in files:
            path=safe_path(self.stage,row["path"])
            if path.stat().st_size!=row["bytes"] or sha(path)!=row["sha256"]:raise ValueError("publication race")
        guard.check(finalizing=True,check_disk=True)
        if self.final.exists():raise FileExistsError("publication target exists")
        self.stage.rename(self.final);self.published=True;guard.output=self.final
        guard.check(finalizing=True,check_disk=True)
    def fail(self,error,*,incomplete=False):
        if not self.claimed:return
        location=self.final if self.published else self.stage
        # Only runner-owned postpublication failure; validator never calls this.
        if self.published and (location/"COMPLETE.json").exists():
            (location/"COMPLETE.json").rename(location/"RUNNER_REJECTED_COMPLETE.json")
        self.stage=location
        self.write_json("FAILURE.json",dict(identity=self.identity,kind=self.kind,
            status="INCOMPLETE_RESOURCE" if incomplete else "FAIL_INTEGRITY",
            error_type=type(error).__name__,message=str(error)))
        destination=Path(str(self.final)+(".INCOMPLETE" if incomplete else ".FAILED"))
        if destination.exists():raise FileExistsError("failure target exists")
        location.rename(destination)
