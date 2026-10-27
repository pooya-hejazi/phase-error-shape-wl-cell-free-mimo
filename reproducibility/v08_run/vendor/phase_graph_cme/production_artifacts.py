"""Fail-closed artifact and lifecycle primitives for production campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

import numpy as np


THREAD_ENVIRONMENT_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256(header + b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def require_single_thread_runtime() -> dict[str, str]:
    observed = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
    invalid = {name: value for name, value in observed.items() if value != "1"}
    if invalid:
        raise RuntimeError(
            "all BLAS/thread environment variables must be the literal string '1': "
            f"{invalid}"
        )
    return {name: "1" for name in THREAD_ENVIRONMENT_VARIABLES}


def peak_process_rss_bytes() -> int:
    """Return the low-overhead process peak RSS on Windows/POSIX."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = psapi.GetProcessMemoryInfo
        query.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        query.restype = wintypes.BOOL
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        result = int(counters.PeakWorkingSetSize)
    else:
        import resource

        maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result = maximum if sys.platform == "darwin" else maximum * 1024
    if result <= 0:
        raise RuntimeError("peak process RSS measurement is unavailable")
    return result


def resolve_inside_absent(project_root: Path, relative: str) -> Path:
    """Resolve a new path inside the project and reject symlinked ancestry."""

    root = project_root.resolve(strict=True)
    candidate = root.joinpath(*relative.split("/") ).resolve(strict=False)
    if root not in candidate.parents:
        raise RuntimeError("resolved output path leaves project root")
    current = root
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise RuntimeError("output ancestry contains a symbolic link")
    return candidate


def write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
        default=json_default,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
        default=json_default,
    ) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl_record(
    handle: TextIO,
    value: Mapping[str, Any],
    *,
    ordered_fields: tuple[str, ...] | None = None,
) -> None:
    if ordered_fields is not None and tuple(value) != ordered_fields:
        raise RuntimeError("JSONL record does not match the registered ordered schema")
    handle.write(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            default=json_default,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def write_jsonl_exclusive(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    ordered_fields: tuple[str, ...] | None = None,
) -> int:
    """Write a deterministic registry with one final durability sync."""

    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration as error:
        # Refuse the empty registry before creating a misleading zero-byte
        # artifact that would survive the raised exception.
        raise RuntimeError("refusing to write an empty JSONL registry") from error

    count = 0
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for value in chain((first,), iterator):
            if ordered_fields is not None and tuple(value) != ordered_fields:
                raise RuntimeError(
                    "JSONL record does not match the registered ordered schema"
                )
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    default=json_default,
                )
                + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def output_tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


@dataclass
class AtomicRunDirectory:
    """Same-filesystem STAGING/FAILED/COMPLETE lifecycle.

    The caller owns payload creation.  Any exception after ``start`` preserves
    the entire staging tree under ``*.FAILED`` with an exclusive failure marker.
    """

    project_root: Path
    output_relative: str
    experiment_id: str

    def __post_init__(self) -> None:
        self.output = resolve_inside_absent(self.project_root, self.output_relative)
        self.staging = self.output.with_name(f".{self.output.name}.STAGING")
        self.failed = self.output.with_name(f"{self.output.name}.FAILED")
        self.incomplete = self.output.with_name(f"{self.output.name}.INCOMPLETE")
        for path in (self.output, self.staging, self.failed, self.incomplete):
            if path.exists():
                raise FileExistsError(f"refusing overwrite or retry: {path}")
        self._started = False
        self._finished = False

    def start(self, marker: Mapping[str, Any]) -> Path:
        if self._started:
            raise RuntimeError("run lifecycle has already started")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(exist_ok=False)
        self._started = True
        write_json_exclusive(
            self.staging / "RUNNING.json",
            {"status": "RUNNING", "experiment_id": self.experiment_id, **marker},
        )
        return self.staging

    def complete(self, marker: Mapping[str, Any]) -> Path:
        if not self._started or self._finished:
            raise RuntimeError("run lifecycle is not open")
        write_json_atomic(
            self.staging / "COMPLETE.json",
            {"status": "COMPLETE", "experiment_id": self.experiment_id, **marker},
        )
        (self.staging / "RUNNING.json").unlink()
        os.replace(self.staging, self.output)
        self._finished = True
        return self.output

    def fail(self, error: BaseException, marker: Mapping[str, Any]) -> Path | None:
        if not self._started or self._finished or not self.staging.exists():
            return None
        for temporary in self.staging.glob(".*.tmp"):
            try:
                temporary.unlink()
            except OSError:
                pass
        failure = {
            "status": "FAILED",
            "experiment_id": self.experiment_id,
            "error_type": type(error).__name__,
            "error": str(error),
            **marker,
        }
        try:
            write_json_exclusive(self.staging / "FAILED.json", failure)
            os.replace(self.staging, self.failed)
            self._finished = True
            return self.failed
        except BaseException:
            return None
