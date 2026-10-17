#!/usr/bin/env python3
"""Build FreeToken's TVM FFI kernel cache as Windows DLLs.

The upstream cache wheel builder is written for Linux and collects ``.so``
files.  The kernel implementation itself is platform-neutral: ``tvm_ffi``
uses MSVC/nvcc and emits ``.dll`` on Windows.  This helper keeps the upstream
spec table and compiler calls intact, then collects the Windows outputs into a
portable cache tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "windows-kernel-cache",
        help="destination cache tree (default: %(default)s)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build" / "windows-kernel-cache",
        help="per-spec compiler scratch directory (default: %(default)s)",
    )
    parser.add_argument(
        "--spec",
        action="append",
        dest="specs",
        help="build one spec; may be repeated. The default builds all specs.",
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        help="newline-delimited spec names; relative paths use the repository root",
    )
    parser.add_argument(
        "--arch",
        default="12.0",
        help="space-separated CUDA compute capabilities for the fatbin (default: 12.0)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of concurrent spec builds (default: 1)",
    )
    parser.add_argument(
        "--keep-build",
        action="store_true",
        help="keep compiler scratch files after a successful build",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the upstream default spec names and exit",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dll_candidates(build_dir: Path, name: str) -> list[Path]:
    candidates = [
        path
        for path in build_dir.rglob("*.dll")
        if path.is_file() and path.name.lower() == f"{name.lower()}.dll"
    ]
    if candidates:
        return candidates
    # Keep the fallback broad enough for a future tvm_ffi naming change while
    # excluding runtime dependencies copied beside the output.
    return [
        path
        for path in build_dir.rglob("*.dll")
        if path.is_file() and path.stem.lower().startswith(name.lower())
    ]


def _load_specs():
    sys.path.insert(0, str(ROOT / "python"))
    from freetoken.kernel.aot import default_kernel_specs

    return default_kernel_specs()


def _read_spec_file(path: Path) -> list[str]:
    path = path if path.is_absolute() else ROOT / path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read --spec-file {path}: {exc}") from exc
    names = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not names:
        raise SystemExit(f"--spec-file is empty: {path}")
    return names


def _patch_tvm_ffi_windows_flags() -> None:
    """Apply the small Windows/CUDA compatibility fixes needed by this build.

    That release emits ``-Xcompiler /std:c++17 /O2`` in Ninja.  CUDA 12.8's
    Windows parser treats the second MSVC flag as another input file, and the
    C++17 flag overrides FreeToken's required C++20 mode.  The comma form is
    accepted by CUDA and forwards the corrected flags to cl.exe.  FreeToken's
    portable headers use the GCC-style ``__always_inline`` spelling, which is
    not predefined by nvcc on this Windows toolchain, so map it to CUDA's
    ``__forceinline__`` spelling as well.  Finally, the Windows link rule in
    this tvm_ffi release does not add CUDA's import library, so add
    ``cudart.lib`` from ``CUDA_HOME``.  Keep all workarounds local to this
    builder. The companion ``jit`` sources carry one small source-level
    compatibility edit: redundant template arguments on the symbolic
    ``with_dtype`` overload are removed because nvcc 12.8 rejects that spelling
    on Windows. The edit is behavior-preserving; the overload ignores those
    template arguments when a symbolic dtype is supplied.
    """
    if sys.platform != "win32":
        return
    import tvm_ffi.cpp.extension as extension

    original = extension._generate_ninja_build
    if getattr(original, "_freetoken_windows_patch", False):
        return

    def generate(*args, **kwargs):
        text = original(*args, **kwargs)
        # tvm_ffi emits POSIX-style flags in the MSVC rule as well.  cl.exe
        # ignores ``-std``/``-O3`` and keeps its leading C++17 setting, which
        # breaks the C++20-only radix source.  Rewrite only the cxxflags line;
        # the CUDA rule needs its own ``-std=c++20`` flag below.
        cxx_lines = []
        for line in text.splitlines(keepends=True):
            if line.startswith("cxxflags = "):
                line = line.replace("/std:c++17", "/std:c++20")
                line = line.replace(" -std=c++20", "")
                line = line.replace(" -O3", " /O2")
            cxx_lines.append(line)
        text = "".join(cxx_lines)
        text = text.replace(
            "-Xcompiler /std:c++17 /O2",
            "-Xcompiler=/std:c++20,/O2",
        )
        if "-D__always_inline=" not in text:
            text = text.replace(
                "cuda_cflags = ",
                "cuda_cflags = -D__always_inline=__forceinline__ ",
                1,
            )
        compat_include = ROOT / "tools" / "windows-compat"
        compat_flag = f"-I{compat_include}"
        if compat_flag not in text:
            text = text.replace(
                "cuda_cflags = ",
                f"cuda_cflags = {compat_flag} ",
                1,
            )
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home and "cuda_cflags = " in text and "cudart.lib" not in text:
            cuda_lib = Path(cuda_home) / "lib" / "x64"
            text = text.replace(
                "ldflags = ",
                f"ldflags = /LIBPATH:{cuda_lib} cudart.lib ",
                1,
            )
        return text

    generate._freetoken_windows_patch = True
    extension._generate_ninja_build = generate


def _build_one(spec, build_root: Path, output_root: Path) -> dict[str, object]:
    scratch = build_root / spec.name
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[build] {spec.name}", flush=True)
    spec.build(scratch)

    candidates = _dll_candidates(scratch, spec.name)
    if len(candidates) != 1:
        found = ", ".join(str(path) for path in candidates) or "<none>"
        raise RuntimeError(
            f"expected one DLL for {spec.name}, found {len(candidates)}: {found}"
        )
    source = candidates[0]
    destination_dir = output_root / spec.name
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{spec.name}.dll"
    shutil.copy2(source, destination)
    size = destination.stat().st_size
    digest = _sha256(destination)
    print(f"[done ] {spec.name} size={size} sha256={digest}", flush=True)
    return {
        "name": spec.name,
        "path": str(destination.relative_to(output_root)).replace("\\", "/"),
        "size": size,
        "sha256": digest,
    }


def main() -> int:
    args = _parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")

    # Force source compilation even if a cache wheel happens to be installed.
    os.environ["FREETOKEN_DISABLE_KERNEL_CACHE"] = "1"
    os.environ.pop("FREETOKEN_DISABLE_JIT", None)
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = args.arch
    os.environ.setdefault("FREETOKEN_KERNEL_CACHE_JOBS", str(args.jobs))
    _patch_tvm_ffi_windows_flags()

    specs = _load_specs()
    by_name = {spec.name: spec for spec in specs}
    if args.list:
        if args.specs and args.spec_file:
            raise SystemExit("use either --spec or --spec-file, not both")
        names = (
            args.specs
            if args.specs
            else _read_spec_file(args.spec_file)
            if args.spec_file
            else [spec.name for spec in specs]
        )
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise SystemExit("unknown spec(s): " + ", ".join(unknown))
        for name in names:
            print(name)
        return 0
    if args.specs and args.spec_file:
        raise SystemExit("use either --spec or --spec-file, not both")
    selected_names = (
        args.specs
        if args.specs
        else _read_spec_file(args.spec_file)
        if args.spec_file
        else [spec.name for spec in specs]
    )
    unknown = [name for name in selected_names if name not in by_name]
    if unknown:
        raise SystemExit(
            "unknown spec(s): " + ", ".join(unknown) + "\nvalid names:\n" + "\n".join(by_name)
        )

    output_root = args.output.resolve()
    build_root = args.build_dir.resolve()
    if output_root == ROOT or build_root == ROOT:
        raise SystemExit("refusing to use the repository root as an output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    # A sequential build is deliberate by default: each nvcc/MSVC invocation is
    # large, and it keeps logs and GPU/toolchain failures attributable to one spec.
    records = []
    for name in selected_names:
        records.append(_build_one(by_name[name], build_root, output_root))

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except Exception as exc:  # pragma: no cover - metadata must not hide DLLs
        torch_version = f"unavailable: {exc}"
        torch_cuda = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metadata = {
        "format": 1,
        "platform": "windows-x86_64",
        "source_commit": commit,
        "python": sys.version,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "cuda_arches": args.arch.split(),
        "artifacts": records,
    }
    (output_root / "build-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"built {len(records)} DLL(s) under {output_root}", flush=True)
    if not args.keep_build:
        shutil.rmtree(build_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
