"""Build the selected extension independently of unrelated Windows setup failures."""
import argparse
import os
from pathlib import Path
import runpy
import sys

import setuptools
import torch.utils.cpp_extension as cpp_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--baseline-source", type=Path)
parser.add_argument("--with-pinned", action="store_true")
parser.add_argument("--cpu-output", type=Path)
args = parser.parse_args()
os.chdir(args.root)
sys.argv = ["setup.py", "build_ext", "--inplace"]
original_setup = setuptools.setup

if args.cpu_output:
    output = args.cpu_output.resolve()
    sys.argv = ["setup.py", "build_ext", "--build-lib", str(output),
                "--build-temp", str(output / "temp")]
    original_setup(ext_modules=[CppExtension(
        "freetoken.kernel._row_store",
        [str((args.baseline_source or Path("python/freetoken/kernel/csrc/row_store/row_store_ext.cpp")).resolve())],
        extra_compile_args=["/O2", "/std:c++20"],
    )], cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)})
    raise SystemExit(0)

def selected_setup(**kwargs):
    if args.baseline_source:
        kwargs["ext_modules"] = [CppExtension(
            "freetoken.kernel._row_store", [str(args.baseline_source)],
            extra_compile_args=["/O2", "/std:c++20"],
        )]
    else:
        all_extensions = kwargs["ext_modules"]
        kwargs["ext_modules"] = [ext for ext in all_extensions if ext.name.endswith("._row_store")]
        assert len(kwargs["ext_modules"]) == 1
        if args.with_pinned:
            pinned = next(ext for ext in all_extensions if ext.name.endswith("._pinned_tensor"))
            pinned.extra_compile_args = ["/O2", "/std:c++20"]
            pinned.include_dirs = [str(Path(directory).resolve()) for directory in pinned.include_dirs]
            pinned.library_dirs.append(str(Path(os.environ["CUDA_HOME"]) / "lib" / "x64"))
            kwargs["ext_modules"].append(pinned)
    return original_setup(**kwargs)

setuptools.setup = selected_setup
runpy.run_path("setup.py", run_name="__main__")
