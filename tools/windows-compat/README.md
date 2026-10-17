# Windows compatibility shim

The upstream AOT sources use the POSIX header `<sys/cdefs.h>` for the
`__always_inline` annotation.  MSVC does not ship that header, so the Windows
builder adds this directory to the CUDA include path.  The shim only defines
the annotation in terms of MSVC/CUDA's `__forceinline__`; it does not replace
any platform headers or change the kernel ABI.

Keep this directory in the repository.  Removing it causes the Windows build
to fail before nvcc starts compiling the kernels.
