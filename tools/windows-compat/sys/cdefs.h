#pragma once

// FreeToken's CUDA headers use the Linux spelling for this inline hint.
// nvcc on Windows provides the equivalent MSVC/CUDA spelling instead.
#ifndef __always_inline
#define __always_inline __forceinline__
#endif
