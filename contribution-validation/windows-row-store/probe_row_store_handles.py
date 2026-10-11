import ctypes
from ctypes import wintypes
from pathlib import Path
import tempfile
import time
import torch
from freetoken.kernel import _row_store

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
counts = []
def count():
    value = wintypes.DWORD()
    kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(value))
    return value.value

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "table.bin"
    table = torch.zeros((64,264),dtype=torch.uint8)
    path.write_bytes(table.numpy().tobytes())
    ids = torch.arange(64,dtype=torch.int64)
    staging = torch.empty_like(table)
    initial = count()
    print("initial",initial,flush=True)
    for iteration in range(10):
        store = _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0],
                                    rows_per_extent=64, row_bytes=264, row_stride=264,use_io_uring=False)
        for _ in range(8):
            store.stage_rows(ids.data_ptr(),64,staging.data_ptr(),0)
            store.flush(0)
        del store
        print(iteration,count(),flush=True)
        time.sleep(0.1)
        settled = count()
        counts.append(settled)
        print("settled",settled,flush=True)
    assert counts == [initial] * len(counts), (initial, counts)
