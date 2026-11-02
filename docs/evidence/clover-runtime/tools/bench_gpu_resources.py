"""Separate diagnostic campaign: whole-device NVML counters, not primary timings.

NVML reference: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
GetPcieThroughput reports KB/s over a 20ms window. Integrated bytes are estimates,
include desktop traffic and protocol overhead, and cannot identify experts.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--resource-frontend', choices=['native', 'stock'], required=True)
args, rest = p.parse_known_args()
sys.argv = [sys.argv[0], *rest]
output = Path(rest[rest.index('--output') + 1])
sys.path.insert(0, 'C:/workspace/wt/T-20261102-clover-q8/FreeToken/benchmarks')
nvml = c.WinDLL('C:/Windows/System32/nvml.dll')
assert nvml.nvmlInit_v2() == 0
handle = c.c_void_p()
assert nvml.nvmlDeviceGetHandleByIndex_v2(0, c.byref(handle)) == 0
class Memory(c.Structure):
    _fields_ = [('total', c.c_ulonglong), ('free', c.c_ulonglong), ('used', c.c_ulonglong)]
samples, intervals = [], []
stop = threading.Event()
def observe():
    while not stop.is_set():
        rx, tx, memory = c.c_uint(), c.c_uint(), Memory()
        codes = [nvml.nvmlDeviceGetPcieThroughput(handle, 1, c.byref(rx)),
            nvml.nvmlDeviceGetPcieThroughput(handle, 0, c.byref(tx)),
            nvml.nvmlDeviceGetMemoryInfo(handle, c.byref(memory))]
        samples.append(dict(t=time.perf_counter(), rx_KBps=rx.value, tx_KBps=tx.value,
            device_used_bytes=memory.used, status=codes))
        stop.wait(.02)
thread = threading.Thread(target=observe, daemon=True)
thread.start()
time.sleep(1)
if args.resource_frontend == 'native':
    import bench_clover as bench
    original = bench.LLM
    def instrumented(*a, **kw):
        llm = original(*a, **kw)
        generate = llm.generate
        def measured(*gargs, **gkw):
            start = time.perf_counter()
            result = generate(*gargs, **gkw)
            intervals.append(dict(start=start, end=time.perf_counter()))
            return result
        llm.generate = measured
        return llm
    bench.LLM = instrumented
else:
    import bench_clover_llama as bench
    original = bench.urllib.request.urlopen
    class ObservedResponse:
        def __init__(self, response, start):
            self.response, self.start = response, start
        def __enter__(self):
            return self.response.__enter__()
        def __exit__(self, *a):
            result = self.response.__exit__(*a)
            intervals.append(dict(start=self.start, end=time.perf_counter()))
            return result
    def measured(request, *a, **kw):
        start = time.perf_counter()
        response = original(request, *a, **kw)
        if getattr(request, 'full_url', '').endswith('/completion'):
            return ObservedResponse(response, start)
        return response
    bench.urllib.request.urlopen = measured
try:
    bench.main()
finally:
    stop.set()
    thread.join()
    nvml.nvmlShutdown()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / 'results.json'
    native = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else {}
    runs = native.get('runs', [])
    for index, interval in enumerate(intervals):
        selected = [s for s in samples if interval['start'] <= s['t'] <= interval['end']]
        rx = tx = 0.
        for left, right in zip(samples, samples[1:]):
            begin, end = max(left['t'], interval['start']), min(right['t'], interval['end'])
            if end > begin:
                rx += (end - begin) * (left['rx_KBps'] + right['rx_KBps']) / 2
                tx += (end - begin) * (left['tx_KBps'] + right['tx_KBps']) / 2
        interval.update(rx_integrated_KB=rx, tx_integrated_KB=tx,
            rx_estimated_bytes_KB1024=rx * 1024, tx_estimated_bytes_KB1024=tx * 1024,
            device_peak_used_bytes=max((s['device_used_bytes'] for s in selected), default=None),
            counter_window_seconds=.02, samples=len(selected),
            actual_sample_gap_max_seconds=max((b['t'] - a['t'] for a, b in zip(selected, selected[1:])), default=None))
        if index < len(runs):
            interval.update({k: runs[index][k] for k in ('prompt', 'mode', 'width', 'repeat', 'warmup')})
    record = dict(state='complete' if len(runs) == len(intervals) and all(s['status'] == [0, 0, 0] for s in samples) else 'incomplete',
        scope='Separate instrumented diagnostic; whole-device physical counter estimate, includes desktop and nonexpert bytes; primary medians are uninstrumented',
        source='https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html',
        raw_unit='NVML KB/s; byte estimates explicitly use 1024 bytes/KB',
        integration='trapezoidal, actual sample gaps; counters smooth a 20ms window, no exact per-request byte count claim',
        sampling_note='20ms is the driver counter window, not the achieved polling interval; inspect actual gaps (Windows NVML calls can block)',
        result_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest() if results_path.exists() else None,
        device_peak_used_bytes=max((s['device_used_bytes'] for s in samples), default=None),
        intervals=intervals, samples=samples)
    (output / 'physical-gpu-resources.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
