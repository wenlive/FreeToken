"""Check exact bytes, duplicate destinations, eviction, disable/reset and extent boundaries."""
from pathlib import Path
import tempfile
import torch
import pytest
_row_store=pytest.importorskip('freetoken.kernel._row_store')


@pytest.mark.parametrize('cols',[90,160,264])
def test_cached_rows_preserve_payloads_and_storage_boundaries(cols):
    native=_row_store
    rows=257
    torch.manual_seed(613)
    original=torch.randint(0,256,(rows,cols),dtype=torch.uint8)
    results=[]
    with tempfile.TemporaryDirectory(prefix='ple-row-cache-') as folder:
        path=Path(folder)/'rows.bin'
        path.write_bytes(bytes(original.flatten().tolist())+bytes(8192))
        store=native.RowStore([str(path)],[0],[0],rows,cols,cols,False)

        def read(ids,stride=0):
            index=torch.tensor(ids,dtype=torch.int64)
            actual_stride=stride or cols
            out=torch.full((len(ids),actual_stride),173,dtype=torch.uint8)
            store.stage_rows(index.data_ptr(),len(ids),out.data_ptr(),stride)
            store.flush()
            assert torch.equal(out[:,:cols],original[index])
            if actual_stride>cols:assert torch.all(out[:,cols:]==173)
            return out

        store.set_row_cache(64)
        read([0,1,1,45,46,90,91,256],cols+11)
        before=store.io_stats()
        read([0,1,1,45,46,90,91,256])
        after=store.io_stats()
        assert after['read_requests_completed']==before['read_requests_completed']
        assert after['row_cache_hits']-before['row_cache_hits']==8
        for start in range(0,rows,13):read(list(range(start,min(start+31,rows))))
        read([0,45,46,255,256],cols+7)
        assert store.io_stats()['row_cache_capacity']==64
        for capacity in (8,0,128):
            store.set_row_cache(capacity)
            read([0,1,256]);baseline=store.io_stats();read([0,1,256]);current=store.io_stats()
            assert current['row_cache_capacity']==capacity
            if capacity==0:assert current['read_requests_completed']>baseline['read_requests_completed']
            else:assert current['read_requests_completed']==baseline['read_requests_completed']
            results.append(dict(capacity=capacity,stats=current))
        index=torch.tensor([2],dtype=torch.int64);out=torch.empty((1,cols),dtype=torch.uint8)
        store.stage_rows(index.data_ptr(),1,out.data_ptr())
        try:store.set_row_cache(8)
        except RuntimeError:pass
        else:raise AssertionError('Unflushed cache resize must reject')
        store.flush();assert torch.equal(out,original[2:3])
        for capacity in (-1,1<<25):
            try:store.set_row_cache(capacity)
            except ValueError:pass
            else:raise AssertionError('Invalid cache capacity accepted')
        del store
