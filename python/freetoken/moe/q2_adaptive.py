"""Online Q2 owner exchange: true compressed bytes, fixed banks, graph-stable tables."""
from collections import deque
import time
import numpy as np
import torch
import triton
import triton.language as tl

@triton.jit
def _exchange(GPU, HOST, FEATS, SLOTS, COUNT, MAX_SWAPS:tl.constexpr, BLOCKS:tl.constexpr,
              UNITS:tl.constexpr):
    # No CPU expert readers run here; the caller drained all GPU users before exchanging unique pairs.
    bank=tl.program_id(0)
    block=tl.program_id(1)
    count=tl.load(COUNT)
    row_units=tl.load(FEATS+bank)//16
    gpu=tl.load(GPU+bank)
    lanes=tl.arange(0,UNITS)
    words=tl.arange(0,4)
    total=count*row_units
    for start in range(tl.cdiv(total,BLOCKS*UNITS)):
        unit=start*(BLOCKS*UNITS)+block*UNITS+lanes
        valid=unit<total
        item=unit//row_units
        slot=tl.load(SLOTS+item,valid,0).to(tl.int64)
        host=tl.load(HOST+bank*MAX_SWAPS+item,valid,0)
        column=(unit%row_units)*16
        gpu_ptr=(gpu+slot*row_units*16+column).to(tl.pointer_type(tl.uint32))
        host_ptr=(host+column).to(tl.pointer_type(tl.uint32))
        old=tl.load(gpu_ptr[:,None]+words[None,:],valid[:,None],0)
        new=tl.load(host_ptr[:,None]+words[None,:],valid[:,None],0)
        tl.store(gpu_ptr[:,None]+words[None,:],new,valid[:,None])
        tl.store(host_ptr[:,None]+words[None,:],old,valid[:,None])


@triton.jit
def _publish_exchange(COMMANDS, HOST, GPU, FEATS, TABLES, MAP, REVERSE, USAGE,
                      INITIAL, PROTECTED_REVERSE, COUNT,
                      EXPERTS:tl.constexpr, PROTECTED_BEGIN:tl.constexpr,
                      MAX_SWAPS:tl.constexpr, BANKS:tl.constexpr, BLOCK:tl.constexpr):
    # Publish only after the byte exchange on this stream. Commands are disjoint.
    item=tl.arange(0,BLOCK)
    valid=item<tl.load(COUNT)
    layer=tl.load(COMMANDS+item*4,valid,0).to(tl.int64)
    cold=tl.load(COMMANDS+item*4+1,valid,0).to(tl.int64)
    hot=tl.load(COMMANDS+item*4+2,valid,0).to(tl.int64)
    slot=tl.load(COMMANDS+item*4+3,valid,0).to(tl.int64)
    incoming=layer*EXPERTS+cold
    outgoing=layer*EXPERTS+hot
    previous=tl.load(MAP+incoming,valid,-1).to(tl.int64)
    disposable=valid&(previous>=0)&(previous<PROTECTED_BEGIN)
    tl.store(REVERSE+previous,-1,disposable)
    tl.store(USAGE+previous,0,disposable)
    tl.store(MAP+outgoing,-1,valid)
    tl.store(MAP+incoming,slot,valid)
    tl.store(REVERSE+slot,incoming,valid)
    tl.store(INITIAL+outgoing,-1,valid)
    tl.store(INITIAL+incoming,slot,valid)
    tl.store(PROTECTED_REVERSE+slot,incoming,valid)
    table=tl.load(TABLES+layer,valid,0).to(tl.pointer_type(tl.int64))
    for bank in tl.static_range(BANKS):
        row_bytes=tl.load(FEATS+bank)
        gpu=tl.load(GPU+bank)+slot*row_bytes
        host=tl.load(HOST+bank*MAX_SWAPS+item,valid,0)
        tl.store(table+bank*EXPERTS+cold,gpu,valid)
        tl.store(table+bank*EXPERTS+hot,host,valid)


class AdaptiveQ2Owners:
    def __init__(self,cache,*,every=32,max_swaps=96,decay=1.0):
        """Use small GPU command tables and exchange original host rows without full-expert staging."""
        self.cache=cache
        self.every=every
        self.max_swaps=max_swaps
        self.decay=decay
        self.calls=0
        self.rounds=deque(maxlen=256)
        p=cache.protected_payload
        self.pointer_tables=[table.cpu().numpy().copy() for table in cache._owner_tables]
        self.source_ptrs=torch.empty((len(cache.bank_schema),max_swaps),device=cache.device,dtype=torch.int64)
        self.gpu_slots=torch.empty(max_swaps,device=cache.device,dtype=torch.int32)
        self.count=torch.zeros(1,device=cache.device,dtype=torch.int64)
        self.commands=torch.empty((max_swaps,4),device=cache.device,dtype=torch.int32)
        self.owner_table_ptrs=torch.tensor([table.data_ptr() for table in cache._owner_tables],
                                         device=cache.device,dtype=torch.int64)
        self.staging_bytes=0
        self.heat=np.zeros((p.layers,p.experts),dtype=np.float32)
        self.last_counts=np.zeros((p.layers,p.experts),dtype=np.int64)
        cache.decode_freq.zero_()

    def candidates(self,freq):
        """Choose same-layer exchanges from completed routes with Strata's count and gain floors."""
        p=self.cache.protected_payload
        choices=[]
        for layer in range(p.layers):
            host=np.asarray(p.host_indices[layer])
            incoming=np.flatnonzero((host>=0)&(freq[layer]>=2))
            outgoing=np.flatnonzero(host<0)
            incoming=sorted(incoming,key=lambda e:(-float(freq[layer,e]),int(e)))
            outgoing=sorted(outgoing,key=lambda e:(float(freq[layer,e]),int(e)))
            for cold,hot in zip(incoming,outgoing):
                gain=float(freq[layer,cold])-float(freq[layer,hot])
                if gain<1.5:break
                choices.append((gain,layer,int(cold),int(hot)))
        choices.sort(key=lambda row:(-row[0],row[1:]))
        return choices[:self.max_swaps]

    def adapt(self):
        """Drain users, exchange bytes, then publish tables in place so graph addresses remain valid."""
        start=time.perf_counter()
        c=self.cache
        p=c.protected_payload
        torch.cuda.synchronize(c.device)
        counts=c.decode_freq.cpu().numpy()
        self.heat+=counts-self.last_counts
        self.last_counts=counts.copy()
        swaps=self.candidates(self.heat)
        self.heat*=self.decay
        if not swaps:
            self.rounds.append(dict(calls=self.calls,swaps=0,wall_ms=(time.perf_counter()-start)*1000))
            return
        initial=p.initial_map.numpy()
        staged=[]
        host_pointers=np.zeros((len(c.bank_schema),self.max_swaps),dtype=np.int64)
        gpu_slots=np.zeros(self.max_swaps,dtype=np.int32)
        commands=np.zeros((self.max_swaps,4),dtype=np.int32)
        for i,(gain,layer,cold,hot) in enumerate(swaps):
            slot=p.protected_slots[(layer,hot)]
            host_index=p.host_indices[layer][cold]
            for bank,role in enumerate(c.bank_schema):
                host_pointers[bank,i]=self.pointer_tables[layer][bank,cold]
            gpu_slots[i]=slot
            commands[i]=(layer,cold,hot,slot)
            staged.append((i,layer,cold,hot,slot,host_index))
        self.source_ptrs.copy_(torch.from_numpy(host_pointers))
        self.gpu_slots.copy_(torch.from_numpy(gpu_slots))
        self.commands.copy_(torch.from_numpy(commands))
        self.count.fill_(len(swaps))
        _exchange[(len(c.bank_schema),32)](c._copy_dst_ptrs,self.source_ptrs,
            c._copy_feat_bytes,self.gpu_slots,self.count,self.max_swaps,32,1024,num_warps=8)
        _publish_exchange[(1,)](self.commands,self.source_ptrs,c._copy_dst_ptrs,c._copy_feat_bytes,
            self.owner_table_ptrs,c.slot_for_id.view(-1),c.id_of_slot,c.usage,
            c._protected_initial_map.view(-1),c._protected_reverse_map,self.count,
            p.experts,p.protected_begin,self.max_swaps,len(c.bank_schema),
            triton.next_power_of_2(self.max_swaps),num_warps=4)
        for i,layer,cold,hot,slot,host_index in staged:
            for bank,role in enumerate(c.bank_schema):
                # WDDM needs the device-mapped host address, not the CPU virtual address.
                old_host=self.pointer_tables[layer][bank,cold]
                self.pointer_tables[layer][bank,cold]=c.bank_caches[role].data_ptr()+slot*c.bank_caches[role][0].numel()*c.bank_caches[role].element_size()
                self.pointer_tables[layer][bank,hot]=old_host
            initial[layer,hot]=-1
            initial[layer,cold]=slot
            del p.protected_slots[(layer,hot)]
            p.protected_slots[(layer,cold)]=slot
            p.host_indices[layer][cold]=-1
            p.host_indices[layer][hot]=host_index
        torch.cuda.synchronize(c.device)
        self.rounds.append(dict(calls=self.calls,swaps=len(swaps),wall_ms=(time.perf_counter()-start)*1000,
                               gain_sum=sum(s[0] for s in swaps)))

    def step(self):
        """Run maintenance inside generation latency, without moving it outside the timed request."""
        self.calls+=1
        if self.calls%self.every==0:self.adapt()

