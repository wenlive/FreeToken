"""Compare complete-cycle policies in nearby ABBA blocks, excluding stale exploitation samples."""
from collections import deque
import math


class PairedPolicy:
    def __init__(self,arms=('full-3','cjk-2')):
        self.arms=arms
        self.preferred=arms[0]
        self.steps=0
        self.chosen={arm:0 for arm in arms}
        self.samples={arm:deque(maxlen=32) for arm in arms}
        self.ratios=deque(maxlen=8)
        self.block=[]
        self.block_data={arm:[] for arm in arms}
        self.blocks=0
        self.last_probe=0
        self.record_pair=False

    def _start_block(self):
        """Alternate ABBA and BAAB to balance local context/cache drift."""
        a,b=self.arms
        self.block=[a,b,b,a] if self.blocks%2==0 else [b,a,a,b]
        self.block_data={arm:[] for arm in self.arms}

    def observe(self,arm,accepted,seconds):
        """Only contemporaneous exploration blocks can alter the comparison."""
        if seconds<=0:return
        self.steps+=1
        if self.steps<=4:return
        self.samples[arm].append((accepted+1,seconds))
        if not self.record_pair:return
        self.block_data[arm].append((accepted+1,seconds))
        if not self.block and all(len(self.block_data[a])==2 for a in self.arms):
            rates={a:sum(x[0] for x in self.block_data[a])/sum(x[1] for x in self.block_data[a]) for a in self.arms}
            self.ratios.append(rates[self.arms[1]]/rates[self.arms[0]])
            self.blocks+=1
            ratio=math.exp(sum(math.log(x) for x in self.ratios)/len(self.ratios))
            if len(self.ratios)>=4:
                if ratio>1.03:self.preferred=self.arms[1]
                elif ratio<1/1.03:self.preferred=self.arms[0]

    def choose(self):
        """Use four calibration blocks, then one balanced block per48 windows."""
        self.record_pair=False
        if self.steps<4:arm=self.arms[0]
        else:
            if not self.block and (self.blocks<4 or self.steps-self.last_probe>=48):
                self._start_block();self.last_probe=self.steps
            if self.block:
                arm=self.block.pop(0);self.record_pair=True
            else:arm=self.preferred
        self.chosen[arm]+=1
        return arm

    def snapshot(self):
        return dict(steps=self.steps,preferred=self.preferred,chosen=self.chosen.copy(),paired_blocks=self.blocks,
            paired_rate_ratios=list(self.ratios),comparison='ABBA/BAAB full-cycle, exploitation excluded',
            rolling_cycle_tps={a:sum(x[0] for x in v)/sum(x[1] for x in v) if v else 0 for a,v in self.samples.items()})
