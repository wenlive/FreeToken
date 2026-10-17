"""Compare nearby ABBA cycle rates; keep exploitation outside the paired estimate."""
from collections import deque
import math


def _rate(samples):
    return sum(tokens for tokens, _ in samples) / sum(seconds for _, seconds in samples)


class PairedPolicy:
    def __init__(self, arms=('full-3', 'cjk-2')):
        self.arms = arms
        self.preferred = arms[0]
        self.steps = 0
        self.chosen = {arm: 0 for arm in arms}
        self.samples = {arm: deque(maxlen=32) for arm in arms}
        self.ratios = deque(maxlen=8)
        self.block = []
        self.block_data = {arm: [] for arm in arms}
        self.blocks = 0
        self.last_probe = 0
        self.record_pair = False

    def _start_block(self):
        """Alternate ABBA and BAAB to balance local context and cache drift."""
        a, b = self.arms
        self.block = [a, b, b, a] if self.blocks % 2 == 0 else [b, a, a, b]
        self.block_data = {arm: [] for arm in self.arms}

    def observe(self, arm, accepted, seconds):
        """Only a complete local exploration block can change the preferred arm."""
        if seconds <= 0:
            return
        self.steps += 1
        if self.steps <= 4:
            return
        self.samples[arm].append((accepted + 1, seconds))
        if not self.record_pair:
            return
        self.block_data[arm].append((accepted + 1, seconds))
        if self.block or any(len(self.block_data[a]) != 2 for a in self.arms):
            return
        full, cjk = self.arms
        self.ratios.append(_rate(self.block_data[cjk]) / _rate(self.block_data[full]))
        self.blocks += 1
        ratio = math.exp(sum(math.log(value) for value in self.ratios) / len(self.ratios))
        if len(self.ratios) >= 4:
            if ratio > 1.03:
                self.preferred = cjk
            elif ratio < 1 / 1.03:
                self.preferred = full

    def choose(self):
        """Use four calibration blocks, then one balanced block every 48 windows."""
        self.record_pair = False
        if self.steps < 4:
            arm = self.arms[0]
        else:
            if not self.block and (self.blocks < 4 or self.steps - self.last_probe >= 48):
                self._start_block()
                self.last_probe = self.steps
            if self.block:
                arm = self.block.pop(0)
                self.record_pair = True
            else:
                arm = self.preferred
        self.chosen[arm] += 1
        return arm

    def snapshot(self):
        """Expose bounded diagnostics without feeding rolling rates into decisions."""
        return dict(
            steps=self.steps,
            preferred=self.preferred,
            chosen=self.chosen.copy(),
            paired_blocks=self.blocks,
            paired_rate_ratios=list(self.ratios),
            comparison='ABBA/BAAB full-cycle, exploitation excluded',
            rolling_cycle_tps={arm: _rate(samples) if samples else 0 for arm, samples in self.samples.items()},
        )
