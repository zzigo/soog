"""Server load figures for the SESSION panel, read from /proc with no extra dependency.

CPU utilisation is the share of non-idle jiffies between two samples of /proc/stat. The
sampler keeps the previous sample, so a call returns the load since the call before it
(or since a short blocking sample on the very first call).
"""

import os
import threading
import time
from typing import Dict, Optional, Tuple

PROC_STAT = '/proc/stat'
PROC_MEMINFO = '/proc/meminfo'
PROC_SELF_STATUS = '/proc/self/status'

_MIN_SAMPLE_INTERVAL_SEC = 0.25


def parse_proc_stat(text: str) -> Optional[Tuple[int, int]]:
    """Return (busy, total) jiffies from the aggregate 'cpu' line of /proc/stat."""
    for line in (text or '').splitlines():
        if line.startswith('cpu '):
            parts = [int(value) for value in line.split()[1:] if value.isdigit()]
            if len(parts) < 4:
                return None
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)  # idle + iowait
            total = sum(parts[:8])  # user nice system idle iowait irq softirq steal
            return total - idle, total
    return None


def parse_meminfo(text: str) -> Dict[str, int]:
    """Return /proc/meminfo as {key: kB}."""
    values: Dict[str, int] = {}
    for line in (text or '').splitlines():
        key, _, rest = line.partition(':')
        digits = rest.strip().split(' ')[0]
        if digits.isdigit():
            values[key.strip()] = int(digits)
    return values


def _read(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return handle.read()
    except OSError:
        return ''


class CpuSampler:
    def __init__(self, reader=None):
        self._reader = reader or (lambda: _read(PROC_STAT))
        self._lock = threading.Lock()
        self._last: Optional[Tuple[int, int]] = None
        self._last_at = 0.0
        self._last_pct: Optional[float] = None

    def sample(self) -> Optional[float]:
        """CPU utilisation in percent since the previous sample, or None when unreadable."""
        with self._lock:
            now = time.monotonic()
            current = parse_proc_stat(self._reader())
            if current is None:
                return None
            if self._last is None:
                self._last, self._last_at = current, now
                time.sleep(0.1)
                current = parse_proc_stat(self._reader())
                if current is None:
                    return None
                now = time.monotonic()
            elif now - self._last_at < _MIN_SAMPLE_INTERVAL_SEC and self._last_pct is not None:
                return self._last_pct
            busy_delta = current[0] - self._last[0]
            total_delta = current[1] - self._last[1]
            self._last, self._last_at = current, now
            if total_delta <= 0:
                return self._last_pct
            self._last_pct = round(max(0.0, min(100.0, busy_delta / total_delta * 100.0)), 1)
            return self._last_pct


CPU_SAMPLER = CpuSampler()


def _process_rss_mb() -> Optional[float]:
    for line in _read(PROC_SELF_STATUS).splitlines():
        if line.startswith('VmRSS:'):
            digits = line.split()[1]
            return round(int(digits) / 1024.0, 1) if digits.isdigit() else None
    return None


def read_server_metrics(sampler: Optional[CpuSampler] = None) -> Dict[str, object]:
    """Snapshot for operators: CPU, load average, memory, swap and this process."""
    sampler = sampler or CPU_SAMPLER
    metrics: Dict[str, object] = {'at': time.time(), 'cpus': os.cpu_count() or 1}
    cpu_pct = sampler.sample()
    if cpu_pct is not None:
        metrics['cpu_pct'] = cpu_pct
    try:
        load1, load5, load15 = os.getloadavg()
        metrics.update(load1=round(load1, 2), load5=round(load5, 2), load15=round(load15, 2))
    except (OSError, AttributeError):
        pass
    mem = parse_meminfo(_read(PROC_MEMINFO))
    if mem.get('MemTotal'):
        total_mb = mem['MemTotal'] / 1024.0
        available_mb = mem.get('MemAvailable', mem.get('MemFree', 0)) / 1024.0
        metrics.update(
            mem_total_mb=round(total_mb),
            mem_used_mb=round(total_mb - available_mb),
        )
    if mem.get('SwapTotal'):
        metrics.update(
            swap_total_mb=round(mem['SwapTotal'] / 1024.0),
            swap_used_mb=round((mem['SwapTotal'] - mem.get('SwapFree', 0)) / 1024.0),
        )
    rss = _process_rss_mb()
    if rss is not None:
        metrics['process_rss_mb'] = rss
    return metrics
