import pandas as pd
import time
from typing import Dict


class TimerImplementation:
    def __init__(self, memory_length: int = 100):
        self.memory_length = memory_length

        self._start_time = None
        self.average_time = 0.
        self.calls = 0

    def start(self):
        self.calls += 1
        self._start_time = time.time()

    def stop(self):
        time_diff = time.time() - self._start_time
        memory = self.calls if self.memory_length > self.calls else self.memory_length
        self.average_time -= (self.average_time - time_diff) / memory

    def reset(self):
        self._start_time = None
        self.average_time = 0.
        self.calls = 0


class Timer:
    _timers: Dict[str, TimerImplementation] = None

    @classmethod
    def start(cls, name: str):
        if cls._timers is None:
            cls._timers = {}

        if name not in cls._timers:
            cls._timers[name] = TimerImplementation()

        cls._timers[name].start()

    @classmethod
    def stop(cls, name: str):
        assert name in cls._timers

        cls._timers[name].stop()

    @classmethod
    def __str__(cls):
        if cls._timers is None:
            return ""
        out_tables = []
        for name, timer in cls._timers.items():
            out_tables.append({"name": name,
                               "iterations": timer.calls,
                               "average": timer.average_time,
                               "total_time": timer.average_time * timer.calls
                               })
        return str(pd.DataFrame(out_tables))

    @classmethod
    def get_timer(cls, name: str) -> TimerImplementation:
        if name not in cls._timers:
            return None
        return cls._timers[name]