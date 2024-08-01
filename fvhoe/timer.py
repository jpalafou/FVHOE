from dataclasses import dataclass
import numpy as np
import time
from typing import Iterable


@dataclass
class Timer:
    """
    Timer which allows you to start and stop different categories simultaneously
    args:
        cats (Iterable[str]) : series of named timer categories as strings
        precision (int) : number of decimal places to round time in to_dict method
    """

    cats: Iterable[str] = ()
    precision: int = 2

    def __post_init__(self):
        self._is_timed = {cat: False for cat in self.cats}
        self._start_times = {cat: 0.0 for cat in self.cats}
        self.cum_times = {cat: 0.0 for cat in self.cats}

    def check_cat_existance(self, cat: str):
        """
        Check if timer category exists
        args:
            cat (str) : category
        """
        if cat not in self.cats:
            raise ValueError(f"Category '{cat}' not found in timer categories.")

    def start(self, cat: str):
        """
        Start timer of a category
        args:
            cat (str) : category
        """
        self.check_cat_existance(cat)
        if self._is_timed[cat]:
            raise RuntimeError(
                f"Cannot start '{cat}' timer which is already in progress."
            )
        self._is_timed[cat] = True
        self._start_times[cat] = time.time()

    def stop(self, cat: str):
        """
        Stop timer of a category
        args:
            cat (str) : category
        """
        self.check_cat_existance(cat)
        if not self._is_timed[cat]:
            raise RuntimeError(f"Cannot stop '{cat}' timer which is not in progress.")
        self._is_timed[cat] = False
        self.cum_times[cat] += time.time() - self._start_times[cat]

    def to_dict(self) -> dict:
        out = {cat: np.round(t, self.precision) for cat, t in self.cum_times.items()}
        return out
