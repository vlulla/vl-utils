## From the excellent blog post: https://gdmarmerola.github.io/big-data-ml-training/
##
import numpy as np, matplotlib, matplotlib.pyplot as plt


import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from time import sleep

SAMPLING_TIME = 0.001

class MemoryMonitor:
    def __init__(self, close=True):
        tracemalloc.start()
        self.keep_measuring = True
        self.close = close
    def measure_usage(self):
        usage_list = []
        while self.keep_measuring:
            current, peak = tracemalloc.get_traced_memory()
            usage_list.append(current/1e6)
            sleep(SAMPLING_TIME)
        if self.close:
            tracemalloc.stop()
        return usage_list

def plot_memory_use(history, fn_name, open_figure=True, offset=0, **kwargs):
    times = (offset + np.arange(len(history))) * SAMPLING_TIME
    if open_figure:
        plt.figure(figsize=(10,3), dpi=150)
    plt.plot(times, history, "k--", linewidth=1)
    plt.fill_between(times, history, alpha=0.2, **kwargs)
    plt.ylabel("Memory usage [MB]")
    plt.xlabel("Time [sec]")
    plt.title(f"{fn_name} memory usage over time")
    plt.text(np.quantile(times,0.8), 0.8*max(history), f"Peak mem: {max(history):.2f}")
    plt.grid(axis="y", linestyle=(0, (1, 3)))
    # plt.legend()
    ## plt.show() if you are in [i]python

def track_memory_usage(plot=True, close=True, return_history=False):
    """
    In [1]: import memusage, time
       ...: @memusage.track_memory_usage(plot=True, close=False, return_history=True)
       ...: def main():
       ...:     for i in range(8):
       ...:         x = np.random.normal(loc=100, scale=2, size=10**i)
       ...:         time.sleep(0.25)
       ...:     for i in range(7,-1,-1): x = np.random.normal(loc=100, scale=2, size=10**i)
       ...:     time.sleep(1)
       ...:     for i in range(8): x = np.random.normal(loc=100, scale=2, size=10**i)

    In [2]: main()
    In [4]: import matplotlib.pyplot as plt
    In [3]: plt.show()
    """
    def meta_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor() as executor:
                monitor = MemoryMonitor(close=close)
                mem_thread = executor.submit(monitor.measure_usage)
                try:
                    fn_thread = executor.submit(fn, *args, **kwargs)
                    fn_result = fn_thread.result()
                finally:
                    monitor.keep_measuring = False
                    history = mem_thread.result()
                print(f"Current memory usage: {history[-1]:.2f}")
                print(f"Peak memory usage: {max(history):.2f}")
                if plot:
                    plot_memory_use(history, fn.__name__)
            if return_history:
                return fn_result, history
            else:
                return fn_result
        return wrapper
    return meta_wrapper
