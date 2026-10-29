from multiprocessing import Queue, Process

import numpy as np


rng = np.random.default_rng()


def test_running_stats():
    r_vals = rng.uniform(size=100)

    run_stats = RunningStats(stdev=True)

    true_mean = r_vals.mean()
    true_stdev = r_vals.std()

    for v in r_vals:
        run_mean, run_stdev = run_stats(v)
        print(true_mean, run_mean, true_stdev, run_stdev)

    r_vals = rng.uniform(size=100)

    run_stats = RunningStats()

    true_mean = r_vals.mean()

    for v in r_vals:
        run_mean = run_stats(v)
        print(true_mean, run_mean)


# Welford's algorithm
class RunningStats:
    def __init__(self, stdev=False):
        self.stdev = stdev
        self.running_mean = 0
        self.num = 0
        self.prev_mean_delta = 0
        self.new_mean_delta = 0
        self.squared_deviations = 0
        self.running_stdev = 0

    def __call__(self, x):
        self.num += 1
        self.prev_mean_delta = x - self.running_mean
        self.running_mean += self.prev_mean_delta / self.num

        if self.stdev:
            self.new_mean_delta = x - self.running_mean

            self.squared_deviations += (
                self.prev_mean_delta * self.new_mean_delta
            )

            self.running_stdev = np.sqrt(self.squared_deviations / self.num)
            return self.running_mean, self.running_stdev
        else:
            return self.running_mean


class Worker:
    def __init__(self, function, *init_args, out_q=None):
        self.in_q = Queue()
        self.put = self.in_q.put

        if out_q is None:
            self.out_q = Queue()
            self.get = self.out_q.get
        else:
            # Shared output queue
            self.out_q = out_q

        self.process = Process(
            target=function,
            args=(self.in_q, self.out_q, *init_args),
            daemon=True
        )

        self.process.start()


if __name__ == '__main__':
    def fn(in_q, out_q, hello, world):
        print(hello, world, 'from subproc')
        out_q.put('whaddup')

    worker = Worker(fn, 'hello', 'world')
    print(worker.out_q.get())
