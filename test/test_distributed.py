import tempfile
import multiprocessing as mp

from mlprogram import distributed

context = mp.get_context("spawn")


class Values:
    def __init__(self, xs):
        self.xs = xs


def _run(init_dir, rank):
    distributed.initialize(init_dir, rank, 2)
    return [x.xs
            for x in distributed.all_gather(Values([i for i in range(2 * rank + 1)]))]


def test_all_gather():
    with tempfile.TemporaryDirectory() as init_dir:
        with context.Pool(2) as pool:
            procs = []
            for i in range(2):
                p = pool.apply_async(
                    _run,
                    args=(init_dir, i),
                )
                procs.append(p)
            out = [p.get() for p in procs]

    assert out[0] == [[0], [0, 1, 2]]
    assert out[0] == out[1]
