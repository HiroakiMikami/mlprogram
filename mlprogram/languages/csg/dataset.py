import numpy as np

from torch.utils import data
from torch.utils.data import IterableDataset

from typing import Optional, Any, Callable, Tuple

from mlprogram.languages.csg import AST, Interpreter
from mlprogram.languages.csg import Circle, Rectangle
from mlprogram.languages.csg import Translation, Rotation
from mlprogram.languages.csg import Union, Difference


class Dataset(IterableDataset):
    def __init__(
            self, canvas_size: int, canvas_resolution: int,
            depth: int, length_stride: int,
            degree_stride: int, seed: Optional[int] = None,
            transform: Optional[Callable[[Tuple[AST, np.array]], Any]] = None):
        self.canvas_size = canvas_size
        self.canvas_resolution = canvas_resolution
        self.depth = depth
        s = self.canvas_size // 2
        self.length_candidates = list(range(-s, s + 1))[::length_stride]
        self.degree_candidates = list(range(-180, 180))[::degree_stride]
        self.leaf_candidates = ["Circle", "Rectangle"]
        self.node_candidates = ["Translation", "Rotation",
                                "Union", "Difference"]
        self.seed = seed if seed is not None else 0
        self.transform = transform

    def sample_ast(self, rng: np.random.RandomState, depth: int) -> AST:
        if depth == 1:
            t = rng.choice(self.leaf_candidates)
            if t == "Circle":
                return Circle(rng.choice(self.length_candidates))
            elif t == "Rectangle":
                return Rectangle(rng.choice(self.length_candidates),
                                 rng.choice(self.length_candidates))
            else:
                raise Exception(f"Invalid type: {t}")
        else:
            t = rng.choice(self.node_candidates)
            if t == "Translation":
                return Translation(rng.choice(self.length_candidates),
                                   rng.choice(self.length_candidates),
                                   self.sample_ast(rng, depth - 1))
            elif t == "Rotation":
                return Rotation(rng.choice(self.degree_candidates),
                                self.sample_ast(rng, depth - 1))
            elif t == "Union":
                if rng.choice([True, False]):
                    return Union(self.sample_ast(rng, depth - 1),
                                 self.sample_ast(rng, rng.randint(1, depth)))
                else:
                    return Union(self.sample_ast(rng, rng.randint(1, depth)),
                                 self.sample_ast(rng, depth - 1))
            elif t == "Difference":
                if rng.choice([True, False]):
                    return Difference(self.sample_ast(rng, depth - 1),
                                      self.sample_ast(rng,
                                                      rng.randint(1, depth)))
                else:
                    return Difference(self.sample_ast(rng,
                                                      rng.randint(1, depth)),
                                      self.sample_ast(rng, depth - 1))
            else:
                raise Exception(f"Invalid type: {t}")

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            seed = self.seed
        else:
            seed = self.seed + worker_info.id
        rng = np.random.RandomState(seed)

        class InternalIterator:
            def __init__(self, parent: Dataset):
                self.parent = parent
                self.interpreter = Interpreter(
                    self.parent.canvas_size,
                    self.parent.canvas_size,
                    self.parent.canvas_resolution)

            def __next__(self) -> Any:
                depth = np.random.randint(1, self.parent.depth + 1)
                ast = self.parent.sample_ast(rng, depth)
                canvas = self.interpreter.eval(ast)
                retval = (ast, canvas)
                if self.parent.transform is not None:
                    retval = self.parent.transform(retval)
                return retval

        return InternalIterator(self)
