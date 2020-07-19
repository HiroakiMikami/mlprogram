import numpy as np
import logging

from torch.utils import data
from torch.utils.data import IterableDataset

from typing import Optional, Any, Callable, Tuple, Dict

from mlprogram.utils import Reference as R, Token
from mlprogram.encoders import Samples
from mlprogram.languages.csg import AST, Interpreter, Reference
from mlprogram.languages.csg import Circle, Rectangle
from mlprogram.languages.csg import Translation, Rotation
from mlprogram.languages.csg import Union, Difference


logger = logging.getLogger(__name__)


class Dataset(IterableDataset):
    def __init__(
            self, canvas_size: int, canvas_resolution: int,
            depth: int, length_stride: int,
            degree_stride: int,
            reference: bool = False,
            seed: Optional[int] = None,
            transform: Optional[Callable[[Dict[str, Any]], Any]] = None):
        self.canvas_size = canvas_size
        self.canvas_resolution = canvas_resolution
        self.depth = depth
        s = self.canvas_size // 2
        self.reference = reference
        self.size_candidates = \
            list(range(1, s + 1))[::length_stride]
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
                return Circle(rng.choice(self.size_candidates))
            elif t == "Rectangle":
                return Rectangle(rng.choice(self.size_candidates),
                                 rng.choice(self.size_candidates))
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

    def to_reference(self, code: AST, n_ref: int = 0) \
            -> Tuple[Dict[R, AST], int]:
        if isinstance(code, Circle):
            return {R(str(n_ref)): code}, n_ref
        elif isinstance(code, Rectangle):
            return {R(str(n_ref)): code}, n_ref
        elif isinstance(code, Translation):
            retval, n_ref = self.to_reference(code.child, n_ref)
            retval[R(str(n_ref + 1))] = Translation(code.x, code.y,
                                                    Reference(R(str(n_ref))))
            return retval, n_ref + 1
        elif isinstance(code, Rotation):
            retval, n_ref = self.to_reference(code.child, n_ref)
            retval[R(str(n_ref + 1))] = Rotation(code.theta_degree,
                                                 Reference(R(str(n_ref))))
            return retval, n_ref + 1
        elif isinstance(code, Union):
            retval0, n_ref0 = self.to_reference(code.a, n_ref)
            retval1, n_ref1 = self.to_reference(code.b, n_ref0 + 1)
            retval1[R(str(n_ref1 + 1))] = Union(Reference(R(str(n_ref0))),
                                                Reference(R(str(n_ref1))))
            for r, code in retval0.items():
                retval1[r] = code
            return retval1, n_ref1 + 1
        elif isinstance(code, Difference):
            retval0, n_ref0 = self.to_reference(code.a, n_ref)
            retval1, n_ref1 = self.to_reference(code.b, n_ref0 + 1)
            retval1[R(str(n_ref1 + 1))] = Difference(Reference(R(str(n_ref0))),
                                                     Reference(R(str(n_ref1))))
            for r, code in retval0.items():
                retval1[r] = code
            return retval1, n_ref1 + 1
        logger.warning(f"Invalid node type {code.type_name()}")
        return {}, -1

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
                if self.parent.reference:
                    refs, output = self.parent.to_reference(ast)
                    references = [Token(None, ref) for ref in refs.keys()]
                    references.sort(key=lambda x: str(x.value.name))
                    retval = {
                        "ground_truth": refs,
                        "references": references,
                        "output_reference": R(str(output)),
                        "test_case": canvas
                    }
                else:
                    retval = {
                        "ground_truth": ast,
                        "test_case": canvas
                    }
                if self.parent.transform is not None:
                    retval = self.parent.transform(retval)
                return retval

        return InternalIterator(self)

    @property
    def samples(self) -> Samples:
        # rules: List[Rule]
        # node_types: List[NodeType]
        # tokens: List[str]  # TODO V
        pass
