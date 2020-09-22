import numpy as np

from torch.utils import data
from torch.utils.data import IterableDataset
from typing import Optional, Any, Tuple, Dict, List

from mlprogram.interpreters import Reference as R
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.languages.csg import AST, Reference
from mlprogram.languages.csg import Circle, Rectangle
from mlprogram.languages.csg import Translation, Rotation
from mlprogram.languages.csg import Union, Difference
from mlprogram import logging


logger = logging.Logger(__name__)


class Dataset(IterableDataset):
    def __init__(
            self, canvas_size: int,
            min_object: int, max_object: int, length_stride: int,
            degree_stride: int,
            reference: bool = False,
            seed: Optional[int] = None):
        self.canvas_size = canvas_size
        self.min_object = min_object
        self.max_object = max_object
        s = self.canvas_size // 2
        self.reference = reference
        self.size_candidates = \
            list(range(1, s + 1))[::length_stride]
        self.length_candidates = list(range(-s, s + 1))[::length_stride]
        self.degree_candidates = list(range(-180, 180))[::degree_stride]
        self.leaf_candidates = ["Circle", "Rectangle"]
        self.branch_candidates = ["Union", "Difference"]
        self.node_candidates = ["Translation", "Rotation"]
        self.seed = \
            seed if seed is not None else np.random.randint(0, 2 ** 32 - 1)

    def sample_ast(self, rng: np.random.RandomState, n_object: int) -> AST:
        objects: Dict[int, AST] = {}
        for i, t in enumerate(rng.choice(self.leaf_candidates, n_object)):
            if t == "Circle":
                objects[i] = Circle(rng.choice(self.size_candidates))
            elif t == "Rectangle":
                objects[i] = Rectangle(rng.choice(self.size_candidates),
                                       rng.choice(self.size_candidates))
            else:
                raise Exception(f"Invalid type: {t}")
        ops = {}
        for i, t in enumerate(
                rng.choice(self.branch_candidates, n_object - 1)):
            ops[i] = t
        n_node = rng.randint(0, len(ops) + 2)
        n = len(ops)
        for i, t in enumerate(rng.choice(self.node_candidates, n_node)):
            ops[i + n] = t

        while len(objects) > 1 and len(ops) != 0:
            op_key = rng.choice(list(ops.keys()))
            op = ops.pop(op_key)
            obj0_key = rng.choice(list(objects.keys()))
            obj0 = objects.pop(obj0_key)
            if op == "Translation":
                objects[obj0_key] = Translation(
                    rng.choice(self.length_candidates),
                    rng.choice(self.length_candidates),
                    obj0)
            elif op == "Rotation":
                objects[obj0_key] = Rotation(
                    rng.choice(self.degree_candidates), obj0)
            else:
                obj1_key = rng.choice(list(objects.keys()))
                obj1 = objects.pop(obj1_key)
                if op == "Union":
                    objects[obj0_key] = Union(obj0, obj1)
                elif op == "Difference":
                    objects[obj0_key] = Difference(obj0, obj1)
                else:
                    raise Exception(f"Invalid type: {t}")
        return list(objects.values())[0]

    def to_reference(self, code: AST, n_ref: int = 0) \
            -> Tuple[List[Statement[AST]], int]:
        if isinstance(code, Circle):
            return [Statement(R(str(n_ref)), code)], n_ref
        elif isinstance(code, Rectangle):
            return [Statement(R(str(n_ref)), code)], n_ref
        elif isinstance(code, Translation):
            retval, n_ref = self.to_reference(code.child, n_ref)
            retval.append(Statement(
                R(str(n_ref + 1)),
                Translation(code.x, code.y,
                            Reference(R(str(n_ref))))
            ))
            return retval, n_ref + 1
        elif isinstance(code, Rotation):
            retval, n_ref = self.to_reference(code.child, n_ref)
            retval.append(Statement(
                R(str(n_ref + 1)),
                Rotation(code.theta_degree,
                         Reference(R(str(n_ref))))
            ))
            return retval, n_ref + 1
        elif isinstance(code, Union):
            retval0, n_ref0 = self.to_reference(code.a, n_ref)
            retval1, n_ref1 = self.to_reference(code.b, n_ref0 + 1)
            retval0.extend(retval1)
            retval0.append(Statement(
                R(str(n_ref1 + 1)),
                Union(Reference(R(str(n_ref0))),
                      Reference(R(str(n_ref1))))
            ))
            return retval0, n_ref1 + 1
        elif isinstance(code, Difference):
            retval0, n_ref0 = self.to_reference(code.a, n_ref)
            retval1, n_ref1 = self.to_reference(code.b, n_ref0 + 1)
            retval0.extend(retval1)
            retval0.append(Statement(
                R(str(n_ref1 + 1)),
                Difference(Reference(R(str(n_ref0))),
                           Reference(R(str(n_ref1))))
            ))
            return retval0, n_ref1 + 1
        logger.warning(f"Invalid node type {code.type_name()}")
        # TODO throw exception
        return [], -1

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            seed = self.seed
        else:
            seed = (self.seed * (worker_info.id + 1)) % (2 ** 32 - 1)
        rng = np.random.RandomState(seed)

        class InternalIterator:
            def __init__(self, parent: Dataset):
                self.parent = parent
                self.obj_prob = \
                    [float(n) for n in range(self.parent.min_object,
                                             self.parent.max_object + 1)]
                self.obj_prob = \
                    [p / sum(self.obj_prob) - 1e-5 for p in self.obj_prob]

            def __next__(self) -> Any:
                n_object = rng.multinomial(1, self.obj_prob).nonzero()[0] + 1
                ast = self.parent.sample_ast(rng, n_object)
                if self.parent.reference:
                    refs, output = self.parent.to_reference(ast)
                    retval: Dict[str, Any] = {
                        "ground_truth": [SequentialProgram(refs)]
                    }
                else:
                    retval = {
                        "ground_truth": [ast]
                    }
                return retval

        return InternalIterator(self)
