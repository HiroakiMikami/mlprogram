from typing import List, Optional, Tuple

import numpy as np

from mlprogram.languages.linediff import AST, Delta, Diff, Replace


class TypoMutator:
    def __init__(self, max_mutation: int, seed: Optional[int] = None):
        self.max_mutation = max_mutation
        self.rng = np.random.RandomState(seed or 0)

    def mutate(self, code: str) -> Tuple[str, List[Tuple[str, str]], AST]:
        n_mutation = self.rng.randint(1, self.max_mutation + 1)

        # Based on https://bitbucket.org/iiscseal/deepfix/src/master/data_processing/typo_mutator.py  # noqa
        # Duplicate '(', ')', '{', '}', ',', ';'
        # Delete '(', ')', '{', '}', ',', ';'
        # Replace ';' -> ',', ',' -> ';', ';' -> '.', ');' -> ';)'
        def get_operations(ch):
            retval = []
            if ch in set(["(", ")", "{", "}", ",", ";"]):
                retval.append(("Duplicate", ch))
                retval.append(("Delete", ch))
            if ch == ";":
                retval.append(("Replace", ","))
                retval.append(("Replace", "."))
            if ch == ",":
                retval.append(("Replace", ";"))
            if ch == ");":
                retval.append(("Replace", ";)"))
            return retval

        def get_candidate(text: str):
            candidates = []
            for offset, (c0, c1) in enumerate(zip(text, text[1:] + " ")):
                if f"{c0}{c1}" == ");":
                    candidates.append((offset, f"{c0}{c1}"))
                if c0 in set(["(", ")", "{", "}", ",", ";"]):
                    candidates.append((offset, c0))
            return candidates

        if len(code) == 0:
            return (code, [(code, code)], Diff([]))
        lines = list(code.split("\n"))
        targets = self.rng.choice(len(lines), size=min(len(lines), n_mutation),
                                  replace=False)
        targets.sort()
        deltas: List[Delta] = []
        for line in targets:
            candidates = get_candidate(lines[line])
            if len(candidates) == 0:
                continue
            i = self.rng.choice(len(candidates))
            offset, value = candidates[i]

            operations = get_operations(value)
            if len(operations) == 0:
                continue
            i = self.rng.choice(len(operations))
            op, value = operations[i]

            if op == "Duplicate":
                new_line = \
                    f"{lines[line][:offset]}{value}{lines[line][offset:]}"
            elif op == "Delete":
                new_line = \
                    f"{lines[line][:offset]}{lines[line][offset + 1:]}"
            elif op == "Replace":
                new_line = \
                    f"{lines[line][:offset]}{value}{lines[line][offset + 1:]}"
            deltas.append(Replace(line, lines[line]))
            lines[line] = new_line

        text = "\n".join(lines)
        # TODO remove test case from the outputs
        return (text, [(text, code)], Diff(deltas))
