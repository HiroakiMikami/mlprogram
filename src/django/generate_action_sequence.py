"""
Copyright (c) 2018 Hiroaki Mikami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code was modified from ase15-django/dataset/src/parse.py
(https://github.com/odashi/ase15-django-dataset/blob/master/src/parse.py),
which is subject to the MIT license.
Here is the original copyright notice for ase15-django/dataset/src/parse.py:

Code copyright 2015 Yusuke Oda and AHCLab
Code released under the MIT License

Copyright (c) 2015 Yusuke Oda and AHCLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import ast
import sys
import re
import argparse
import json
import os
from ..python.grammar import to_sequence, to_ast

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')


def parse(code):
    l = code.strip()
    if not l:
        return ""

    if p_elif.match(l): l = 'if True: pass\n' + l
    if p_else.match(l): l = 'if True: pass\n' + l

    if p_try.match(l): l = l + 'pass\nexcept: pass'
    elif p_except.match(l): l = 'try: pass\n' + l
    elif p_finally.match(l): l = 'try: pass\n' + l

    if p_decorator.match(l): l = l + '\ndef dummy(): pass'
    if l[-1] == ':': l = l + 'pass'

    return ast.parse(l).body[0]


parser = argparse.ArgumentParser()
parser.add_argument('--ids', type=str, nargs='+')
parser.add_argument(
    '--directory',
    type=str,
    default=os.path.join("dataset", "django", "train"))
parser.add_argument('--validate', action='store_true')

args = parser.parse_args()

import transpyle
unparser = transpyle.python.unparser.NativePythonUnparser()

for id in args.ids:
    # Read the code file
    with open(os.path.join(args.directory, "{}.code".format(id))) as f:
        code = "\n".join(f.readlines())

    # Parse code snippet
    node = parse(code)
    # Generate action sequence
    sequence = to_sequence(node)

    # Validate action sequence
    if args.validate:
        node2 = to_ast(sequence)
        assert (unparser.unparse(node) == unparser.unparse(node2))

    with open(
            os.path.join(args.directory, "{}.reference_seq.json".format(id)),
            mode='w') as f:
        json.dump(sequence, f)
