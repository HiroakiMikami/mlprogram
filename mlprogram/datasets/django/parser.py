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

import re
from typing import Callable, List, Optional

from mlprogram.languages.ast import AST
from mlprogram.languages.python.parser import Parser as BaseParser

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')


class Parser(BaseParser):
    def __init__(self, split_value: Callable[[str], List[str]]):
        super().__init__(split_value)

    def parse(self, code: str) -> Optional[AST]:
        """
        Return the AST of the code

        Parameters
        ----------
        code: str
            The code to be parsed

        Returns
        -------
        AST
            The AST of the code
        """
        try:
            code = code.strip()
            if not code:
                return None

            if p_elif.match(code):
                code = 'if True: pass\n' + code
            if p_else.match(code):
                code = 'if True: pass\n' + code

            if p_try.match(code):
                code = code + 'pass\nexcept: pass'
            elif p_except.match(code):
                code = 'try: pass\n' + code
            elif p_finally.match(code):
                code = 'try: pass\n' + code

            if p_decorator.match(code):
                code = code + '\ndef dummy(): pass'
            if code[-1] == ':':
                code = code + 'pass'

            return super().parse(code)
        except:  # noqa
            return None
