"""
Copyright (c) 2020 Hiroaki Mikami

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

This code was modified from deepfix/util/c_tokenizer.py
(https://bitbucket.org/iiscseal/deepfix/src/master/util/c_tokenizer.py),
which is subject to the Apache License, Version2.0.
Here is the original copyright notice for deepfix/util/c_tokenizer.py:

Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re
from typing import List, Optional, Tuple

from mlprogram import logging
from mlprogram.languages import Lexer as BaseLexer
from mlprogram.languages import Token

logger = logging.Logger(__name__)


class _Mapping:
    def __init__(self, prefix):
        self.mapping = {}
        self.prefix = prefix

    def __call__(self, value):
        if value not in self.mapping:
            self.mapping[value] = len(self.mapping)
        return f"___{self.prefix}@{self.mapping[value]}___"


class Lexer(BaseLexer[str, str]):
    _keywords = ['auto', 'break', 'case', 'const', 'continue', 'default',
                 'do', 'else', 'enum', 'extern', 'for', 'goto', 'if',
                 'register', 'return', 'signed', 'sizeof', 'static', 'switch',
                 'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL',
                 'null', 'struct', 'union']
    _includes = ['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h',
                 'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h']
    _calls = ['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen',
              'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free']
    _types = ['char', 'double', 'float', 'int', 'long', 'short', 'unsigned']

    def __init__(self, delimiter: str = " "):
        super().__init__()
        self.delimiter = delimiter

    def tokenize_with_offset(self, code: str) \
            -> Optional[List[Tuple[int, Token[str, str]]]]:
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment',
             r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"?'),
            ('char', r"'(?:\\?[^'\n]|\\')'"),
            ('char_continue', r"'[^']*"),
            ('number', r'[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'),
            ('include', r'(?<=\#include) *<([_A-Za-z]\w*(?:\.h))?>'),
            ('op',
             r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=]=|[-<>~!%^&*\/+=?|.,:;#]'),  # noqa
            ('name', r'[_A-Za-z]\w*'),
            ('whitespace', r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH', r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' %
                             pair for pair in token_specification)
        tokens = []
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            if kind is None:
                continue
            value = mo.group(kind)
            if kind in set(['NEWLINE', 'SKIP', 'whitespace']):
                pass
            elif kind == 'MISMATCH':
                return None
            else:
                if kind == 'name' and value in keywords:
                    kind = value
                offset = mo.start()
                tokens.append((offset, Token(kind, value, value)))

        name_to_idx = _Mapping("name")
        number_to_idx = _Mapping("number")
        str_to_idx = _Mapping("string")
        chr_to_idx = _Mapping("char")

        if tokens is None:
            return None
        retval = []
        for offset, token in tokens:
            if token.kind == "name":
                retval.append((offset, Token(token.kind, name_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "number":
                retval.append((offset, Token(token.kind, number_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "string":
                retval.append((offset, Token(token.kind, str_to_idx(token.raw_value),
                                             token.raw_value)))
            elif token.kind == "char" or token.kind == "char_continue":
                retval.append((offset, Token(token.kind, chr_to_idx(token.raw_value),
                                             token.raw_value)))
            else:
                retval.append((offset, token))
        return retval
