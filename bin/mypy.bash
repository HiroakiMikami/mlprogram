#! /bin/bash

mypy \
    --no-site-packages \
    --ignore-missing-imports \
    --follow-imports normal \
    --allow-redefinition \
    --warn-redundant-casts \
    --warn-unused-ignores \
    --warn-return-any \
    --no-implicit-optional \
    --show-column-numbers \
    mlprogram/**/*.py test/**/*.py || exit 1


# --disallow-any-expr
# --disallow-any-decorated
# --disallow-any-explicit
# --disallow-any-generics
# --disallow-untyped-calls
# --disallow-untyped-defs
# --disallow-subclassing-any
# --disallow-untyped-decorators
# --disallow-any-unimported
# --check-untyped-defs
# --warn-unreachable
