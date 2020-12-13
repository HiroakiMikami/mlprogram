#! /bin/bash

mypy \
    --no-site-packages \
    --ignore-missing-imports \
    --follow-imports normal \
    --allow-redefinition \
    --warn-redundant-casts \
    --warn-unused-ignores \
    mlprogram/**/*.py test/**/*.py || exit 1


# --diallow-any-unimported
# --disallow-any-expr
# --disallow-any-decorated
# --disallow-any-explicit
# --disallow-any-generics
# --disallow-untyped-calls
# --disallow-untyped-defs
# --disallow-subclassing-any
# --check-untyped-defs
# --disallow-untyped-decorators
# --no-implicit-optional
# --no-warn-no-return
# --warn-return-any

