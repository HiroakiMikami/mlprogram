.PHONY: check format mypy test test_module test_configs test_integration

check:
	black . --diff || exit 1
	black configs --diff || exit 1
	isort . --diff || exit 1
	python -m flatpython $(shell find ./examples -name "*.py") || exit 1

format:
	black .
	black configs
	isort .
	python -m flatpython $(shell find ./examples -name "*.py") || exit 1

mypy:
	mypy --no-site-packages \
	     --ignore-missing-imports \
	     --follow-imports normal \
	     --allow-redefinition \
	     --warn-redundant-casts \
	     --warn-return-any \
	     --no-implicit-optional \
	     --show-column-numbers \
	     --warn-unreachable \
	    mlprogram/**/*.py test/**/*.py || exit 1
# --warn-unused-ignores
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

test: test_module test_configs test_integration

test_module:
	pytest test/mlprogram || exit 1

test_configs:
	pytest test/configs || exit 1

test_integration:
	pytest -s -vv test_integration || exit 1
