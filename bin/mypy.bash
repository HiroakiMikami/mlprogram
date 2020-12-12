#! /bin/bash

mypy --no-site-packages --ignore-missing-imports mlprogram/**/*.py test/**/*.py || exit 1
