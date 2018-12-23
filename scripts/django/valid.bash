#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python $PROJECT_ROOT/main.py src.valid \
	--valid $PROJECT_ROOT/dataset/django/valid \
    --result $PROJECT_ROOT/result/django \
    --output $PROJECT_ROOT/result/django \
    $@
