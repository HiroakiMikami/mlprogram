#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python $PROJECT_ROOT/main.py src.valid \
	--valid $PROJECT_ROOT/dataset/toy/valid \
	--result $PROJECT_ROOT/result/toy \
	--output $PROJECT_ROOT/result/toy \
	--max-query-length 5 \
	--max-action-length 12 \
	--beam-size 1 \
    $@
