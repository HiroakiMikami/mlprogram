#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python $PROJECT_ROOT/main.py src.train \
		--train $PROJECT_ROOT/dataset/toy/train \
		--test $PROJECT_ROOT/dataset/toy/test \
		--output $PROJECT_ROOT/result/toy \
		--max-query-length 5 \
		--max-action-length 12 \
		--beam-size 1 \
        $@
