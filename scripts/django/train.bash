#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python $PROJECT_ROOT/main.py src.train \
	--train $PROJECT_ROOT/dataset/django/train \
	--test $PROJECT_ROOT/dataset/django/test \
	--output $PROJECT_ROOT/result/django \
    $@
