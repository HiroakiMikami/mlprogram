#! /bin/bash
set -u

PROJECT_ROOT=$(dirname $0)/../../

python ${PROJECT_ROOT}/main.py src.toy.create_dataset --directory ${PROJECT_ROOT}/dataset/toy
