#! /bin/bash
set -u

test_type=${1:-"unit"}

pytest test || exit 1

if [ $test_type = "all" ]
then
    pytest -s -vv test_integration || exit 1
fi
