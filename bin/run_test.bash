#! /bin/bash
set -u

test_type=${1:-"unit"}

pytest test

if [ $test_type = "all" ]
then
    pytest --workers auto -s test_integration || exit 1
fi
